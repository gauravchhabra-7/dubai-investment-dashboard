"""
Project Analysis Module - Transaction-Based Adaptive CAGR Implementation
=======================================================================

This module implements adaptive CAGR calculation for real estate projects based on:
- Age-dependent window lengths (31/92/183 days)
- Median price calculations within windows
- Business quality flags and validation rules
- Full compatibility with existing dashboard interfaces

Business Rules:
- Age ≥ 547 days: 183-day windows, ≥5 deeds each
- 183 ≤ age < 547: 92-day windows, ≥3 deeds each, 30-day gap required
- Age < 183 days: 31-day windows, ≥2 deeds each
- Price filter: 100 < meter_sale_price_sqft < 10,000 AED/ft²
- Volume calculation: Sum of actual_worth in recent window
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import get_path

# Global cache for performance optimization
_all_projects_cache = None
_transaction_data_cache = None

# =============================================================================
# CORE DATA LOADING AND PREPROCESSING
# =============================================================================

def load_transaction_data():
    """
    Load and cache transaction data for project analysis
    
    Returns:
        pd.DataFrame: Cleaned transaction data with proper dtypes
    """
    global _transaction_data_cache
    
    if _transaction_data_cache is not None:
        return _transaction_data_cache.copy()
    
    try:
        # Load data using config path
        data_path = get_path('project_txn_data')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Transaction data not found at: {data_path}")
        
        print(f"Loading transaction data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Smart date parsing
        df['instance_date'] = pd.to_datetime(df['instance_date'], errors='coerce')
        
        # Remove records with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['instance_date'])
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df)} records with invalid dates")
        
        # Price filtering (using meter_sale_price_sqft as specified)
        price_col = 'meter_sale_price_sqft'
        if price_col not in df.columns:
            raise ValueError(f"Required price column '{price_col}' not found in data")
        
        # Apply price range filter
        initial_count = len(df)
        df = df[(df[price_col] > 100) & (df[price_col] < 10000)].copy()
        print(f"Applied price filter (100-10,000): {initial_count} → {len(df)} records")
        
        # Ensure required columns exist
        required_cols = [
            'project_number_int', 'project_name_en', 'developer_name', 
            'area_name_en', 'instance_date', price_col, 'actual_worth'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove records with missing critical data
        df = df.dropna(subset=required_cols)
        
        # Sort by project and date for efficient processing
        df = df.sort_values(['project_number_int', 'instance_date'])
        
        # Cache the cleaned data
        _transaction_data_cache = df
        print(f"Successfully loaded and cached {len(df):,} transaction records")
        
        return df.copy()
        
    except Exception as e:
        print(f"Error loading transaction data: {e}")
        raise

def get_project_age_category(start_date, end_date):
    """
    Determine project age category and corresponding window parameters
    
    Args:
        start_date (datetime): Project start date
        end_date (datetime): Project end date
        
    Returns:
        dict: Window parameters based on project age
    """
    age_days = (end_date - start_date).days
    
    if age_days >= 547:  # ≥ 18 months
        return {
            'age_days': age_days,
            'window_length': 183,
            'min_deeds': 5,
            'category': 'mature',
            'requires_gap': False
        }
    elif age_days >= 183:  # 6-18 months
        return {
            'age_days': age_days,
            'window_length': 92,
            'min_deeds': 3,
            'category': 'medium',
            'requires_gap': True
        }
    else:  # < 6 months
        return {
            'age_days': age_days,
            'window_length': 31,
            'min_deeds': 2,
            'category': 'young',
            'requires_gap': False
        }

def calculate_window_bounds(project_txns, window_params):
    """
    Calculate first and recent window bounds with gap checking
    
    Args:
        project_txns (pd.DataFrame): Project transactions sorted by date
        window_params (dict): Window parameters from get_project_age_category
        
    Returns:
        dict: Window bounds and validation results
    """
    if len(project_txns) == 0:
        return None
    
    start_date = project_txns['instance_date'].min()
    end_date = project_txns['instance_date'].max()
    window_len = window_params['window_length']
    
    # Calculate initial window bounds
    first_window_start = start_date
    first_window_end = start_date + timedelta(days=window_len)
    
    recent_window_start = end_date - timedelta(days=window_len)
    recent_window_end = end_date
    
    # Check for gap requirement (medium-age projects)
    if window_params['requires_gap']:
        gap_days = (recent_window_start - first_window_end).days
        
        if gap_days < 30:
            # Push recent window forward to maintain 30-day gap
            adjustment_days = 30 - gap_days
            recent_window_start = recent_window_start + timedelta(days=adjustment_days)
            recent_window_end = recent_window_end + timedelta(days=adjustment_days)
            
            # Ensure recent window doesn't exceed project end
            if recent_window_end > end_date:
                recent_window_end = end_date
                recent_window_start = recent_window_end - timedelta(days=window_len)
    
    return {
        'first_window_start': first_window_start,
        'first_window_end': first_window_end,
        'recent_window_start': recent_window_start,
        'recent_window_end': recent_window_end,
        'window_length_days': window_len
    }

def get_window_transactions(project_txns, window_bounds):
    """
    Extract transactions within specific windows
    
    Args:
        project_txns (pd.DataFrame): Project transactions
        window_bounds (dict): Window bounds from calculate_window_bounds
        
    Returns:
        tuple: (first_window_txns, recent_window_txns)
    """
    first_window = project_txns[
        (project_txns['instance_date'] >= window_bounds['first_window_start']) &
        (project_txns['instance_date'] <= window_bounds['first_window_end'])
    ].copy()
    
    recent_window = project_txns[
        (project_txns['instance_date'] >= window_bounds['recent_window_start']) &
        (project_txns['instance_date'] <= window_bounds['recent_window_end'])
    ].copy()
    
    return first_window, recent_window

def calculate_window_median_price(window_txns):
    """
    Calculate median price for a window
    
    Args:
        window_txns (pd.DataFrame): Transactions in the window
        
    Returns:
        float: Median price per sqft or None if insufficient data
    """
    if len(window_txns) == 0:
        return None
    
    return window_txns['meter_sale_price_sqft'].median()

def calculate_cagr(first_price, recent_price, first_midpoint, recent_midpoint):
    """
    Calculate CAGR between two price points
    
    Args:
        first_price (float): Initial price
        recent_price (float): Recent price
        first_midpoint (datetime): Midpoint of first window
        recent_midpoint (datetime): Midpoint of recent window
        
    Returns:
        float: CAGR as percentage or None if invalid
    """
    if first_price is None or recent_price is None or first_price <= 0:
        return None
    
    # Calculate time difference in days
    delta_days = (recent_midpoint - first_midpoint).days
    
    if delta_days <= 0:
        return None
    
    # Convert to annual growth rate
    years = delta_days / 365.25
    if years <= 0:
        return None
    
    try:
        cagr = ((recent_price / first_price) ** (1 / years) - 1) * 100
        return cagr
    except (ZeroDivisionError, ValueError, OverflowError):
        return None

# =============================================================================
# MAIN PROJECT METRICS CALCULATION
# =============================================================================

def calculate_project_metrics(txn_df, room_type_filter='All'):
    """
    Calculate adaptive CAGR metrics for all projects
    
    Args:
        txn_df (pd.DataFrame): Transaction data
        room_type_filter (str): Room type filter ('All' or specific room type)
        
    Returns:
        pd.DataFrame: Project metrics with adaptive CAGR calculations
    """
    print(f"Calculating project metrics for {len(txn_df):,} transactions...")
    
    # Apply room type filter if specified
    if room_type_filter != 'All' and 'rooms_en' in txn_df.columns:
        txn_df = txn_df[txn_df['rooms_en'] == room_type_filter].copy()
        print(f"Filtered to {len(txn_df):,} transactions for room type: {room_type_filter}")
    
    project_results = []
    
    # Group by project
    for project_id, project_txns in txn_df.groupby('project_number_int'):
        try:
            if len(project_txns) < 2:
                # Handle single transaction case
                project_info = project_txns.iloc[0]
                result = {
                    'project_number_int': project_id,
                    'project_name_en': project_info.get('project_name_en', f'Project {project_id}'),
                    'developer_name': project_info.get('developer_name', 'Unknown'),
                    'area_name_en': project_info.get('area_name_en', 'Unknown'),
                    'property_type_en': project_info.get('property_type_en', 'Unknown'),
                    'age_days': 0,
                    'first_deeds': 1,
                    'recent_deeds': 1,
                    'window_len_days': 0,
                    'cagr': 0.0,
                    'launch_price_sqft': project_info['meter_sale_price_sqft'],
                    'recent_price_sqft': project_info['meter_sale_price_sqft'],
                    'is_early_launch': True,
                    'is_thin': True,
                    'needs_review': False,
                    'first_window_start': project_info['instance_date'],
                    'first_window_end': project_info['instance_date'],
                    'recent_window_start': project_info['instance_date'],
                    'recent_window_end': project_info['instance_date'],
                    'recent_aed_volume': project_info.get('actual_worth', 0),
                    'transaction_count': 1,
                    'single_transaction': True
                }
                project_results.append(result)
                continue
            
            # Sort transactions by date
            project_txns = project_txns.sort_values('instance_date')
            
            # Get project metadata
            project_info = project_txns.iloc[0]
            start_date = project_txns['instance_date'].min()
            end_date = project_txns['instance_date'].max()
            
            # Determine window parameters based on project age
            window_params = get_project_age_category(start_date, end_date)
            
            # Calculate window bounds
            window_bounds = calculate_window_bounds(project_txns, window_params)
            if window_bounds is None:
                continue
            
            # Extract window transactions
            first_window, recent_window = get_window_transactions(project_txns, window_bounds)
            
            # Check minimum deed requirements
            first_deeds = len(first_window)
            recent_deeds = len(recent_window)
            min_deeds = window_params['min_deeds']
            
            # Calculate window median prices
            first_price = calculate_window_median_price(first_window)
            recent_price = calculate_window_median_price(recent_window)
            
            # Calculate window midpoints for CAGR calculation
            first_midpoint = window_bounds['first_window_start'] + timedelta(
                days=window_params['window_length'] / 2
            )
            recent_midpoint = window_bounds['recent_window_start'] + timedelta(
                days=window_params['window_length'] / 2
            )
            
            # Calculate CAGR
            cagr = calculate_cagr(first_price, recent_price, first_midpoint, recent_midpoint)
            
            # Calculate recent window volume
            recent_volume = recent_window['actual_worth'].sum() if len(recent_window) > 0 else 0
            
            # Apply business flags
            is_early_launch = window_params['age_days'] < 270  # < 9 months
            is_thin = (first_deeds < min_deeds) or (recent_deeds < min_deeds)
            needs_review = cagr is not None and abs(cagr) > 400
            
            # Handle extreme CAGR values
            if needs_review:
                cagr = 0.0
            
            # Create project result
            result = {
                'project_number_int': project_id,
                'project_name_en': project_info.get('project_name_en', f'Project {project_id}'),
                'developer_name': project_info.get('developer_name', 'Unknown'),
                'area_name_en': project_info.get('area_name_en', 'Unknown'),
                'property_type_en': project_info.get('property_type_en', 'Unknown'),
                'age_days': window_params['age_days'],
                'first_deeds': first_deeds,
                'recent_deeds': recent_deeds,
                'window_len_days': window_params['window_length'],
                'cagr': cagr if cagr is not None else 0.0,
                'launch_price_sqft': first_price if first_price is not None else 0.0,
                'recent_price_sqft': recent_price if recent_price is not None else 0.0,
                'is_early_launch': is_early_launch,
                'is_thin': is_thin,
                'needs_review': needs_review,
                'first_window_start': window_bounds['first_window_start'],
                'first_window_end': window_bounds['first_window_end'],
                'recent_window_start': window_bounds['recent_window_start'],
                'recent_window_end': window_bounds['recent_window_end'],
                'recent_aed_volume': recent_volume,
                'transaction_count': len(project_txns),
                'single_transaction': False
            }
            
            project_results.append(result)
            
        except Exception as e:
            print(f"Error processing project {project_id}: {e}")
            continue
    
    if not project_results:
        print("No valid projects found for analysis")
        return pd.DataFrame()
    
    # Create DataFrame
    results_df = pd.DataFrame(project_results)
    
    print(f"Successfully calculated metrics for {len(results_df)} projects")
    print(f"Early launch projects: {results_df['is_early_launch'].sum()}")
    print(f"Thin data projects: {results_df['is_thin'].sum()}")
    print(f"Projects needing review: {results_df['needs_review'].sum()}")
    
    return results_df

# =============================================================================
# FILTERING AND DATA PREPARATION
# =============================================================================

def apply_deed_filters(txn_df, property_type='All', area='All', developer='All', room_type='All'):
    """
    Apply filters at the transaction level
    
    Args:
        txn_df (pd.DataFrame): Transaction data
        property_type (str): Property type filter
        area (str): Area filter
        developer (str): Developer filter
        room_type (str): Room type filter
        
    Returns:
        pd.DataFrame: Filtered transaction data
    """
    filtered_df = txn_df.copy()
    
    if property_type != 'All' and 'property_type_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['property_type_en'] == property_type]
    
    if area != 'All' and 'area_name_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['area_name_en'] == area]
    
    if developer != 'All' and 'developer_name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['developer_name'] == developer]
    
    if room_type != 'All' and 'rooms_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rooms_en'] == room_type]
    
    return filtered_df

def add_compatibility_columns(project_df):
    """
    Add columns for backward compatibility with existing dashboard code
    
    Args:
        project_df (pd.DataFrame): Project metrics DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with compatibility columns added
    """
    if len(project_df) == 0:
        return project_df
    
    # Add compatibility columns
    project_df = project_df.copy()
    
    # Legacy naming compatibility
    project_df['price_sqft_cagr'] = project_df['cagr']
    project_df['has_quality_concerns'] = project_df['is_thin'] | project_df['needs_review']
    project_df['is_illiquid'] = project_df['is_thin']  # Alias for dashboard compatibility
    project_df['duration_years'] = project_df['age_days'] / 365.25
    
    return project_df

def apply_quality_filters(project_df, has_room_filter=False):
    """
    Apply quality filters and sort results
    
    Args:
        project_df (pd.DataFrame): Project metrics DataFrame
        has_room_filter (bool): Whether room type filter was applied
        
    Returns:
        pd.DataFrame: Filtered and sorted DataFrame
    """
    if len(project_df) == 0:
        return project_df
    
    # Sort by CAGR descending
    project_df = project_df.sort_values('cagr', ascending=False)
    
    return project_df

# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def get_all_project_metrics(force_refresh=False):
    """
    Get or calculate all project metrics (cached for performance)
    
    Args:
        force_refresh (bool): Force recalculation even if cached
        
    Returns:
        pd.DataFrame: All project metrics
    """
    global _all_projects_cache
    
    if _all_projects_cache is not None and not force_refresh:
        print("Using cached project metrics")
        return _all_projects_cache.copy()
    
    print("Calculating all project metrics (this may take a moment)...")
    
    # Load transaction data
    txn_df = load_transaction_data()
    
    # Calculate metrics for ALL projects
    all_metrics = calculate_project_metrics(txn_df, 'All')
    
    # Add compatibility columns
    all_metrics = add_compatibility_columns(all_metrics)
    
    # Apply quality filters
    all_metrics = apply_quality_filters(all_metrics, False)
    
    # Cache the results
    _all_projects_cache = all_metrics
    print(f"Cached metrics for {len(all_metrics)} projects")
    
    return all_metrics.copy()

def prepare_project_data(df=None, property_type='All', area='All', developer='All', room_type='All'):
    """
    Main interface function for project data preparation with filtering
    
    Args:
        df (pd.DataFrame, optional): Not used - kept for compatibility
        property_type (str): Property type filter
        area (str): Area filter  
        developer (str): Developer filter
        room_type (str): Room type filter
        
    Returns:
        pd.DataFrame: Enhanced project dataset with CAGR metrics and flags
    """
    print(f"Processing projects with filters: Property={property_type}, Area={area}, Developer={developer}, Room={room_type}")
    
    # If only developer/area filters (no property/room filters), use cached data
    if property_type == 'All' and room_type == 'All':
        # Get pre-calculated metrics
        all_metrics = get_all_project_metrics()
        
        # Apply developer/area filters at project level
        filtered_metrics = all_metrics.copy()
        
        if developer != 'All':
            filtered_metrics = filtered_metrics[filtered_metrics['developer_name'] == developer]
            print(f"Filtered to {len(filtered_metrics)} projects by developer: {developer}")
            
        if area != 'All':
            filtered_metrics = filtered_metrics[filtered_metrics['area_name_en'] == area]
            print(f"Filtered to {len(filtered_metrics)} projects in area: {area}")
        
        return filtered_metrics
    
    else:
        # For property/room filters, we need to recalculate
        print("Property/Room filter detected - recalculating metrics...")
        
        # Load transaction data
        txn_df = load_transaction_data()
        
        # Apply deed-level filters
        filtered_txn_df = apply_deed_filters(txn_df, property_type, area, developer, room_type)
        
        if len(filtered_txn_df) == 0:
            print("No transactions match the selected filters")
            return pd.DataFrame()
        
        # Calculate metrics
        project_metrics = calculate_project_metrics(filtered_txn_df, room_type)
        
        # Add compatibility columns
        project_metrics = add_compatibility_columns(project_metrics)
        
        # Apply quality filters
        project_metrics = apply_quality_filters(project_metrics, room_type != 'All')
        
        return project_metrics

def filter_project_data(df, property_type='All', area='All', developer='All', room_type='All'):
    """
    Filter project data - wrapper for prepare_project_data for compatibility
    
    Args:
        df (pd.DataFrame): Project data (ignored - recalculated)
        property_type (str): Property type filter
        area (str): Area filter
        developer (str): Developer filter
        room_type (str): Room type filter
        
    Returns:
        pd.DataFrame: Filtered project data
    """
    return prepare_project_data(df, property_type, area, developer, room_type)

# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_to_area_level(project_df):
    """
    Create area-level aggregations with both simple and weighted averages
    
    Args:
        project_df (pd.DataFrame): Project-level metrics
        
    Returns:
        pd.DataFrame: Area-level aggregations matching output schema
    """
    if len(project_df) == 0:
        return pd.DataFrame()
    
    # Basic aggregations
    area_agg = project_df.groupby('area_name_en').agg({
        'cagr': ['count', 'mean'],
        'transaction_count': 'sum',
        'is_early_launch': 'sum',
        'is_thin': 'sum',
        'needs_review': 'sum',
        'recent_aed_volume': 'median'
    }).round(2)
    
    # Flatten column names
    area_agg.columns = ['project_count', 'simple_avg_cagr', 'total_transactions', 
                        'early_count', 'thin_count', 'review_count', 'median_aed_volume']
    
    # Calculate weighted average CAGR
    def weighted_cagr(group):
        weights = group['transaction_count']
        if weights.sum() > 0:
            return np.average(group['cagr'], weights=weights)
        return np.nan
    
    weighted_avg = project_df.groupby('area_name_en').apply(weighted_cagr).round(2)
    area_agg['weighted_avg_cagr'] = weighted_avg
    
    # Calculate share of early/thin projects
    area_agg['share_early_thin'] = ((area_agg['early_count'] + area_agg['thin_count']) / 
                                   area_agg['project_count'] * 100).round(1)
    
    # Count excluded projects (those needing review)
    area_agg['excluded_count'] = area_agg['review_count']
    
    # Reset index and select final columns
    area_agg = area_agg.reset_index()
    
    # Match output schema
    final_columns = [
        'area_name_en', 'project_count', 'simple_avg_cagr', 'weighted_avg_cagr',
        'share_early_thin', 'median_aed_volume', 'excluded_count'
    ]
    
    area_agg = area_agg[final_columns].sort_values('weighted_avg_cagr', ascending=False)
    
    return area_agg

def aggregate_to_developer_level(project_df):
    """
    Create developer-level aggregations with portfolio analysis
    
    Args:
        project_df (pd.DataFrame): Project-level metrics
        
    Returns:
        pd.DataFrame: Developer-level aggregations matching output schema
    """
    if len(project_df) == 0:
        return pd.DataFrame()
    
    # Basic aggregations
    dev_agg = project_df.groupby('developer_name').agg({
        'cagr': ['count', 'mean', 'std'],
        'transaction_count': 'sum',
        'is_early_launch': 'sum',
        'is_thin': 'sum',
        'needs_review': 'sum'
    }).round(2)
    
    # Flatten column names
    dev_agg.columns = ['active_projects', 'portfolio_mean_cagr', 'cagr_std',
                       'total_transactions', 'early_count', 'thin_count', 'review_count']
    
    # Calculate weighted portfolio CAGR
    def weighted_cagr(group):
        weights = group['transaction_count']
        if weights.sum() > 0:
            return np.average(group['cagr'], weights=weights)
        return np.nan
    
    weighted_avg = project_df.groupby('developer_name').apply(weighted_cagr).round(2)
    dev_agg['portfolio_weighted_cagr'] = weighted_avg
    
    # Calculate stability score (inverse of coefficient of variation)
    dev_agg['stability_score'] = np.where(
        dev_agg['cagr_std'] > 0,
        (100 / (1 + dev_agg['cagr_std'] / abs(dev_agg['portfolio_mean_cagr']))).round(1),
        100.0
    )
    
    # Calculate share of early/thin projects
    dev_agg['early_thin_share'] = ((dev_agg['early_count'] + dev_agg['thin_count']) / 
                                  dev_agg['active_projects'] * 100).round(1)
    
    # Count excluded projects
    dev_agg['excluded_count'] = dev_agg['review_count']
    
    # Reset index and select final columns
    dev_agg = dev_agg.reset_index()
    
    # Match output schema
    final_columns = [
        'developer_name', 'active_projects', 'portfolio_mean_cagr', 
        'portfolio_weighted_cagr', 'stability_score', 'early_thin_share', 'excluded_count'
    ]
    
    dev_agg = dev_agg[final_columns].sort_values('portfolio_weighted_cagr', ascending=False)
    
    return dev_agg

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_empty_figure(message="No data available"):
    """Create empty figure with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
        plot_bgcolor='white'
    )
    return fig

# REPLACE these visualization functions in project_analysis.py:

def create_individual_project_analysis(df, title="Top 20 Projects by CAGR"):
    """
    Create horizontal bar chart for individual project performance - TOP 20 PROJECTS
    
    Args:
        df (pd.DataFrame): Project metrics data
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Interactive bar chart
    """
    if len(df) == 0:
        return create_empty_figure("No projects available for analysis")
    
    # Take top 20 projects by CAGR (changed from 25)
    display_df = df.nlargest(20, 'cagr').copy()
    
    # Reverse order for proper display (top performers at top)
    display_df = display_df.iloc[::-1].copy()
    
    # Create project labels
    display_df['project_label'] = display_df.apply(
        lambda x: f"{x['project_name_en'][:35]}..." if len(x['project_name_en']) > 35 
        else x['project_name_en'], axis=1
    )
    
    # Define colors based on flags
    colors = []
    for _, row in display_df.iterrows():
        if row['single_transaction']:
            colors.append('lightgray')  # Single transaction
        elif row['needs_review']:
            colors.append('red')  # Review needed
        elif row['is_thin']:
            colors.append('orange')  # Thin data
        elif row['is_early_launch']:
            colors.append('lightblue')  # Early launch
        else:
            colors.append('darkblue')  # Normal
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=display_df['project_label'],
        x=display_df['cagr'],
        orientation='h',
        marker_color=colors,
        text=[f"{val:.1f}%" for val in display_df['cagr']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "CAGR: %{x:.1f}%<br>" +
            "Developer: %{customdata[0]}<br>" +
            "Area: %{customdata[1]}<br>" +
            "Launch Price: AED %{customdata[2]:,.0f}/ft²<br>" +
            "Recent Price: AED %{customdata[3]:,.0f}/ft²<br>" +
            "Transaction Count: %{customdata[4]:,}<br>" +
            "Project Age: %{customdata[5]:.0f} days<br>" +
            "<extra></extra>"
        ),
        customdata=display_df[['developer_name', 'area_name_en', 'launch_price_sqft', 
                              'recent_price_sqft', 'transaction_count', 'age_days']].values
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Annual Price Appreciation (CAGR %)",
        yaxis_title="Project",
        height=max(700, len(display_df) * 30 + 150),  # Increased height for 20 projects
        showlegend=False,
        margin=dict(l=350, r=150, t=80, b=80),  # Increased left margin
        clickmode='event+select'
    )
    
    return fig

def create_area_performance_comparison(df, title="Top 20 Areas by Volume-Weighted CAGR"):
    """
    Create horizontal OVERLAY bar chart for area performance comparison - TOP 20 AREAS
    Simple average bar behind, volume-weighted bar in front
    """
    if len(df) == 0:
        return create_empty_figure("No data available for area comparison")
    
    # Get area aggregations
    area_stats = aggregate_to_area_level(df)
    
    # Filter areas with at least 3 projects
    area_stats = area_stats[area_stats['project_count'] >= 1].copy()
    
    if len(area_stats) == 0:
        return create_empty_figure("Insufficient data for area comparison (need ≥3 projects per area)")
    
    # Sort and take top 20 (changed from 15)
    top_areas = area_stats.nlargest(20, 'weighted_avg_cagr').iloc[::-1].copy()
    
    # Create labels
    top_areas['area_label'] = top_areas.apply(
        lambda x: f"{x['area_name_en']} ({x['project_count']} projects)", axis=1
    )
    
    # Create figure with OVERLAY bars (not grouped)
    fig = go.Figure()
    
    # Simple average bars (BEHIND - lighter color)
    fig.add_trace(go.Bar(
        y=top_areas['area_label'],
        x=top_areas['simple_avg_cagr'],
        name='Simple Average',
        orientation='h',
        marker_color='lightblue',
        marker_line=dict(color='rgba(50,50,50,0.8)', width=0.5),
        text=[f"{val:.1f}%" for val in top_areas['simple_avg_cagr']],
        textposition='outside',
        opacity=0.7  # Make it more transparent since it's behind
    ))
    
    # Volume-weighted bars (FRONT - darker color) 
    fig.add_trace(go.Bar(
        y=top_areas['area_label'],
        x=top_areas['weighted_avg_cagr'],
        name='Volume-Weighted',
        orientation='h',
        marker_color='darkblue',
        marker_line=dict(color='rgba(20,20,20,0.8)', width=1),
        text=[f"{val:.1f}%" for val in top_areas['weighted_avg_cagr']],
        textposition='outside',
        opacity=0.9  # More opaque since it's in front
    ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Update layout for OVERLAY (not grouped)
    fig.update_layout(
        title=title,
        xaxis_title="Average CAGR (%)",
        yaxis_title="Area",
        height=max(700, len(top_areas) * 30 + 150),  # Increased height for 20 areas
        barmode='group',  # CHANGED from 'group' to 'overlay'
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=350, r=120, t=100, b=80)  # Increased left margin
    )
    
    return fig

def create_developer_track_record_analysis(df, title="Top 20 Developers by Volume-Weighted CAGR"):
    """
    Create horizontal OVERLAY bar chart for developer track record - TOP 20 DEVELOPERS
    """
    if len(df) == 0:
        return create_empty_figure("No data available for developer comparison")
    
    # Get developer aggregations
    dev_stats = aggregate_to_developer_level(df)
    
    # Filter developers with at least 2 projects
    dev_stats = dev_stats[dev_stats['active_projects'] >= 1].copy()
    
    if len(dev_stats) == 0:
        return create_empty_figure("Insufficient data for developer comparison (need ≥2 projects per developer)")
    
    # Sort and take top 20 (changed from 15)
    top_devs = dev_stats.nlargest(20, 'portfolio_weighted_cagr').iloc[::-1].copy()
    
    # Create labels
    top_devs['dev_label'] = top_devs.apply(
        lambda x: f"{x['developer_name'][:30]}..." if len(x['developer_name']) > 30 
        else x['developer_name'], axis=1
    )
    top_devs['dev_label'] = top_devs['dev_label'] + " (" + top_devs['active_projects'].astype(str) + ")"
    
    # Create figure with overlay bars
    fig = go.Figure()
    
    # Simple average bars (BEHIND - lighter color)
    fig.add_trace(go.Bar(
        y=top_devs['dev_label'],
        x=top_devs['portfolio_mean_cagr'],
        name='Simple Average',
        orientation='h',
        marker_color='lightgreen',
        marker_line=dict(color='rgba(50,50,50,0.8)', width=0.5),
        text=[f"{val:.1f}%" for val in top_devs['portfolio_mean_cagr']],
        textposition='outside',
        opacity=0.7  # More transparent since it's behind
    ))
    
    # Volume-weighted bars (FRONT - darker color)
    fig.add_trace(go.Bar(
        y=top_devs['dev_label'],
        x=top_devs['portfolio_weighted_cagr'],
        name='Volume-Weighted',
        orientation='h',
        marker_color='darkgreen',
        marker_line=dict(color='rgba(20,20,20,0.8)', width=1),
        text=[f"{val:.1f}%" for val in top_devs['portfolio_weighted_cagr']],
        textposition='outside',
        opacity=0.9  # More opaque since it's in front
    ))
    
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Update layout for overlay
    fig.update_layout(
        title=title,
        xaxis_title="Portfolio-Weighted CAGR (%)",
        yaxis_title="Developer",
        height=max(700, len(top_devs) * 30 + 150),  # Increased height for 20 developers
        barmode='group',  # OVERLAY bars
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=400, r=120, t=100, b=80)  # Increased left margin for longer labels
    )
    
    return fig

# =============================================================================
# DASHBOARD COMPATIBILITY FUNCTIONS
# =============================================================================

def prepare_insights_metadata(df, filters=None, peer_group_info=None):
    """
    Prepare metadata dictionary for the insights system
    
    Args:
        df (pd.DataFrame): Project metrics data
        filters (dict): Applied filters
        peer_group_info (dict): Peer group information
        
    Returns:
        dict: Metadata for insights generation
    """
    if len(df) == 0:
        return {
            'data_quality': 'none',
            'coverage_pct': 0.0,
            'estimation_method': 'transaction_based',
            'total_projects': 0
        }
    
    # Calculate data quality based on flags
    total_projects = len(df)
    good_quality = len(df[~(df['is_thin'] | df['needs_review'])])
    quality_pct = (good_quality / total_projects * 100) if total_projects > 0 else 0
    
    if quality_pct >= 70:
        data_quality = 'high'
    elif quality_pct >= 40:
        data_quality = 'medium'
    else:
        data_quality = 'low'
    
    metadata = {
        'data_quality': data_quality,
        'coverage_pct': quality_pct,
        'estimation_method': 'adaptive_cagr',
        'total_projects': total_projects,
        'total_transactions': df['transaction_count'].sum(),
        'median_age_days': df['age_days'].median(),
        'early_launch_count': df['is_early_launch'].sum(),
        'thin_data_count': df['is_thin'].sum(),
        'review_count': df['needs_review'].sum()
    }
    
    if peer_group_info:
        metadata.update(peer_group_info)
    
    return metadata

def get_peer_group_info(df, property_type='All', area='All'):
    """
    Get peer group information for context
    
    Args:
        df (pd.DataFrame): Project metrics data
        property_type (str): Property type filter
        area (str): Area filter
        
    Returns:
        dict: Peer group information
    """
    if len(df) == 0:
        return {}
    
    peer_info = {
        'peer_group_size': len(df),
        'peer_median_cagr': df['cagr'].median(),
        'peer_avg_transactions': df['transaction_count'].mean(),
    }
    
    if property_type != 'All':
        peer_info['property_type_focus'] = property_type
    
    if area != 'All':
        peer_info['area_focus'] = area
    
    return peer_info

# =============================================================================
# ROOM TYPE ANALYSIS (FOR DRILL-DOWN)
# =============================================================================

def get_project_room_breakdown(project_id, property_type='All', area='All', developer='All'):
    """
    Get room type breakdown for a specific project (for drill-down functionality)
    
    Args:
        project_id (int): Project number
        property_type (str): Property type filter
        area (str): Area filter
        developer (str): Developer filter
        
    Returns:
        pd.DataFrame: Room type breakdown for the project
    """
    try:
        # Load transaction data
        txn_df = load_transaction_data()
        
        # Filter to specific project
        project_txns = txn_df[txn_df['project_number_int'] == project_id].copy()
        
        # Apply other filters (but NOT room type)
        if property_type != 'All':
            project_txns = project_txns[project_txns['property_type_en'] == property_type]
        if area != 'All':
            project_txns = project_txns[project_txns['area_name_en'] == area]
        if developer != 'All':
            project_txns = project_txns[project_txns['developer_name'] == developer]
        
        if len(project_txns) == 0 or 'rooms_en' not in project_txns.columns:
            return pd.DataFrame()
        
        # Calculate metrics for each room type
        room_results = []
        
        for room_type in project_txns['rooms_en'].unique():
            room_txns = project_txns[project_txns['rooms_en'] == room_type].copy()
            
            if len(room_txns) < 2:
                continue
            
            # Calculate basic metrics using our adaptive system
            room_metrics = calculate_project_metrics(room_txns, room_type)
            
            if len(room_metrics) > 0:
                room_record = room_metrics.iloc[0].copy()
                room_record['room_type'] = room_type
                room_results.append(room_record)
        
        if room_results:
            return pd.DataFrame(room_results).sort_values('cagr', ascending=False)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in room breakdown analysis: {e}")
        return pd.DataFrame()