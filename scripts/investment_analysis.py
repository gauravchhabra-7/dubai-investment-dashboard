import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, dash_table

def get_time_horizon_display(time_horizon):
    """Convert time horizon parameter to display text"""
    if time_horizon == "short_term_growth":
        return "Short-term (1 year) Growth"
    elif time_horizon == "medium_term_growth":
        return "Medium-term (3 year) Growth"
    elif time_horizon == "long_term_cagr":
        return "Long-term (5 year) CAGR"
    else:
        return "Recent Growth"

def select_best_growth_column(df, requested_horizon=None, min_coverage_pct=5):
    """
    Intelligently select the best growth column based on data coverage
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        requested_horizon (str, optional): Specifically requested time horizon
        min_coverage_pct (float): Minimum percentage of non-null values required (default: 15%)
        
    Returns:
        tuple: (selected_column, coverage_pct, data_quality)
               where data_quality is "high", "medium", or "low"
    """
    print(f"Selecting best growth column (requested: {requested_horizon})")
    
    # 1. Check direct growth columns
    growth_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('growth_')]
    
    # Include other known growth metrics
    growth_columns.extend([col for col in df.columns if col in [
        'short_term_growth', 'medium_term_growth', 'long_term_cagr', 'recent_growth'
    ]])
    
    # Remove duplicates while preserving order
    growth_columns = list(dict.fromkeys(growth_columns))
    
    if not growth_columns:
        print("No growth columns found in dataframe")
        return None, 0, "none"
    
    # Create a dictionary of column coverage
    column_coverage = {}
    for col in growth_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            coverage_pct = (non_null_count / len(df)) * 100
            column_coverage[col] = coverage_pct
            print(f"  - {col}: {coverage_pct:.1f}% coverage ({non_null_count}/{len(df)} rows)")
    
    # Add after column_coverage dictionary is created
    for col, coverage in column_coverage.items():
        print(f"DEBUG TIME FILTER: Column {col} has {coverage:.1f}% coverage")
    print(f"DEBUG TIME FILTER: Requested horizon: {requested_horizon}, min coverage: {min_coverage_pct}%")
    
    # Sort by coverage (highest first)
    sorted_columns = sorted(column_coverage.items(), key=lambda x: x[1], reverse=True)
    
    # 2. Check if requested horizon has adequate coverage
    if requested_horizon and requested_horizon in column_coverage:
        coverage = column_coverage[requested_horizon]
        if coverage >= min_coverage_pct:
            data_quality = "high" if coverage >= 50 else "medium" if coverage >= 25 else "low"
            print(f"Using requested horizon {requested_horizon} with {coverage:.1f}% coverage (quality: {data_quality})")
            return requested_horizon, coverage, data_quality
        else:
            print(f"Requested horizon {requested_horizon} has insufficient coverage ({coverage:.1f}%)")

    # Add right after selecting the column
    print(f"DEBUG TIME FILTER: Selected column: {col}, with coverage: {coverage_pct}%")
    print(f"DEBUG TIME FILTER: Was requested column used? {col == requested_horizon}")
    
    # 3. Find the best growth column with adequate coverage
    for col, coverage in sorted_columns:
        if coverage >= min_coverage_pct:
            data_quality = "high" if coverage >= 50 else "medium" if coverage >= 25 else "low"
            print(f"Selected {col} with {coverage:.1f}% coverage (quality: {data_quality})")
            return col, coverage, data_quality
    
    # 4. Check if we can calculate growth from year columns
    years = sorted([col for col in df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025])
    if len(years) >= 2:
        latest_year = years[-1]
        prev_year = years[-2]
        
        # Calculate available data for these years
        latest_coverage = (df[latest_year].notna().sum() / len(df)) * 100
        prev_coverage = (df[prev_year].notna().sum() / len(df)) * 100
        
        # Combined coverage (both years need data for growth calculation)
        combined_coverage = (df[latest_year].notna() & df[prev_year].notna()).sum() / len(df) * 100
        
        if combined_coverage >= min_coverage_pct:
            data_quality = "high" if combined_coverage >= 50 else "medium" if combined_coverage >= 25 else "low"
            print(f"Will calculate growth from {prev_year} to {latest_year} with {combined_coverage:.1f}% coverage (quality: {data_quality})")
            # Return a special identifier to indicate we need to calculate growth
            return f"calc_growth_{prev_year}_to_{latest_year}", combined_coverage, data_quality
    
    # 5. As a last resort, return the column with highest coverage, even if below threshold
    if sorted_columns:
        best_col, best_coverage = sorted_columns[0]
        data_quality = "low"
        print(f"Using best available column {best_col} with limited coverage {best_coverage:.1f}% (quality: {data_quality})")
        return best_col, best_coverage, data_quality
    
    # 6. Truly no growth data available
    print("No usable growth data found")
    return None, 0, "none"

def calculate_growth_from_years(df, prev_year, latest_year):
    """
    Calculate growth rates directly from year price columns
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        prev_year (int): The earlier year
        latest_year (int): The later year
        
    Returns:
        pd.Series: Calculated growth rates
    """
    # Create a series with NaN values
    growth = pd.Series(np.nan, index=df.index)
    
    # Only calculate for rows where both years have data
    mask = (df[prev_year].notna() & df[latest_year].notna() & (df[prev_year] > 0))
    
    # Calculate growth for valid rows
    growth.loc[mask] = ((df.loc[mask, latest_year] / df.loc[mask, prev_year]) - 1) * 100
    
    return growth

def estimate_growth_from_price_level(df, market_median=None):
    """
    Estimate growth based on price level relative to market
    Used as a last resort when no growth data is available
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        market_median (float, optional): Market median price (calculated if not provided)
        
    Returns:
        pd.Series: Estimated growth values
    """
    if 'median_price_sqft' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    # Calculate market median if not provided
    if market_median is None:
        market_median = df['median_price_sqft'].median()
    
    # Skip if market_median is invalid
    if pd.isna(market_median) or market_median <= 0:
        return pd.Series(np.nan, index=df.index)
    
    # Create a series with NaN values
    estimated_growth = pd.Series(np.nan, index=df.index)
    
    # Only estimate for rows with valid prices
    mask = df['median_price_sqft'].notna() & (df['median_price_sqft'] > 0)
    
    # Calculate price ratio and convert to growth estimate
    # Using logarithmic scaling to provide reasonable estimates
    # Higher-priced areas typically have experienced stronger historical growth
    price_ratios = df.loc[mask, 'median_price_sqft'] / market_median
    estimated_growth.loc[mask] = np.log(price_ratios) * 5
    
    # Apply reasonable bounds
    estimated_growth = estimated_growth.clip(-15, 25)
    
    return estimated_growth

def perform_microsegment_analysis(df, filters=None, growth_column=None, min_coverage_pct=15):
    """
    Perform microsegment analysis on the dataframe with optional filters
    Enhanced to better handle sparse data
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        filters (dict, optional): Dictionary of filters to apply. Defaults to None.
            Example: {'property_type_en': 'Apartment', 'reg_type_en': 'Existing Properties'}
        growth_column (str, optional): Specific growth column to use. Defaults to None.
            If None, selects the best available column.
        min_coverage_pct (float): Minimum percentage of non-null values required (default: 15%)
            
    Returns:
        pd.DataFrame: DataFrame with microsegment analysis results
        dict: Dictionary with metadata about the analysis quality
    """
    # Create metadata dictionary to track analysis quality
    metadata = {
        'growth_column': None,
        'coverage_pct': 0,
        'data_quality': 'none',
        'estimation_method': None,
    }
    
    # Apply filters if provided
    filtered_df = df.copy()
    if filters:
        for column, value in filters.items():
            if column != 'time_horizon' and value != 'All' and value is not None and column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[column] == value]
    
    print(f"Filtered data size: {len(filtered_df)} rows")
    
    # Handle empty dataframe after filtering
    if len(filtered_df) == 0:
        print("No data after filtering")
        # Return empty dataframe with required columns
        empty_df = pd.DataFrame(columns=[
            'property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en',
            'median_price_sqft', 'transaction_count', 'investment_score',
            'recent_growth'
        ])
        return empty_df, metadata
    
    # Get years for price data
    years = sorted([col for col in filtered_df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025])
    
    # Select best growth column based on data coverage
    selected_growth, coverage_pct, data_quality = select_best_growth_column(
        filtered_df, 
        requested_horizon=growth_column,
        min_coverage_pct=min_coverage_pct
    )
    
    # Update metadata
    metadata['growth_column'] = selected_growth
    metadata['coverage_pct'] = coverage_pct
    metadata['data_quality'] = data_quality
    
    # If no usable growth data at all, create a basic result with median prices
    if selected_growth is None:
        print("No usable growth column found, using estimated growth")
        
        # Group by microsegments and calculate basic metrics
        microsegments = filtered_df.groupby([
            'property_type_en', 
            'rooms_en', 
            'reg_type_en', 
            'area_name_en'
        ]).agg({
            'median_price_sqft': 'median',
            'transaction_count': 'sum',
        }).reset_index()
        
        # Estimate growth based on price level
        microsegments['recent_growth'] = estimate_growth_from_price_level(microsegments)
        metadata['estimation_method'] = 'price_level'
        
        # Calculate investment score
        microsegments['investment_score'] = calculate_investment_score(microsegments, is_estimated=True)
        
        # Sort by investment score
        microsegments.sort_values('investment_score', ascending=False, inplace=True)
        
        return microsegments, metadata
    
    # Special case: calculate growth from year columns
    if isinstance(selected_growth, str) and selected_growth.startswith('calc_growth_'):
        print("Calculating growth from year columns")
        _, prev_year, _, latest_year = selected_growth.split('_')
        prev_year = int(prev_year)
        latest_year = int(latest_year)
        
        # Add calculated growth column to filtered_df
        column_name = f'growth_{prev_year}_to_{latest_year}'
        filtered_df[column_name] = calculate_growth_from_years(filtered_df, prev_year, latest_year)
        selected_growth = column_name
        metadata['estimation_method'] = 'year_columns'
    
    # Group by microsegments and calculate metrics
    try:
        microsegments = filtered_df.groupby([
            'property_type_en', 
            'rooms_en', 
            'reg_type_en', 
            'area_name_en'
        ]).agg({
            'median_price_sqft': 'median',
            'transaction_count': 'sum',
            selected_growth: 'median',
        }).reset_index()
        
        # Make sure all necessary columns exist
        if 'median_price_sqft' not in microsegments.columns:
            microsegments['median_price_sqft'] = np.nan
        if 'transaction_count' not in microsegments.columns:
            microsegments['transaction_count'] = 0
            
        # Rename the growth column to 'recent_growth' for consistency in the calculate_investment_score function
        microsegments['recent_growth'] = microsegments[selected_growth]
        
        # Calculate investment score
        microsegments['investment_score'] = calculate_investment_score(
            microsegments, 
            is_estimated=metadata['estimation_method'] is not None
        )
        
        # Sort by investment score
        microsegments.sort_values('investment_score', ascending=False, inplace=True)
        
        return microsegments, metadata
    except Exception as e:
        print(f"Error in microsegment analysis: {e}")
        # Return empty dataframe with required columns
        empty_df = pd.DataFrame(columns=[
            'property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en',
            'median_price_sqft', 'transaction_count', 'investment_score',
            'recent_growth'
        ])
        return empty_df, metadata

def calculate_investment_score(df, is_estimated=False):
    """
    Calculate investment score based on multiple factors
    Enhanced to better handle limited data
    
    Formula:
    - 40% weight: Recent growth (or less if estimated)
    - 30% weight: Relative price (lower price gets higher score)
    - 20% weight: Transaction volume (liquidity)
    - 10% weight: Developer reputation (if available)
    
    Args:
        df (pd.DataFrame): DataFrame with microsegment data
        is_estimated (bool): Whether growth data is estimated rather than actual
        
    Returns:
        pd.Series: Investment scores
    """
    # Check if we have enough data
    if len(df) == 0 or 'recent_growth' not in df.columns:
        return pd.Series([50] * len(df))  # Default score
    
    # Adjust weights based on data quality
    weights = {
        'growth_score': 0.4,
        'price_score': 0.3,
        'volume_score': 0.2,
        'developer_score': 0.1
    }
    
    # If growth is estimated, reduce its weight and redistribute
    if is_estimated:
        weights['growth_score'] = 0.2
        weights['price_score'] = 0.4
        weights['volume_score'] = 0.3
    
    # Normalize metrics to 0-100 scale
    df['growth_score'] = normalize_score(df['recent_growth'])
    df['price_score'] = 100 - normalize_score(df['median_price_sqft'])  # Inverse relationship
    df['volume_score'] = normalize_score(df['transaction_count'])
    
    # Calculate weighted score
    weighted_score = pd.Series(0, index=df.index)
    for col, weight in weights.items():
        if col == 'developer_score':
            # Add developer score if available (10% weight)
            if 'developer_reputation_score' in df.columns:
                weighted_score += df['developer_reputation_score'] * weight
            elif 'developer_name' in df.columns:
                # Create a simple developer score based on name presence
                weighted_score += 5  # Small default bonus
        elif col in df.columns:
            weighted_score += df[col] * weight
    
    return weighted_score

def normalize_score(series):
    """Normalize a series to 0-100 scale"""
    if series.isna().all() or len(series) == 0:
        return pd.Series(50, index=series.index)  # Default to middle score if all NaN
    
    # Fill NaNs with median to avoid issues
    filled_series = series.fillna(series.median())
    
    min_val = filled_series.min()
    max_val = filled_series.max()
    
    if max_val == min_val:
        return pd.Series(50, index=series.index)  # Default to middle score if all values are the same
        
    return 100 * (filled_series - min_val) / (max_val - min_val)

def create_investment_heatmap(df, time_horizon="short_term_growth"):
    """
    Create investment heatmap visualization with time horizon consideration
    Enhanced to better handle sparse data and show data quality
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        time_horizon (str): Time horizon to use for analysis (short_term_growth, medium_term_growth, long_term_cagr)
        
    Returns:
        go.Figure: Plotly figure with investment heatmap
    """
    print(f"Creating heatmap with time horizon: {time_horizon}")
    
    # Check if we have enough data for a meaningful heatmap
    if len(df) < 2:
        # Not enough data, return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for investment heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Perform micro-segment analysis with the time horizon
    microsegments, metadata = perform_microsegment_analysis(df, growth_column=time_horizon)
    
    if len(microsegments) < 2:
        # Not enough data after analysis
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for investment heatmap after filtering",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get actual growth column used
    growth_column = metadata['growth_column']
    data_quality = metadata['data_quality']
    coverage_pct = metadata['coverage_pct']
    
    # Try to create a property type vs room type heatmap
    try:
        pivot_data = microsegments.pivot_table(
            index='property_type_en',
            columns='rooms_en',
            values='recent_growth',
            aggfunc='median'
        )
        
        # Check if we have valid data
        if pivot_data.isnull().all().all():
            raise ValueError("No valid data in pivot table")
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Room Type", y="Property Type", color=get_time_horizon_display(time_horizon)),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale="RdYlGn",
            title=f"Investment Analysis by Property Type and Room Type<br><sub>Based on {get_time_horizon_display(time_horizon)}</sub>"
        )
    except Exception as e:
        print(f"Error creating property/room heatmap: {e}")
        
        # Try creating an investment score heatmap instead
        try:
            # Create pivot table with investment score
            pivot_data = microsegments.pivot_table(
                index='property_type_en',
                columns='rooms_en',
                values='investment_score',
                aggfunc='mean'
            )
            
            # Create heatmap
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Room Type", y="Property Type", color="Investment Score"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale="RdYlGn",
                title=f"Investment Score Heatmap - Property Type vs Room Type<br><sub>Based on {get_time_horizon_display(time_horizon)}</sub>"
            )
        except Exception as e2:
            print(f"Error creating investment score heatmap: {e2}")
            
            # Create a fallback scatter plot
            fig = px.scatter(
                microsegments,
                x='median_price_sqft',
                y='recent_growth',
                size='transaction_count',
                color='investment_score',
                hover_name='area_name_en',
                labels={
                    'median_price_sqft': 'Median Price (AED/sqft)',
                    'recent_growth': f'{get_time_horizon_display(time_horizon)} (%)',
                    'transaction_count': 'Transaction Volume',
                    'investment_score': 'Investment Score'
                },
                color_continuous_scale="RdYlGn",
                title=f"Investment Analysis - Price vs Growth<br><sub>Based on {get_time_horizon_display(time_horizon)}</sub>"
            )
    
    # Add data quality annotation
    # data_quality_colors = {
    #     "high": "green",
    #     "medium": "orange",
    #     "low": "red",
    #     "none": "red"
    # }
    
    # # Add a data quality indicator in the corner
    # fig.add_annotation(
    #     x=1,
    #     y=0,
    #     xref="paper",
    #     yref="paper",
    #     text=f"Data Quality: {data_quality.upper()} ({coverage_pct:.1f}% coverage)",
    #     showarrow=False,
    #     font=dict(
    #         size=12,
    #         color="white"
    #     ),
    #     align="right",
    #     bgcolor=data_quality_colors.get(data_quality, "gray"),
    #     bordercolor="black",
    #     borderwidth=1,
    #     borderpad=4,
    #     opacity=0.8
    # )
    
    # If data is estimated, add note
    if metadata['estimation_method'] is not None:
        method_note = "price levels" if metadata['estimation_method'] == 'price_level' else "yearly prices"
        fig.add_annotation(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            text=f"Note: Growth estimated from {method_note}",
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="left",
            bgcolor="rgba(255, 255, 200, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    # Improve layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=70, b=30),
        title=dict(
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        coloraxis_colorbar=dict(
            title="Score" if "investment_score" in str(fig) else f"{get_time_horizon_display(time_horizon)} (%)"
        )
    )
    
    return fig

def create_opportunity_scatter(df, time_horizon="short_term_growth", max_points=100):
    """
    Create an improved opportunity scatter plot visualization
    Enhanced to better handle sparse data and show data quality
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        time_horizon (str): Time horizon to use for analysis
        max_points (int): Maximum number of points to display
        
    Returns:
        go.Figure: Plotly figure with opportunity scatter plot
    """
    print(f"Creating opportunity scatter with time horizon: {time_horizon}")
    
    # Calculate investment scores with the specific time horizon
    investment_data, metadata = perform_microsegment_analysis(df, growth_column=time_horizon)
    
    if len(investment_data) == 0:
        # No data available
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for investment opportunities visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get the growth quality information
    growth_column = 'recent_growth'  # This will always be available from perform_microsegment_analysis
    data_quality = metadata['data_quality']
    coverage_pct = metadata['coverage_pct']
    
    # Check if we have the necessary data
    if growth_column not in investment_data.columns or 'median_price_sqft' not in investment_data.columns:
        missing_cols = []
        if growth_column not in investment_data.columns:
            missing_cols.append(growth_column)
        if 'median_price_sqft' not in investment_data.columns:
            missing_cols.append('median_price_sqft')
            
        print(f"Missing Data: Required columns {', '.join(missing_cols)} not found")
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required data columns: {', '.join(missing_cols)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Group data by property type to maintain diversity while limiting points
    grouped_data_list = []
    
    for prop_type in investment_data['property_type_en'].unique():
        prop_df = investment_data[investment_data['property_type_en'] == prop_type]
        # Take top 5 for each property type or fewer if not available
        top_n = min(5, len(prop_df))
        top_props = prop_df.nlargest(top_n, 'investment_score')
        grouped_data_list.append(top_props)
    
    # Combine all property types
    if grouped_data_list:
        grouped_data = pd.concat(grouped_data_list)
        
        # If still too many points, take top overall by score
        if len(grouped_data) > max_points:
            grouped_data = grouped_data.nlargest(max_points, 'investment_score')
    else:
        # If grouping failed, just take top N by investment score
        grouped_data = investment_data.nlargest(max_points, 'investment_score')
    
    print(f"Selected {len(grouped_data)} data points for visualization")
    
    # Create figure
    fig = px.scatter(
        grouped_data,
        x='median_price_sqft',
        y=growth_column,
        size='transaction_count',
        color='investment_score',
        hover_name='area_name_en',
        hover_data={
            'property_type_en': True,
            'rooms_en': True,
            'area_name_en': True,
            'median_price_sqft': ':.0f',
            growth_column: ':.1f',
            'transaction_count': ':.0f',
            'investment_score': ':.1f'
        },
        labels={
            'median_price_sqft': 'Median Price (AED/sqft)',
            growth_column: f'{get_time_horizon_display(time_horizon)} (%)',
            'transaction_count': 'Transaction Volume',
            'investment_score': 'Investment Score'
        },
        color_continuous_scale="RdYlGn",
        title=f"Investment Opportunities - Price vs Growth Matrix<br><sub>Based on {get_time_horizon_display(time_horizon)}</sub>"
    )
    
    # Get median values for quadrant lines
    median_price = grouped_data['median_price_sqft'].median()
    growth_median = grouped_data[growth_column].median()
    
    # Get min/max values for annotations and shapes
    x_min = grouped_data['median_price_sqft'].min()
    x_max = grouped_data['median_price_sqft'].max()
    y_min = grouped_data[growth_column].min()
    y_max = grouped_data[growth_column].max()
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=median_price,
        y0=y_min,
        x1=median_price,
        y1=y_max,
        line=dict(color="gray", width=1.5, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=x_min,
        y0=growth_median,
        x1=x_max,
        y1=growth_median,
        line=dict(color="gray", width=1.5, dash="dash")
    )
    
    # Calculate positions for quadrant labels
    x_left = x_min + (median_price - x_min) * 0.3
    x_right = median_price + (x_max - median_price) * 0.3
    y_top = growth_median + (y_max - growth_median) * 0.3
    y_bottom = y_min + (growth_median - y_min) * 0.3
    
    # Add quadrant annotations
    fig.add_annotation(
        x=x_left,
        y=y_top,
        text="Value Opportunities<br><sub>(High Growth, Low Price)</sub>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(0,255,0,0.2)",
        bordercolor="green",
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    fig.add_annotation(
        x=x_right,
        y=y_top,
        text="Premium Growth<br><sub>(High Growth, High Price)</sub>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,0,0.2)",
        bordercolor="gold",
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    fig.add_annotation(
        x=x_left,
        y=y_bottom,
        text="Underperforming<br><sub>(Low Growth, Low Price)</sub>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,0,0,0.2)",
        bordercolor="red",
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    fig.add_annotation(
        x=x_right,
        y=y_bottom,
        text="Premium Plateau<br><sub>(Low Growth, High Price)</sub>",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(0,0,255,0.2)",
        bordercolor="blue",
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    # Add subtle quadrant backgrounds
    fig.add_shape(
        type="rect",
        x0=x_min, y0=growth_median,
        x1=median_price, y1=y_max,
        fillcolor="rgba(0,255,0,0.05)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=median_price, y0=growth_median,
        x1=x_max, y1=y_max,
        fillcolor="rgba(255,255,0,0.05)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=x_min, y0=y_min,
        x1=median_price, y1=growth_median,
        fillcolor="rgba(255,0,0,0.05)",
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="rect",
        x0=median_price, y0=y_min,
        x1=x_max, y1=growth_median,
        fillcolor="rgba(0,0,255,0.05)",
        line=dict(width=0),
        layer="below"
    )
    
    # If data is estimated, add note
    if metadata['estimation_method'] is not None:
        method_note = "price levels" if metadata['estimation_method'] == 'price_level' else "yearly prices"
        fig.add_annotation(
            x=0,
            y=0,
            xref="paper",
            yref="paper",
            text=f"Note: Growth estimated from {method_note}",
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="left",
            bgcolor="rgba(255, 255, 200, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Investment Opportunities - Price vs Growth Matrix<br><sub>Based on {get_time_horizon_display(time_horizon)}</sub>",
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Median Price (AED/sqft)",
            titlefont=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(211,211,211,0.3)"
        ),
        yaxis=dict(
            title=f"{get_time_horizon_display(time_horizon)} (%)",
            titlefont=dict(size=14),
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(211,211,211,0.3)"
        ),
        legend_title="Investment Score",
        height=650,
        margin=dict(l=10, r=10, t=100, b=30),
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='white',
        hovermode='closest',
        coloraxis_colorbar=dict(
            title="Investment Score",
            titleside="right"
        )
    )
    
    # Make sure markers are visible
    fig.update_traces(
        marker=dict(
            line=dict(width=2, color='DarkSlateGrey'),
            sizemin=8  # Set minimum size for markers
        )
    )
    
    return fig

def create_microsegment_table(microsegment_df, metadata=None):
    """
    Create a table visualization for microsegment analysis
    Enhanced to include data quality indicators
    
    Args:
        microsegment_df (pd.DataFrame): DataFrame with microsegment analysis
        metadata (dict): Dictionary with data quality information
        
    Returns:
        dash component: Dash data table
    """
    # Check for valid data
    if microsegment_df is None or len(microsegment_df) == 0:
        return html.Div([
            html.H5("No Data Available"),
            html.P("No microsegment data available for the selected filters.")
        ])
    
    # Process metadata
    data_quality = "unknown"
    coverage_pct = 0
    estimation_method = None
    if metadata is not None:
        data_quality = metadata.get('data_quality', 'unknown')
        coverage_pct = metadata.get('coverage_pct', 0)
        estimation_method = metadata.get('estimation_method')
    
    # Select columns to display
    display_columns = []
    
    # Core columns
    core_columns = ['property_type_en', 'rooms_en', 'area_name_en', 
                   'median_price_sqft', 'transaction_count']
    
    # Growth metrics in order of preference
    growth_metrics = ['recent_growth', 'short_term_growth', 'medium_term_growth', 'long_term_cagr']
    
    # Add core columns if available
    for col in core_columns:
        if col in microsegment_df.columns:
            display_columns.append(col)
    
    # Add a growth metric if available
    growth_col_found = False
    for metric in growth_metrics:
        if metric in microsegment_df.columns:
            display_columns.append(metric)
            growth_col_found = True
            break
    
    # Add investment score if available
    if 'investment_score' in microsegment_df.columns:
        display_columns.append('investment_score')
    
    # Prepare display DataFrame
    display_df = microsegment_df[display_columns].copy()
    
    # Sort by investment score if available, otherwise by growth
    if 'investment_score' in display_df.columns:
        display_df = display_df.sort_values('investment_score', ascending=False)
    else:
        for growth_col in growth_metrics:
            if growth_col in display_df.columns:
                display_df = display_df.sort_values(growth_col, ascending=False)
                break
    
    # Rename columns for display
    column_map = {
        'property_type_en': 'Property Type',
        'rooms_en': 'Room Config',
        'area_name_en': 'Area',
        'median_price_sqft': 'Price (AED/sqft)',
        'transaction_count': 'Transactions',
        'recent_growth': 'Growth (%)',
        'short_term_growth': 'Growth (%)',
        'medium_term_growth': 'Medium-Term Growth (%)',
        'long_term_cagr': 'Long-Term CAGR (%)',
        'investment_score': 'Investment Score'
    }
    
    display_df = display_df.rename(columns={col: column_map.get(col, col) for col in display_df.columns})
    
    # Format numeric columns with proper type casting
    for col in display_df.columns:
        if col == 'Price (AED/sqft)' and col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(0)
        elif col == 'Transactions' and col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(0)
        elif 'Growth' in col and col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(1)
        elif col == 'Investment Score' and col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(1)
    
    # Limit to top 15 rows
    display_df = display_df.head(15)
    
    # Create style conditions for highlighting
    style_conditions = []
    
    # Highlight investment score
    if 'Investment Score' in display_df.columns:
        style_conditions.extend([
            {
                'if': {'column_id': 'Investment Score', 'filter_query': '{Investment Score} >= 80'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'Investment Score', 'filter_query': '{Investment Score} >= 65 && {Investment Score} < 80'},
                'backgroundColor': 'rgba(255, 255, 0, 0.2)'
            }
        ])
    
    # Highlight growth
    growth_col = next((col for col in display_df.columns if 'Growth' in col), None)
    if growth_col:
        style_conditions.extend([
            {
                'if': {'column_id': growth_col, 'filter_query': f'{{{growth_col}}} >= 10'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)'
            },
            {
                'if': {'column_id': growth_col, 'filter_query': f'{{{growth_col}}} < 0'},
                'backgroundColor': 'rgba(255, 0, 0, 0.2)'
            }
        ])
    
    # Create the table
    table = dash_table.DataTable(
        id='microsegment-table',
        columns=[
            {"name": i, "id": i} for i in display_df.columns
        ],
        data=display_df.to_dict('records'),
        sort_action="native",
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px'
        },
        style_data_conditional=style_conditions
    )
    
    # Data quality indicator colors
    quality_colors = {
        'high': 'green',
        'medium': 'orange',
        'low': 'red',
        'none': 'red',
        'unknown': 'gray'
    }
    
    # Note about data estimation if applicable
    estimation_note = None
    if estimation_method is not None:
        method_text = "price levels" if estimation_method == 'price_level' else "yearly prices"
        estimation_note = html.Div([
            html.Span(
                f"Note: Growth values are estimated from {method_text} due to limited historical data",
                style={
                    'backgroundColor': 'rgba(255, 255, 200, 0.8)',
                    'color': 'black',
                    'padding': '4px 8px',
                    'borderRadius': '4px',
                    'fontSize': '12px',
                    'marginTop': '5px',
                    'display': 'inline-block'
                }
            )
        ])
    
    # Build the complete component
    header_components = [
        html.H6("Top Investment Opportunities by Micro-Segment")
    ]
    
    if estimation_note:
        header_components.append(estimation_note)
    
    return html.Div([
        html.Div(header_components, style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '5px'}),
        html.P("Segments are ranked by investment potential, combining growth, price level, and liquidity."),
        table
    ])