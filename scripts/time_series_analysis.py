"""
Time Series Analysis Module for Dubai Real Estate Dashboard

This module provides functions for analyzing and visualizing time series
data including price trends, market cycles, and price forecasts.

Key functions:
- calculate_growth_rates: Calculate and standardize growth rates from yearly price data
- create_price_trends_chart: Visualize historical price movements with trends
- detect_market_cycles: Identify market phases (expansion, peak, contraction, recovery)
- create_price_forecast: Generate future price projections with confidence intervals

Each function handles progressive filtering and data quality assessment.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import find_peaks
import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No frequency information was provided")


def calculate_growth_rates(df, min_data_points=3):
    """
    Calculate and standardize growth rates from yearly price data
    
    Args:
        df (pd.DataFrame): DataFrame containing yearly price data
        min_data_points (int): Minimum number of non-null data points required for reliable growth
        
    Returns:
        pd.DataFrame: DataFrame with standardized growth rates
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Identify year columns (string or numeric columns representing years 2000-2025)
    year_columns = []  # Actual column names as they appear in the dataframe
    year_values = []   # Integer year values for sorting and calculations
    
    for col in df.columns:
        # Direct numeric column (unlikely but possible)
        if isinstance(col, (int, float)) and 2000 <= col <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
        # String column that's a year
        elif isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
    
    # Sort columns by year value
    sorted_pairs = sorted(zip(year_values, year_columns))
    year_values = [pair[0] for pair in sorted_pairs]
    year_columns = [pair[1] for pair in sorted_pairs]
    
    if not year_columns:
        print("No year columns found in data")
        return result_df
    
    print(f"Found {len(year_columns)} year columns from {min(year_values)} to {max(year_values)}")    
    
    # Check and standardize the growth rate columns
    for i in range(len(year_columns) - 1):
        curr_col = year_columns[i]
        next_col = year_columns[i+1]
        curr_year = year_values[i]
        next_year = year_values[i+1]
        growth_col = f"growth_{curr_year}_to_{next_year}"
        
        # Check if the growth column already exists
        if growth_col in result_df.columns:
            # If it exists but is a string, convert to numeric
            if result_df[growth_col].dtype == 'object':
                print(f"Converting {growth_col} from {result_df[growth_col].dtype} to numeric")
                result_df[growth_col] = pd.to_numeric(result_df[growth_col], errors='coerce')
        else:
            # Calculate the growth rate if it doesn't exist
            print(f"Calculating missing growth rate: {growth_col}")
            
            # Identify rows with valid data for both years
            mask = (result_df[curr_col].notna() & result_df[next_col].notna() & (result_df[curr_col] > 0))
            valid_data_count = mask.sum()
            
            # Only calculate if we have sufficient data points
            if valid_data_count >= min_data_points:
                # Calculate growth rate as percentage
                result_df.loc[mask, growth_col] = (
                    (result_df.loc[mask, next_col] / result_df.loc[mask, curr_col]) - 1
                ) * 100
                print(f"  - Calculated {growth_col} for {valid_data_count} records")
            else:
                # Create empty column if insufficient data
                result_df[growth_col] = np.nan
                print(f"  - Insufficient data for {growth_col} (only {valid_data_count} valid points)")
    
    # Create additional derived growth metrics
    growth_columns = [col for col in result_df.columns if isinstance(col, str) and col.startswith('growth_')]
    
    if len(growth_columns) >= 1:
        # Latest growth rate (short term)
        latest_growth = sorted(growth_columns)[-1]
        if 'short_term_growth' not in result_df.columns:
            result_df['short_term_growth'] = result_df[latest_growth]
            print(f"Created short_term_growth from {latest_growth}")
    
    if len(growth_columns) >= 3:
        # Medium term growth (3-year if available)
        medium_growth = sorted(growth_columns)[-3]
        if 'medium_term_growth' not in result_df.columns:
            result_df['medium_term_growth'] = result_df[medium_growth]
            print(f"Created medium_term_growth from {medium_growth}")
    
    # Calculate long-term CAGR if not present
    if 'long_term_cagr' not in result_df.columns and len(year_values) >= 5:
        # Use 5-year CAGR if available, otherwise use the full range
        try:
            start_idx = -5 if len(year_values) >= 5 else 0
            start_year = year_values[start_idx]
            end_year = year_values[-1]
            start_col = year_columns[start_idx]
            end_col = year_columns[-1]
            years_diff = end_year - start_year
            
            # CAGR calculation - [(End Value / Start Value) ^ (1 / Years)] - 1
            mask = (result_df[start_col].notna() & result_df[end_col].notna() & (result_df[start_col] > 0))
            if mask.sum() >= min_data_points:
                result_df.loc[mask, 'long_term_cagr'] = (
                    (result_df.loc[mask, end_col] / result_df.loc[mask, start_col]) ** (1 / years_diff) - 1
                ) * 100
                print(f"Calculated long_term_cagr from {start_year} to {end_year} ({years_diff} years)")
        except Exception as e:
            print(f"Error calculating long-term CAGR: {e}")
    
    # Calculate rolling averages for smoothing
    if len(year_columns) >= 3:
        # Create rolling averages for 3-year periods
        for i in range(2, len(year_columns)):
            roll_cols = year_columns[i-2:i+1]
            roll_years = year_values[i-2:i+1]
            roll_col = f"roll_avg_{roll_years[0]}_{roll_years[-1]}"
            
            # Calculate the rolling average (ignoring NaNs)
            result_df[roll_col] = result_df[roll_cols].mean(axis=1, skipna=True)
    
    return result_df

def extract_market_summary(df, period_col='transaction_year'):
    """
    Extract market-level summary statistics by year
    
    Args:
        df (pd.DataFrame): DataFrame containing real estate data
        period_col (str): Column name for the period (typically transaction_year)
        
    Returns:
        pd.DataFrame: Yearly market summary with prices and growth rates
    """
    # Identify year columns
    year_columns = []  # Actual column names
    year_values = []   # Integer year values
    
    for col in df.columns:
        # Direct numeric column (unlikely but possible)
        if isinstance(col, (int, float)) and 2000 <= col <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
        # String column that's a year
        elif isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
    
    # Sort columns by year value
    sorted_pairs = sorted(zip(year_values, year_columns))
    year_values = [pair[0] for pair in sorted_pairs]
    year_columns = [pair[1] for pair in sorted_pairs]
    
    if not year_columns:
        return pd.DataFrame()  # Empty DataFrame if no year columns
    
    # Create summary DataFrame
    summary_data = []
    
    for i, col in enumerate(year_columns):
        # Get data for this year
        year_data = df[col].dropna()
        
        if len(year_data) > 0:
            summary_data.append({
                'year': year_values[i],
                'median_price': year_data.median(),
                'mean_price': year_data.mean(),
                'min_price': year_data.min(),
                'max_price': year_data.max(),
                'price_range': year_data.max() - year_data.min(),
                'std_dev': year_data.std(),
                'count': len(year_data),
            })
    
    # Convert to DataFrame
    market_summary = pd.DataFrame(summary_data)
    
    # Calculate year-over-year growth
    if len(market_summary) > 1:
        market_summary['yoy_growth'] = market_summary['median_price'].pct_change() * 100
        
        # Calculate moving averages for trend analysis
        market_summary['growth_3yr_ma'] = market_summary['yoy_growth'].rolling(3, min_periods=1).mean()
    
    return market_summary

def get_market_phase(growth_rate, prev_growth=None, threshold=0.5):
    """
    Determine market phase based on growth rate and comparison with previous growth
    
    Args:
        growth_rate (float): Current growth rate
        prev_growth (float): Previous growth rate
        threshold (float): Threshold for significant change
        
    Returns:
        str: Market phase ('expansion', 'peak', 'contraction', or 'recovery')
    """
    if pd.isna(growth_rate):
        return 'unknown'
    
    # If we don't have previous growth, use just the current growth
    if prev_growth is None or pd.isna(prev_growth):
        if growth_rate > 5:
            return 'expansion'
        elif growth_rate > 0:
            return 'recovery'
        else:
            return 'contraction'
    
    # Define phases based on current growth and change from previous
    if growth_rate > 0:
        if growth_rate > prev_growth + threshold:
            return 'expansion'  # Positive and accelerating
        elif growth_rate < prev_growth - threshold:
            return 'peak'  # Positive but decelerating
        else:
            return 'stable growth'  # Positive and steady
    else:
        if growth_rate < prev_growth - threshold:
            return 'contraction'  # Negative and worsening
        elif growth_rate > prev_growth + threshold:
            return 'recovery'  # Negative but improving
        else:
            return 'stable decline'  # Negative and steady


def get_market_summary_for_filtered_data(df, property_type=None, area=None, room_type=None, registration_type=None):
    """
    Generate a market summary for filtered data
    
    Args:
        df (pd.DataFrame): The full DataFrame
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        
    Returns:
        pd.DataFrame: Market summary for the filtered data
    """
    # Apply filters
    filtered_df = df.copy()
    
    if property_type != 'All' and property_type is not None and 'property_type_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['property_type_en'] == property_type]
    
    if area != 'All' and area is not None and 'area_name_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['area_name_en'] == area]
    
    if room_type != 'All' and room_type is not None and 'rooms_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['rooms_en'] == room_type]
    
    if registration_type != 'All' and registration_type is not None and 'reg_type_en' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['reg_type_en'] == registration_type]
    
    # Extract market summary from filtered data
    return extract_market_summary(filtered_df)


def get_data_quality_metrics(df):
    """
    Calculate data quality metrics for the time series data
    
    Args:
        df (pd.DataFrame): DataFrame with yearly price data
        
    Returns:
        dict: Dictionary with data quality metrics
    """
    # Identify year columns
    year_columns = []  # Actual column names
    year_values = []   # Integer year values
    
    for col in df.columns:
        # Direct numeric column (unlikely but possible)
        if isinstance(col, (int, float)) and 2000 <= col <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
        # String column that's a year
        elif isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2025:
            year_columns.append(col)
            year_values.append(int(col))
    
    # Sort columns by year value
    sorted_pairs = sorted(zip(year_values, year_columns))
    year_values = [pair[0] for pair in sorted_pairs]
    year_columns = [pair[1] for pair in sorted_pairs]
    
    if not year_columns:
        return {'quality': 'none', 'coverage': 0, 'years_available': 0}
    
    # Calculate coverage metrics
    total_cells = len(df) * len(year_columns)
    filled_cells = df[year_columns].notna().sum().sum()
    coverage_pct = (filled_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Calculate average data points per year
    avg_data_per_year = filled_cells / len(year_columns) if len(year_columns) > 0 else 0
    
    # Assess quality based on coverage
    quality = 'high' if coverage_pct >= 50 else 'medium' if coverage_pct >= 25 else 'low'
    
    # Check recency of data
    recent_years = [year_values[i] for i, year in enumerate(year_values) if year >= 2018]
    recent_columns = [year_columns[i] for i, year in enumerate(year_values) if year_values[i] >= 2018]
    
    recent_coverage = 0
    if recent_columns:
        recent_total = len(df) * len(recent_columns)
        recent_filled = df[recent_columns].notna().sum().sum()
        recent_coverage = (recent_filled / recent_total) * 100 if recent_total > 0 else 0
    
    # Create quality metrics dictionary
    metrics = {
        'quality': quality,
        'coverage': coverage_pct,
        'years_available': len(year_columns),
        'avg_data_per_year': avg_data_per_year,
        'recent_coverage': recent_coverage,
        'recent_years': len(recent_years)
    }
    
    return metrics

def create_price_trends_chart(df, property_type=None, area=None, room_type=None, registration_type=None):
    """
    Create a chart showing historical price trends
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        
    Returns:
        go.Figure: Plotly figure with price trends visualization
    """
    # Apply filters progressively to ensure data availability
    filtered_df = progressive_filter(df, property_type, area, room_type, registration_type)
    
    # Calculate missing growth rates if needed
    filtered_df = calculate_growth_rates(filtered_df)
    
    # Get data quality metrics
    quality_metrics = get_data_quality_metrics(filtered_df)
    
    # Get market summary
    market_summary = get_market_summary_for_filtered_data(
        filtered_df, property_type, area, room_type, registration_type
    )
    
    # No data handling
    if len(market_summary) < 2:
        return create_no_data_figure("Price Trends", "Insufficient historical data for selected filters")
    
    # Create figure with dual y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price trend line
    fig.add_trace(
        go.Scatter(
            x=market_summary['year'],
            y=market_summary['median_price'],
            name="Median Price (AED/sqft)",
            line=dict(color='royalblue', width=3),
            hovertemplate="Year: %{x}<br>Price: %{y:.0f} AED/sqft<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add growth rate bars
    if 'yoy_growth' in market_summary.columns:
        fig.add_trace(
            go.Bar(
                x=market_summary['year'],
                y=market_summary['yoy_growth'],
                name="YoY Growth (%)",
                marker_color=['green' if x >= 0 else 'red' for x in market_summary['yoy_growth']],
                opacity=0.7,
                hovertemplate="Year: %{x}<br>Growth: %{y:.1f}%<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Add moving average line for growth if available
    if 'growth_3yr_ma' in market_summary.columns:
        fig.add_trace(
            go.Scatter(
                x=market_summary['year'],
                y=market_summary['growth_3yr_ma'],
                name="3-Year MA Growth (%)",
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate="Year: %{x}<br>3Yr MA Growth: %{y:.1f}%<extra></extra>"
            ),
            secondary_y=True
        )
    
    # Add smoothed price trend if enough data points
    if len(market_summary) >= 5:
        # Create smoothed line using rolling average
        market_summary['smooth_price'] = market_summary['median_price'].rolling(window=3, min_periods=1, center=True).mean()
        
        fig.add_trace(
            go.Scatter(
                x=market_summary['year'],
                y=market_summary['smooth_price'],
                name="Smoothed Trend",
                line=dict(color='rgba(0,0,255,0.3)', width=4),
                hovertemplate="Year: %{x}<br>Smoothed Price: %{y:.0f} AED/sqft<extra></extra>"
            ),
            secondary_y=False
        )
    
    # Add key market events if we have sufficient data
    market_events = []
    
    # Global Financial Crisis
    if any(2008 <= y <= 2009 for y in market_summary['year']):
        market_events.append({
            'year': 2008,
            'event': 'Global Financial Crisis',
            'description': 'Major market correction due to global economic crisis'
        })
    
    # Oil Price Crash
    if any(2014 <= y <= 2015 for y in market_summary['year']):
        market_events.append({
            'year': 2014,
            'event': 'Oil Price Crash',
            'description': 'Market adjustment following oil price decline'
        })
    
    # COVID-19 Pandemic
    if any(y == 2020 for y in market_summary['year']):
        market_events.append({
            'year': 2020,
            'event': 'COVID-19 Pandemic',
            'description': 'Market impact of global pandemic'
        })
    
    # Add vertical lines and annotations for key events
    for event in market_events:
        event_year = event['year']
        if event_year in market_summary['year'].values:
            # Get the price at this event year
            event_price = market_summary.loc[market_summary['year'] == event_year, 'median_price'].values[0]
            
            # Add vertical line
            fig.add_vline(
                x=event_year,
                line_width=1,
                line_dash="dash",
                line_color="gray"
            )
            
            # Add annotation
            fig.add_annotation(
                x=event_year,
                y=event_price * 1.1,  # Position above the price point
                text=event['event'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )
    
    # Determine current market phase if enough data
    current_phase = "Unknown"
    phase_color = "gray"
    
    if len(market_summary) >= 2 and 'yoy_growth' in market_summary.columns:
        latest_growth = market_summary['yoy_growth'].iloc[-1]
        prev_growth = market_summary['yoy_growth'].iloc[-2] if len(market_summary) > 2 else None
        
        current_phase = get_market_phase(latest_growth, prev_growth)
        
        # Set color based on phase
        phase_color_map = {
            'expansion': 'green',
            'peak': 'orange',
            'stable growth': 'lightgreen',
            'contraction': 'red',
            'recovery': 'blue',
            'stable decline': 'pink',
            'unknown': 'gray'
        }
        phase_color = phase_color_map.get(current_phase, 'gray')
    
    # Create subtitle with market phase
    subtitle = f"Current Market Phase: <span style='color:{phase_color};font-weight:bold;'>{current_phase.title()}</span>"
    
    # Add data quality indicator
    quality_indicator = f"Data Quality: {quality_metrics['quality'].upper()} ({quality_metrics['coverage']:.1f}% coverage)"
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Dubai Real Estate Price Trends (2003-2023)<br><span style='font-size:14px;'>{subtitle}</span>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Year",
            tickmode='linear',
            dtick=1,  # Show every year
            tickangle=45,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Median Price (AED/sqft)",
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black'
        ),
        yaxis2=dict(
            title="YoY Growth (%)",
            overlaying='y',
            side='right',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            range=[-max(abs(market_summary['yoy_growth'].min()) * 1.2, 
                       abs(market_summary['yoy_growth'].max()) * 1.2), 
                   max(abs(market_summary['yoy_growth'].min()) * 1.2, 
                       abs(market_summary['yoy_growth'].max()) * 1.2)]
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True,
        height=600,
        # annotations=[
        #     dict(
        #         text=quality_indicator,
        #         align='left',
        #         showarrow=False,
        #         xref='paper',
        #         yref='paper',
        #         x=0,
        #         y=-0.13,
        #         bordercolor='black',
        #         borderwidth=1,
        #         borderpad=4,
        #         bgcolor=get_quality_color(quality_metrics['quality']),
        #         opacity=0.8,
        #         font=dict(size=12, color="white")
        #     )
        #]
    )
    
    # Add filter description
    filter_desc = get_filter_description(property_type, area, room_type, registration_type)
    if filter_desc:
        fig.add_annotation(
            text=filter_desc,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0,
            y=1.12,
            font=dict(size=12)
        )
    
    return fig

def detect_market_cycles(df, property_type=None, area=None, room_type=None, registration_type=None, min_years=5):
    """
    Detect market cycle phases and transition points
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        min_years (int): Minimum number of years required for cycle detection
        
    Returns:
        tuple: (go.Figure, list of breakpoint years)
    """
    # Apply filters progressively to ensure data availability
    filtered_df = progressive_filter(df, property_type, area, room_type, registration_type)
    
    # Calculate missing growth rates if needed
    filtered_df = calculate_growth_rates(filtered_df)
    
    # Get data quality metrics
    quality_metrics = get_data_quality_metrics(filtered_df)
    
    # Get market summary
    market_summary = get_market_summary_for_filtered_data(
        filtered_df, property_type, area, room_type, registration_type
    )
    
    # No data handling
    if len(market_summary) < min_years:
        return create_no_data_figure("Market Cycles", 
                                   f"Insufficient historical data for cycle detection (need {min_years}+ years)"), []
    
    # Initialize figure
    fig = go.Figure()
    
    # Add price trend
    fig.add_trace(
        go.Scatter(
            x=market_summary['year'],
            y=market_summary['median_price'],
            name="Median Price",
            line=dict(color='royalblue', width=3),
            mode='lines+markers',
            hovertemplate="Year: %{x}<br>Price: %{y:.0f} AED/sqft<extra></extra>"
        )
    )
    
    # Identify market phases for each year
    if 'yoy_growth' in market_summary.columns:
        market_summary['market_phase'] = 'unknown'
        
        for i in range(1, len(market_summary)):
            curr_growth = market_summary['yoy_growth'].iloc[i]
            prev_growth = market_summary['yoy_growth'].iloc[i-1]
            market_summary.loc[market_summary.index[i], 'market_phase'] = get_market_phase(curr_growth, prev_growth)
        
        # Set the first year's phase based only on its growth
        if len(market_summary) > 0:
            first_growth = market_summary['yoy_growth'].iloc[0]
            market_summary.loc[market_summary.index[0], 'market_phase'] = get_market_phase(first_growth)
    
    # Identify transition points (places where the market phase changes)
    breakpoints = []
    if 'market_phase' in market_summary.columns:
        prev_phase = None
        
        for idx, row in market_summary.iterrows():
            curr_phase = row['market_phase']
            
            # Check for phase transition
            if prev_phase is not None and curr_phase != prev_phase:
                breakpoints.append({
                    'year': row['year'],
                    'from_phase': prev_phase,
                    'to_phase': curr_phase,
                    'price': row['median_price']
                })
            
            prev_phase = curr_phase
    
    # Algorithm for automatically detecting significant turning points in the price trend
    if len(market_summary) >= 5:
        # Identify turning points in the price trend
        price_series = market_summary['median_price'].values
        years = market_summary['year'].values
        
        # Identify peaks
        max_peaks, _ = find_peaks(price_series, distance=2)
        min_peaks, _ = find_peaks(-price_series, distance=2)
        
        # Add additional breakpoints from peak detection
        for peak in max_peaks:
            year = years[peak]
            # Check if this year isn't already in breakpoints
            if not any(bp['year'] == year for bp in breakpoints):
                breakpoints.append({
                    'year': year,
                    'from_phase': 'peak' if peak > 0 else 'unknown',
                    'to_phase': 'contraction' if peak < len(years)-1 else 'current peak',
                    'price': price_series[peak],
                    'type': 'peak'
                })
        
        for peak in min_peaks:
            year = years[peak]
            # Check if this year isn't already in breakpoints
            if not any(bp['year'] == year for bp in breakpoints):
                breakpoints.append({
                    'year': year,
                    'from_phase': 'contraction' if peak > 0 else 'unknown',
                    'to_phase': 'recovery' if peak < len(years)-1 else 'current trough',
                    'price': price_series[peak],
                    'type': 'trough'
                })
    
    # Sort breakpoints by year
    breakpoints = sorted(breakpoints, key=lambda x: x['year'])
    
    # Create colored regions for different market phases
    if 'market_phase' in market_summary.columns and len(market_summary) > 0:
        # Define phase colors
        phase_colors = {
            'expansion': 'rgba(0, 255, 0, 0.2)',     # Light green
            'peak': 'rgba(255, 165, 0, 0.2)',        # Light orange
            'stable growth': 'rgba(144, 238, 144, 0.2)',  # Light green
            'contraction': 'rgba(255, 0, 0, 0.2)',   # Light red
            'recovery': 'rgba(0, 0, 255, 0.2)',      # Light blue
            'stable decline': 'rgba(255, 192, 203, 0.2)', # Light pink
            'unknown': 'rgba(200, 200, 200, 0.1)'    # Light gray
        }
        
        # Add colored regions for each phase
        curr_phase = market_summary['market_phase'].iloc[0]
        phase_start = market_summary['year'].iloc[0]
        
        for i in range(1, len(market_summary)):
            year = market_summary['year'].iloc[i]
            phase = market_summary['market_phase'].iloc[i]
            
            # Check if phase changed
            if phase != curr_phase:
                # Add colored region for previous phase
                if curr_phase in phase_colors:
                    fig.add_vrect(
                        x0=phase_start,
                        x1=year,
                        fillcolor=phase_colors[curr_phase],
                        opacity=0.8,
                        layer="below",
                        line_width=0,
                    )
                
                # Start new phase
                curr_phase = phase
                phase_start = year
        
        # Add the last phase region
        if curr_phase in phase_colors and len(market_summary) > 0:
            fig.add_vrect(
                x0=phase_start,
                x1=market_summary['year'].iloc[-1] + 0.5,
                fillcolor=phase_colors[curr_phase],
                opacity=0.8,
                layer="below",
                line_width=0,
            )
    
    # Add annotations for breakpoints
    for bp in breakpoints:
        label = f"{bp['to_phase'].title()}"
        
        # Add vertical line
        fig.add_vline(
            x=bp['year'],
            line_width=1.5,
            line_dash="dash",
            line_color="black"
        )
        
        # Add annotation
        fig.add_annotation(
            x=bp['year'],
            y=bp['price'] * 1.05,
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            font=dict(size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )
    
    # Determine current market phase for the title
    current_phase = "Unknown"
    phase_color = "gray"
    
    if 'market_phase' in market_summary.columns and len(market_summary) > 0:
        current_phase = market_summary['market_phase'].iloc[-1]
        
        # Set color based on phase
        phase_color_map = {
            'expansion': 'green',
            'peak': 'orange',
            'stable growth': 'lightgreen',
            'contraction': 'red',
            'recovery': 'blue',
            'stable decline': 'pink',
            'unknown': 'gray'
        }
        phase_color = phase_color_map.get(current_phase, 'gray')
    
    # Add cycle durations if we have enough breakpoints
    cycle_insights = ""
    if len(breakpoints) >= 2:
        # Calculate average cycle length
        cycle_years = []
        for i in range(len(breakpoints) - 1):
            cycle_years.append(breakpoints[i+1]['year'] - breakpoints[i]['year'])
        
        if cycle_years:
            avg_cycle = sum(cycle_years) / len(cycle_years)
            cycle_insights = f"Avg. Phase Duration: {avg_cycle:.1f} years"
    
    # Create subtitle with market phase and cycle insights
    subtitle = f"Current Phase: <span style='color:{phase_color};font-weight:bold;'>{current_phase.title()}</span>"
    if cycle_insights:
        subtitle += f" | {cycle_insights}"
    
    # Add data quality indicator
    quality_indicator = f"Data Quality: {quality_metrics['quality'].upper()} ({quality_metrics['coverage']:.1f}% coverage)"
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Dubai Real Estate Market Cycles<br><span style='font-size:14px;'>{subtitle}</span>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Year",
            tickmode='linear',
            dtick=1,  # Show every year
            tickangle=45,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Median Price (AED/sqft)",
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True,
        height=600,
        # annotations=[
        #     dict(
        #         text=quality_indicator,
        #         align='left',
        #         showarrow=False,
        #         xref='paper',
        #         yref='paper',
        #         x=0,
        #         y=-0.13,
        #         bordercolor='black',
        #         borderwidth=1,
        #         borderpad=4,
        #         bgcolor=get_quality_color(quality_metrics['quality']),
        #         opacity=0.8,
        #         font=dict(size=12, color="white")
        #     )
        # ]
    )
    
    # Add legend for phase colors if we have phase data
    if 'market_phase' in market_summary.columns:
        # Get unique phases
        unique_phases = market_summary['market_phase'].unique()
        
        # Add phase legend
        for i, phase in enumerate(unique_phases):
            if phase in phase_colors:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=15, color=phase_colors[phase]),
                        name=phase.title(),
                        hoverinfo='none'
                    )
                )
    
    # Add filter description
    filter_desc = get_filter_description(property_type, area, room_type, registration_type)
    if filter_desc:
        fig.add_annotation(
            text=filter_desc,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0,
            y=1.12,
            font=dict(size=12)
        )
    
    # Extract years for breakpoints
    breakpoint_years = [bp['year'] for bp in breakpoints]
    
    return fig, breakpoint_years

def create_price_forecast(df, property_type=None, area=None, room_type=None, registration_type=None,
                         forecast_years=3, confidence_level=0.95):
    """
    Create a price forecast chart with confidence intervals
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        forecast_years (int): Number of years to forecast
        confidence_level (float): Confidence level for prediction intervals
        
    Returns:
        go.Figure: Plotly figure with price forecast
    """
    # Apply filters progressively to ensure data availability
    filtered_df = progressive_filter(df, property_type, area, room_type, registration_type)
    
    # Calculate missing growth rates if needed
    filtered_df = calculate_growth_rates(filtered_df)
    
    # Get data quality metrics
    quality_metrics = get_data_quality_metrics(filtered_df)
    
    # Get market summary
    market_summary = get_market_summary_for_filtered_data(
        filtered_df, property_type, area, room_type, registration_type
    )
    
    # Need at least 5 years for a reasonable forecast
    if len(market_summary) < 5:
        return create_no_data_figure("Price Forecast", "Insufficient historical data for forecasting (need 5+ years)")
    
    # Create forecast
    try:
        # Prepare historical data
        historical_years = market_summary['year'].values
        historical_prices = market_summary['median_price'].values
        
        # Determine last historical year and forecast years
        last_year = int(historical_years[-1])
        forecast_years_range = list(range(last_year + 1, last_year + forecast_years + 1))
        
        # Create linear model for trend forecasting
        model = LinearRegression()
        X = np.array(historical_years).reshape(-1, 1)
        y = historical_prices
        model.fit(X, y)
        
        # Make predictions for historical period (for R²) and forecast period
        historical_pred = model.predict(X)
        forecast_X = np.array(forecast_years_range).reshape(-1, 1)
        forecast_prices = model.predict(forecast_X)
        
        # Get recent growth rate for context
        recent_growth_rate = ((historical_prices[-1] / historical_prices[-2]) - 1) * 100 if len(historical_prices) >= 2 else 0
        
        # Calculate R² for model quality assessment
        r_squared = round(model.score(X, y), 2)
        
        # Calculate confidence intervals based on historical volatility
        residuals = historical_prices - historical_pred
        residual_std = np.std(residuals)
        
        # Z-score for confidence level (e.g., 1.96 for 95% CI)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Create increasing confidence intervals for forecast years
        lower_bounds = []
        upper_bounds = []
        
        for i, year in enumerate(forecast_years_range):
            # Increase uncertainty over time
            uncertainty_multiplier = 1 + (i * 0.5)  # Wider bands for later years
            
            margin = z_score * residual_std * uncertainty_multiplier
            lower_bounds.append(forecast_prices[i] - margin)
            upper_bounds.append(forecast_prices[i] + margin)
        
        # Create figure
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=historical_years,
                y=historical_prices,
                name="Historical Prices",
                line=dict(color='royalblue', width=3),
                hovertemplate="Year: %{x}<br>Price: %{y:.0f} AED/sqft<extra></extra>"
            )
        )
        
        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast_years_range,
                y=forecast_prices,
                name="Price Forecast",
                line=dict(color='green', width=3, dash='dot'),
                hovertemplate="Year: %{x}<br>Forecast: %{y:.0f} AED/sqft<extra></extra>"
            )
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast_years_range + forecast_years_range[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(0,176,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{int(confidence_level*100)}% Confidence Interval",
                hoverinfo="skip"
            )
        )
        
        # Calculate key metrics for forecast insights
        forecast_growth_pct = ((forecast_prices[-1] / historical_prices[-1]) - 1) * 100
        forecast_growth_annual = ((forecast_prices[-1] / historical_prices[-1]) ** (1/forecast_years) - 1) * 100
        
        # Update layout
        subtitle = (f"Forecast Growth: {forecast_growth_pct:.1f}% over {forecast_years} years " +
                   f"({forecast_growth_annual:.1f}% annually) | Recent Growth: {recent_growth_rate:.1f}%")
        
        # Add data quality and model quality indicators
        quality_indicator = f"Data Quality: {quality_metrics['quality'].upper()} ({quality_metrics['coverage']:.1f}% coverage) | Model R²: {r_squared}"
        
        fig.update_layout(
            title={
                'text': f"Dubai Real Estate Price Forecast ({last_year+1}-{last_year+forecast_years})<br><span style='font-size:14px;'>{subtitle}</span>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title="Year",
                tickmode='linear',
                dtick=1,  # Show every year
                tickangle=45,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Median Price (AED/sqft)",
                gridcolor='lightgray'
            ),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=True,
            height=600,
            # annotations=[
            #     dict(
            #         text=quality_indicator,
            #         align='left',
            #         showarrow=False,
            #         xref='paper',
            #         yref='paper',
            #         x=0,
            #         y=-0.13,
            #         bordercolor='black',
            #         borderwidth=1,
            #         borderpad=4,
            #         bgcolor=get_quality_color(quality_metrics['quality']),
            #         opacity=0.8,
            #         font=dict(size=12, color=get_quality_text_color(quality_metrics['quality']))
            #     )
            #]
        )
        
        # Add separator between historical and forecast periods
        fig.add_vline(
            x=last_year + 0.5,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Forecast Begins",
            annotation_position="top"
        )
        
        # Add filter description
        filter_desc = get_filter_description(property_type, area, room_type, registration_type)
        if filter_desc:
            fig.add_annotation(
                text=filter_desc,
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=1.12,
                font=dict(size=12)
            )
        
        return fig
        
    except Exception as e:
        print(f"Error creating price forecast: {e}")
        return create_no_data_figure("Price Forecast", f"Error creating forecast: {str(e)}")

def progressive_filter(df, property_type=None, area=None, room_type=None, registration_type=None, min_rows=20):
    """
    Apply filters progressively, relaxing constraints if results are too sparse
    
    Args:
        df (pd.DataFrame): The original dataframe
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter  
        registration_type (str): Registration type filter
        min_rows (int): Minimum number of rows required
        
    Returns:
        pd.DataFrame: Filtered dataframe with sufficient data
    """
    # Skip if any filter is 'All' or None
    property_type = None if property_type == 'All' else property_type
    area = None if area == 'All' else area
    room_type = None if room_type == 'All' else room_type
    registration_type = None if registration_type == 'All' else registration_type
    
    # Create filter dictionary with all filters
    filters = {}
    if property_type is not None and 'property_type_en' in df.columns:
        filters['property_type_en'] = property_type
    if area is not None and 'area_name_en' in df.columns:
        filters['area_name_en'] = area
    if room_type is not None and 'rooms_en' in df.columns:
        filters['rooms_en'] = room_type
    if registration_type is not None and 'reg_type_en' in df.columns:
        filters['reg_type_en'] = registration_type
    
    # Apply initial filters
    filtered_df = apply_filters(df, filters)
    
    # If we have enough data, return it
    if len(filtered_df) >= min_rows:
        return filtered_df
    
    print(f"Initial filtering returned only {len(filtered_df)} rows, trying progressive relaxation")
    
    # Define filter relaxation order (from least to most important to keep)
    relaxation_order = ['rooms_en', 'area_name_en', 'reg_type_en', 'property_type_en']
    
    # Iteratively relax filters
    remaining_filters = dict(filters)
    
    for filter_column in relaxation_order:
        # Skip if this filter isn't active
        if filter_column not in remaining_filters:
            continue
        
        # Create new filter set without this filter
        relaxed_filters = dict(remaining_filters)
        del relaxed_filters[filter_column]
        
        # Apply relaxed filters
        relaxed_df = apply_filters(df, relaxed_filters)
        
        print(f"Relaxed {filter_column} filter, got {len(relaxed_df)} rows")
        
        # If we have enough data now, return it
        if len(relaxed_df) >= min_rows:
            return relaxed_df
            
        # Update remaining filters for next iteration
        remaining_filters = relaxed_filters
    
    # If we've relaxed all filters and still don't have enough data,
    # return whatever we have from the most relaxed filtering
    return apply_filters(df, remaining_filters)


def apply_filters(df, filters):
    """
    Apply a dictionary of filters to a dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        filters (dict): Dictionary mapping column names to filter values
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if not filters:
        return df.copy()
        
    filtered_df = df.copy()
    
    for col, value in filters.items():
        if value is not None and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == value]
    
    return filtered_df


def get_filter_description(property_type=None, area=None, room_type=None, registration_type=None):
    """
    Create a human-readable description of applied filters
    
    Args:
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        
    Returns:
        str: Description of applied filters
    """
    filters = []
    
    if property_type not in (None, 'All'):
        filters.append(f"Property Type: {property_type}")
    
    if area not in (None, 'All'):
        filters.append(f"Area: {area}")
    
    if room_type not in (None, 'All'):
        filters.append(f"Room Type: {room_type}")
    
    if registration_type not in (None, 'All'):
        filters.append(f"Registration Type: {registration_type}")
    
    if filters:
        return "Filters: " + " | ".join(filters)
    else:
        return "Filters: None (Market-wide view)"


def get_quality_color(quality):
    """Get color for data quality indicator"""
    quality_colors = {
        'high': 'green',
        'medium': 'orange',
        'low': 'red',
        'none': 'red',
        'unknown': 'gray'
    }
    return quality_colors.get(quality, 'gray')


def get_quality_text_color(quality):
    """Get text color for data quality indicator based on background color"""
    if quality in ('high', 'medium', 'low', 'none'):
        return 'white'
    return 'black'


def create_no_data_figure(title="No Data", message="No data available for the selected filters"):
    """Create a default figure with a message when data is not available"""
    fig = go.Figure()
    
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 18
                }
            }
        ]
    )
    
    return fig


def generate_time_series_insights(df, analysis_type, property_type=None, area=None, 
                                 room_type=None, registration_type=None):
    """
    Generate insights for time series analysis
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        analysis_type (str): Type of analysis ('price_trends', 'market_cycles', 'price_forecast')
        property_type (str): Property type filter
        area (str): Area filter
        room_type (str): Room type filter
        registration_type (str): Registration type filter
        
    Returns:
        dict: Dictionary with insights
    """
    # Apply filters progressively
    filtered_df = progressive_filter(df, property_type, area, room_type, registration_type)
    
    # Calculate missing growth rates if needed
    filtered_df = calculate_growth_rates(filtered_df)
    
    # Get data quality metrics
    quality_metrics = get_data_quality_metrics(filtered_df)
    
    # Get market summary
    market_summary = get_market_summary_for_filtered_data(
        filtered_df, property_type, area, room_type, registration_type
    )
    
    # Initialize insights
    insights = {
        'data_quality': quality_metrics['quality'],
        'coverage_pct': quality_metrics['coverage'],
        'text': [],
        'metrics': {}
    }
    
    # Handle insufficient data
    if len(market_summary) < 2:
        insights['text'].append("Insufficient historical data for selected filters.")
        return insights
    
    # Generate insights based on analysis type
    if analysis_type == 'price_trends':
        # Calculate key metrics
        if len(market_summary) >= 2:
            latest_price = market_summary['median_price'].iloc[-1]
            earliest_price = market_summary['median_price'].iloc[0]
            start_year = market_summary['year'].iloc[0]
            end_year = market_summary['year'].iloc[-1]
            years_diff = end_year - start_year
            
            # Total appreciation
            total_change_pct = ((latest_price / earliest_price) - 1) * 100
            
            # CAGR
            cagr = ((latest_price / earliest_price) ** (1 / years_diff) - 1) * 100 if years_diff > 0 else 0
            
            # Recent growth
            recent_growth = None
            if 'yoy_growth' in market_summary.columns and len(market_summary) >= 2:
                recent_growth = market_summary['yoy_growth'].iloc[-1]
            
            # Volatility
            if 'yoy_growth' in market_summary.columns and len(market_summary) >= 3:
                volatility = market_summary['yoy_growth'].std()
            else:
                volatility = None
            
            insights['metrics'].update({
                'latest_price': latest_price,
                'total_change_pct': total_change_pct,
                'cagr': cagr,
                'recent_growth': recent_growth,
                'volatility': volatility,
                'years_analyzed': years_diff
            })
            
            # Generate text insights
            insights['text'].append(f"Analysis Period: {start_year} to {end_year} ({years_diff} years)")
            insights['text'].append(f"Current Price: {latest_price:.0f} AED/sqft")
            
            # Growth description based on total change
            if total_change_pct > 50:
                growth_desc = "strong appreciation"
            elif total_change_pct > 10:
                growth_desc = "moderate appreciation"
            elif total_change_pct > 0:
                growth_desc = "slight appreciation"
            elif total_change_pct > -10:
                growth_desc = "slight depreciation"
            else:
                growth_desc = "significant depreciation"
            
            insights['text'].append(f"Price Change: {total_change_pct:.1f}% {growth_desc} over the period ({cagr:.1f}% annually)")
            
            # Recent trend
            if recent_growth is not None:
                if recent_growth > 10:
                    trend_desc = "robust growth"
                elif recent_growth > 5:
                    trend_desc = "healthy growth"
                elif recent_growth > 0:
                    trend_desc = "moderate growth"
                elif recent_growth > -5:
                    trend_desc = "slight decline"
                else:
                    trend_desc = "significant decline"
                
                insights['text'].append(f"Recent Trend: {recent_growth:.1f}% {trend_desc} in the most recent year")
            
            # Market cycles
            if len(market_summary) >= 5 and 'yoy_growth' in market_summary.columns:
                # Identify current market phase
                latest_growth = market_summary['yoy_growth'].iloc[-1]
                prev_growth = market_summary['yoy_growth'].iloc[-2] if len(market_summary) > 2 else None
                current_phase = get_market_phase(latest_growth, prev_growth)
                
                insights['metrics']['current_phase'] = current_phase
                insights['text'].append(f"Current Market Phase: {current_phase.title()}")
                
                # Provide phase-specific guidance
                if current_phase in ('expansion', 'stable growth'):
                    insights['text'].append("Investment Implication: Growth momentum suggests continued price appreciation in the near term.")
                elif current_phase == 'peak':
                    insights['text'].append("Investment Implication: Market may be approaching a transition point; consider monitoring closely for potential slowdown.")
                elif current_phase in ('contraction', 'stable decline'):
                    insights['text'].append("Investment Implication: Current price declines may present buying opportunities for long-term investors, though short-term risks remain.")
                elif current_phase == 'recovery':
                    insights['text'].append("Investment Implication: Early recovery phase often presents value opportunities before broader market recognition.")
    
    elif analysis_type == 'market_cycles':
        # ADDED: Initialize these variables immediately
        avg_durations = {}
        phase_durations = {}
        # Analyze market cycles
        if len(market_summary) >= 5 and 'yoy_growth' in market_summary.columns:
            # Calculate cycle metrics
            growth_values = market_summary['yoy_growth'].dropna().values
            zero_crossings = np.where(np.diff(np.signbit(growth_values)))[0]
            
            # Identify peaks and troughs
            peaks, _ = find_peaks(growth_values, distance=2)
            troughs, _ = find_peaks(-growth_values, distance=2)
            
            # Calculate cycle characteristics
            if len(zero_crossings) >= 2:
                avg_cycle_length = (zero_crossings[-1] - zero_crossings[0]) / (len(zero_crossings) - 1)
            else:
                avg_cycle_length = None
            
            # Calculate typical durations of different phases if we can identify them
            if 'market_phase' in market_summary.columns:
                current_phase = None
                phase_start = None
                
                for idx, row in market_summary.iterrows():
                    phase = row['market_phase']
                    
                    if phase != current_phase:
                        # End of previous phase
                        if current_phase is not None and phase_start is not None:
                            duration = idx - phase_start
                            if current_phase in phase_durations:
                                phase_durations[current_phase].append(duration)
                            else:
                                phase_durations[current_phase] = [duration]
                        
                        # Start new phase
                        current_phase = phase
                        phase_start = idx
                
                # Calculate average durations
                if phase_durations:  # Only calculate if we have phase durations
                    avg_durations = {phase: sum(durations)/len(durations) 
                                  for phase, durations in phase_durations.items() if durations}
                    
                    insights['metrics']['avg_phase_durations'] = avg_durations
            
            # Latest phase info
            latest_growth = market_summary['yoy_growth'].iloc[-1]
            prev_growth = market_summary['yoy_growth'].iloc[-2] if len(market_summary) > 2 else None
            current_phase = get_market_phase(latest_growth, prev_growth)
            
            insights['metrics'].update({
                'current_phase': current_phase,
                'latest_growth': latest_growth,
                'avg_cycle_length': avg_cycle_length,
                'cycles_detected': len(zero_crossings) // 2,  # Full cycles
                'peak_count': len(peaks),
                'trough_count': len(troughs)
            })
            
            # Generate text insights
            insights['text'].append(f"Current Market Phase: {current_phase.title()}")
            
            if avg_cycle_length is not None:
                insights['text'].append(f"Average Cycle Length: Approximately {avg_cycle_length:.1f} years between transitions")
            
            # Describe the current phase and its typical duration
            if current_phase and current_phase in avg_durations:  # Now this is safe!
                avg_duration = avg_durations[current_phase]
                insights['text'].append(f"Typical {current_phase.title()} Phase Duration: {avg_duration:.1f} years based on historical patterns")
            
            # Provide cycle-specific guidance
            if current_phase in ('expansion', 'stable growth'):
                # Safely get avg duration for expansion if it exists
                exp_duration = avg_durations.get('expansion', 2)
                insights['text'].append(f"Investment Strategy: Growth phases have historically lasted {exp_duration:.1f} years. Consider balancing between securing current gains and capitalizing on continued momentum.")
            elif current_phase == 'peak':
                insights['text'].append("Investment Strategy: Market peaks typically transition to contractions within 1-2 years. Consider more defensive positions and focus on properties with strong fundamentals.")
            elif current_phase in ('contraction', 'stable decline'):
                # Safely get avg duration for contraction if it exists
                cont_duration = avg_durations.get('contraction', 2)
                insights['text'].append(f"Investment Strategy: Contractions have historically lasted {cont_duration:.1f} years. Value opportunities may emerge for counter-cyclical investors with longer horizons.")
            elif current_phase == 'recovery':
                insights['text'].append("Investment Strategy: Recovery phases often present the strongest risk-adjusted returns before broader market recognition drives prices higher.")

    elif analysis_type == 'price_forecast':
        # Generate forecast insights
        if len(market_summary) >= 5:
            # Calculate current metrics
            latest_price = market_summary['median_price'].iloc[-1]
            latest_year = market_summary['year'].iloc[-1]
            
            # Calculate recent growth metrics
            if 'yoy_growth' in market_summary.columns and len(market_summary) >= 2:
                recent_growth = market_summary['yoy_growth'].iloc[-1]
                prev_growth = market_summary['yoy_growth'].iloc[-2] if len(market_summary) > 2 else None
                
                # Detect momentum (acceleration or deceleration)
                if prev_growth is not None:
                    momentum = recent_growth - prev_growth
                else:
                    momentum = 0
            else:
                recent_growth = None
                momentum = None
            
            # Estimate forecast growth based on recent trends and momentum
            forecast_years = 3  # Default forecast horizon
            
            if recent_growth is not None:
                # Adjust forecast based on momentum (acceleration/deceleration)
                if momentum is not None and abs(momentum) > 1:
                    # Strong momentum effect
                    forecast_growth_annual = recent_growth + (momentum * 0.5)  # Dampen the momentum effect
                else:
                    # Use recent growth with regression toward historical average
                    historical_avg = market_summary['yoy_growth'].mean() if 'yoy_growth' in market_summary.columns else 5
                    forecast_growth_annual = (recent_growth * 0.7) + (historical_avg * 0.3)  # Weighted average
            else:
                # Fallback to historical CAGR if recent growth not available
                start_price = market_summary['median_price'].iloc[0]
                start_year = market_summary['year'].iloc[0]
                years_diff = latest_year - start_year
                
                forecast_growth_annual = ((latest_price / start_price) ** (1 / years_diff) - 1) * 100 if years_diff > 0 else 5
            
            # Calculate compound forecast
            forecast_growth_total = ((1 + (forecast_growth_annual / 100)) ** forecast_years - 1) * 100
            forecast_price = latest_price * (1 + forecast_growth_total / 100)
            
            # Calculate historical volatility for confidence intervals
            if 'yoy_growth' in market_summary.columns:
                volatility = market_summary['yoy_growth'].std()
            else:
                volatility = 5  # Default volatility assumption
            
            # Calculate 90% confidence interval
            z_score_90 = 1.645
            margin_of_error = z_score_90 * volatility * np.sqrt(forecast_years)
            lower_bound_pct = forecast_growth_total - margin_of_error
            upper_bound_pct = forecast_growth_total + margin_of_error
            
            lower_bound_price = latest_price * (1 + lower_bound_pct / 100)
            upper_bound_price = latest_price * (1 + upper_bound_pct / 100)
            
            insights['metrics'].update({
                'forecast_years': forecast_years,
                'forecast_year': latest_year + forecast_years,
                'forecast_growth_annual': forecast_growth_annual,
                'forecast_growth_total': forecast_growth_total,
                'forecast_price': forecast_price,
                'lower_bound_price': lower_bound_price,
                'upper_bound_price': upper_bound_price,
                'price_volatility': volatility
            })
            
            # Generate text insights
            insights['text'].append(f"Forecast Period: {latest_year+1} to {latest_year+forecast_years}")
            insights['text'].append(f"Price Forecast: {forecast_price:.0f} AED/sqft by {latest_year+forecast_years}")
            insights['text'].append(f"Projected Growth: {forecast_growth_total:.1f}% over {forecast_years} years ({forecast_growth_annual:.1f}% annually)")
            
            insights['text'].append(f"90% Confidence Interval: {lower_bound_price:.0f} to {upper_bound_price:.0f} AED/sqft")
            
            # Guidance based on forecast
            if forecast_growth_annual > 8:
                insights['text'].append("Investment Outlook: Strong growth projected, suggesting favorable market conditions for capital appreciation.")
            elif forecast_growth_annual > 3:
                insights['text'].append("Investment Outlook: Moderate growth projected, in line with historical averages for the Dubai market.")
            elif forecast_growth_annual > 0:
                insights['text'].append("Investment Outlook: Modest growth projected, suggesting focus on rental yield may be more important than capital appreciation.")
            else:
                insights['text'].append("Investment Outlook: Price declines projected, indicating caution is warranted for short-term investors.")
    
  # Add data quality context
    if quality_metrics['quality'] == 'low':
        disclaimer = "Note: Limited historical data available for selected filters. Insights should be considered preliminary."
        insights['text'].append(disclaimer)
    
    return insights