# scripts/segmentation_analysis.py
import pandas as pd
import numpy as np

def create_micro_segments(df):
    """
    Create micro-segments by combining property type, registration type,
    room type, developer, and area.
    """
    # Create a micro-segment identifier
    df['micro_segment'] = (
        df['property_type_en'] + ' | ' +
        df['reg_type_en'] + ' | ' +
        df['rooms_en'] + ' | ' +
        df['area_name_en']
    )
    
    # Create a more granular segment with developer included
    df['micro_segment_with_dev'] = (
        df['property_type_en'] + ' | ' +
        df['reg_type_en'] + ' | ' +
        df['rooms_en'] + ' | ' +
        df['area_name_en'] + ' | ' +
        df['developer_name'].fillna('Unknown Developer')
    )
    
    return df

def calculate_performance_metrics(df):
    """
    Calculate performance metrics for different time horizons.
    Uses the growth columns directly instead of trying to detect year columns.
    """
    print("Calculating performance metrics...")
    
    # List all columns that start with 'growth_'
    growth_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('growth_')]
    print(f"Found {len(growth_cols)} growth columns.")
    
    if growth_cols:
        # Sort the growth columns to find the most recent ones
        sorted_growth_cols = sorted(growth_cols)
        
        # For short-term growth, use the most recent growth column
        latest_growth = sorted_growth_cols[-1]
        df['short_term_growth'] = df[latest_growth]
        print(f"Using {latest_growth} for short-term growth")
        
        # For medium-term, use an earlier period if available
        if len(sorted_growth_cols) >= 3:
            med_term_growth = sorted_growth_cols[-3]
            df['medium_term_growth'] = df[med_term_growth]
            print(f"Using {med_term_growth} for medium-term growth")
        
        # Check for CAGR columns for long-term analysis
        cagr_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('CAGR_')]
        if cagr_cols:
            latest_cagr = sorted(cagr_cols)[-1]
            df['long_term_cagr'] = df[latest_cagr]
            print(f"Using {latest_cagr} for long-term CAGR")
    else:
        print("Warning: No growth columns found.")
    
    return df

def rank_segments(df):
    """
    Rank micro-segments by performance across different time horizons.
    """
    # Rank segments by growth rates if columns exist
    growth_metrics = ['short_term_growth', 'medium_term_growth', 'long_term_cagr']
    existing_metrics = [metric for metric in growth_metrics if metric in df.columns]
    
    for metric in existing_metrics:
        print(f"Ranking segments by {metric}...")
        # Overall ranking
        df[f'{metric}_rank'] = df.groupby(['property_type_en', 'reg_type_en'])[metric].rank(
            ascending=False, method='dense', na_option='bottom'
        )
        
        # Area-specific ranking
        df[f'{metric}_rank_in_area'] = df.groupby(['area_name_en', 'property_type_en', 'reg_type_en'])[metric].rank(
            ascending=False, method='dense', na_option='bottom'
        )
        
        # Developer-specific ranking
        df[f'{metric}_rank_by_developer'] = df.groupby(['developer_name', 'property_type_en', 'reg_type_en'])[metric].rank(
            ascending=False, method='dense', na_option='bottom'
        )
    
    return df

def identify_emerging_segments(df):
    """
    Identify micro-segments showing recent acceleration in price growth.
    """
    # Only proceed if we have the necessary growth columns
    if 'short_term_growth' in df.columns and 'medium_term_growth' in df.columns:
        print("Identifying emerging segments...")
        # Calculate acceleration (difference between recent and previous growth)
        df['growth_acceleration'] = df['short_term_growth'] - df['medium_term_growth']
        
        # Flag emerging segments (high recent growth + positive acceleration)
        growth_threshold = df['short_term_growth'].quantile(0.75)
        df['is_emerging'] = (
            (df['short_term_growth'] > growth_threshold) & 
            (df['growth_acceleration'] > 0)
        )
        
        print(f"Identified {df['is_emerging'].sum()} emerging segments.")
    else:
        print("Warning: Cannot identify emerging segments without sufficient growth metrics.")
    
    return df

def perform_micro_segmentation_analysis(df):
    """Main function to perform micro-segmentation analysis."""
    print("Creating micro-segments...")
    # Create micro-segments
    df = create_micro_segments(df)
    
    # Calculate performance metrics
    df = calculate_performance_metrics(df)
    
    # Rank segments
    df = rank_segments(df)
    
    # Identify emerging segments
    df = identify_emerging_segments(df)
    
    return df

def display_top_segments(df, n=20, metric='short_term_growth'):
    """Display top performing segments based on the specified metric."""
    if metric not in df.columns:
        return pd.DataFrame({"Error": [f"Metric {metric} not found in dataframe"]})
    
    # Get top segments
    top_segments = df.dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(n)
    
    # Select relevant columns for display
    available_cols = df.columns.tolist()
    display_cols = [
        'micro_segment', 'transaction_year', 'median_price_sqft', 
        'transaction_count', metric
    ]
    
    # Add rank column if available
    rank_col = f'{metric}_rank'
    if rank_col in available_cols:
        display_cols.append(rank_col)
    
    display_cols = [col for col in display_cols if col in available_cols]
    
    return top_segments[display_cols]