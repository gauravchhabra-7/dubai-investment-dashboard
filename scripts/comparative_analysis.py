# scripts/comparative_analysis.py
import pandas as pd
import numpy as np

def create_performance_benchmarks(df):
    """
    Create market benchmarks and calculate relative performance metrics.
    """
    print("Creating performance benchmarks...")
    
    # Calculate market averages for growth metrics
    growth_metrics = ['short_term_growth', 'medium_term_growth', 'long_term_cagr']
    existing_metrics = [metric for metric in growth_metrics if metric in df.columns]
    
    for metric in existing_metrics:
        # Calculate market median for this metric
        market_avg = df[metric].median()
        df[f'rel_{metric}'] = df[metric] - market_avg
        print(f"Market median for {metric}: {market_avg:.2f}%")
    
    return df

def calculate_segment_premium(df):
    """
    Calculate price premium/discount for each segment relative to similar segments.
    """
    print("Calculating segment premiums...")
    
    if 'median_price_sqft' in df.columns:
        try:
            # 1. Calculate property type premium
            property_type_avg = df.groupby('property_type_en')[['median_price_sqft']].transform('median')
            df['property_type_premium'] = (df['median_price_sqft'] / property_type_avg - 1) * 100
            
            # 2. Calculate area premium
            area_property_avg = df.groupby(['area_name_en', 'property_type_en'])[['median_price_sqft']].transform('median')
            df['area_premium'] = (df['median_price_sqft'] / area_property_avg - 1) * 100
            
            # 3. Calculate developer premium (only where developer is not null)
            df_with_dev = df.dropna(subset=['developer_name'])
            developer_avg = df_with_dev.groupby(['developer_name', 'property_type_en'])[['median_price_sqft']].transform('median')
            df.loc[df.index.isin(df_with_dev.index), 'developer_premium'] = (
                df_with_dev['median_price_sqft'] / developer_avg - 1
            ) * 100
            
            # 4. Calculate room type premium
            room_property_avg = df.groupby(['rooms_en', 'property_type_en'])[['median_price_sqft']].transform('median')
            df['room_type_premium'] = (df['median_price_sqft'] / room_property_avg - 1) * 100
            
            print("Segment premiums calculated successfully.")
        except Exception as e:
            print(f"Error calculating segment premiums: {e}")
    else:
        print("Warning: Cannot calculate premiums without median_price_sqft column.")
    
    return df

def identify_consistent_outperformers(df):
    """
    Identify segments that consistently outperform the market.
    """
    print("Identifying consistent outperformers...")
    
    # Get relative performance columns
    rel_perf_cols = [col for col in df.columns if col.startswith('rel_') and not col.endswith('premium')]
    
    if len(rel_perf_cols) > 0:
        # Calculate outperformance frequency (% of available periods)
        df['outperformance_count'] = df[rel_perf_cols].gt(0).sum(axis=1)
        df['available_periods'] = df[rel_perf_cols].notna().sum(axis=1)
        df['outperformance_ratio'] = df['outperformance_count'] / df['available_periods'].replace(0, 1)
        
        # Flag consistent outperformers (outperforming in at least 75% of periods)
        df['is_consistent_outperformer'] = df['outperformance_ratio'] >= 0.75
        
        print(f"Identified {df['is_consistent_outperformer'].sum()} consistent outperformers.")
    else:
        print("Warning: No relative performance metrics found for outperformer identification.")
    
    return df

def add_statistical_significance(df):
    """
    Add statistical significance indicators for performance differences.
    """
    print("Adding statistical significance indicators...")
    
    # For segments with enough transaction history, calculate confidence intervals
    if 'short_term_growth' in df.columns and 'transaction_count' in df.columns:
        try:
            # Calculate standard error based on transaction count
            # More transactions = lower error
            stderr = df['short_term_growth'].std() / np.sqrt(df['transaction_count'])
            df['short_term_growth_stderr'] = stderr
            
            # 95% confidence interval
            df['short_term_growth_ci_lower'] = df['short_term_growth'] - 1.96 * stderr
            df['short_term_growth_ci_upper'] = df['short_term_growth'] + 1.96 * stderr
            
            # Flag if statistically significant vs market average
            market_avg = df['short_term_growth'].median()
            df['short_term_growth_significant'] = (
                (df['short_term_growth_ci_lower'] > market_avg) | 
                (df['short_term_growth_ci_upper'] < market_avg)
            )
            
            print("Statistical significance analysis completed.")
        except Exception as e:
            print(f"Error in statistical significance calculation: {e}")
    else:
        print("Warning: Cannot calculate statistical significance without required columns.")
    
    return df

def perform_comparative_analysis(df):
    """Main function to perform comparative analysis."""
    # Create performance benchmarks
    df = create_performance_benchmarks(df)
    
    # Calculate segment premium/discount
    df = calculate_segment_premium(df)
    
    # Identify consistent outperformers
    df = identify_consistent_outperformers(df)
    
    # Add statistical significance
    df = add_statistical_significance(df)
    
    return df