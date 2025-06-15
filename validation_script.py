"""
Simple Dashboard Calculation Verification Script
==============================================
This script manually calculates key metrics for specific filter combinations.
Output is clean and minimal for easy manual comparison with dashboard.

Usage: python simple_validation.py
Then manually set the same filters in dashboard and compare the numbers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'scripts' not in sys.path:
    sys.path.insert(0, os.path.dirname(current_dir))

from scripts.project_analysis import load_transaction_data

def manual_calculate_project_cagr(project_txns):
    """Manually calculate CAGR using exact business rules"""
    if len(project_txns) < 2:
        return 0.0, len(project_txns), 0
    
    # Sort by date
    project_txns = project_txns.sort_values('instance_date').copy()
    
    # Get project dates
    start_date = project_txns['instance_date'].min()
    end_date = project_txns['instance_date'].max()
    age_days = (end_date - start_date).days
    
    # Adaptive window system
    if age_days >= 547:
        window_length = 183
    elif age_days >= 183:
        window_length = 92
    else:
        window_length = 31
    
    # Calculate windows
    first_window_start = start_date
    first_window_end = start_date + timedelta(days=window_length)
    recent_window_start = end_date - timedelta(days=window_length)
    recent_window_end = end_date
    
    # Extract window transactions
    first_window = project_txns[
        (project_txns['instance_date'] >= first_window_start) &
        (project_txns['instance_date'] <= first_window_end)
    ]
    
    recent_window = project_txns[
        (project_txns['instance_date'] >= recent_window_start) &
        (project_txns['instance_date'] <= recent_window_end)
    ]
    
    if len(first_window) == 0 or len(recent_window) == 0:
        return 0.0, len(project_txns), age_days
    
    # Calculate median prices
    first_price = first_window['meter_sale_price_sqft'].median()
    recent_price = recent_window['meter_sale_price_sqft'].median()
    
    if first_price <= 0 or recent_price <= 0:
        return 0.0, len(project_txns), age_days
    
    # Calculate CAGR
    first_midpoint = first_window_start + timedelta(days=window_length/2)
    recent_midpoint = recent_window_start + timedelta(days=window_length/2)
    delta_days = (recent_midpoint - first_midpoint).days
    
    if delta_days <= 0:
        return 0.0, len(project_txns), age_days
    
    years = delta_days / 365.25
    try:
        cagr = ((recent_price / first_price) ** (1 / years) - 1) * 100
        if abs(cagr) > 400:
            cagr = 0.0
    except:
        cagr = 0.0
    
    return cagr, len(project_txns), age_days

def calculate_metrics_for_filters(property_type='All', area='All', developer='All', room_type='All'):
    """Calculate metrics for specific filter combination"""
    
    # Load and filter data
    txn_df = load_transaction_data()
    filtered_df = txn_df.copy()
    
    if property_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type_en'] == property_type]
    if area != 'All':
        filtered_df = filtered_df[filtered_df['area_name_en'] == area]
    if developer != 'All':
        filtered_df = filtered_df[filtered_df['developer_name'] == developer]
    if room_type != 'All':
        filtered_df = filtered_df[filtered_df['rooms_en'] == room_type]
    
    if len(filtered_df) == 0:
        return None
    
    # Calculate project-level metrics
    project_results = []
    for project_id in filtered_df['project_number_int'].unique():
        project_txns = filtered_df[filtered_df['project_number_int'] == project_id]
        project_info = project_txns.iloc[0]
        
        cagr, txn_count, age_days = manual_calculate_project_cagr(project_txns)
        
        project_results.append({
            'project_name': project_info.get('project_name_en', f'Project {project_id}'),
            'area': project_info.get('area_name_en', 'Unknown'),
            'developer': project_info.get('developer_name', 'Unknown'),
            'cagr': cagr,
            'transactions': txn_count
        })
    
    # Convert to DataFrame
    projects_df = pd.DataFrame(project_results)
    
    # Calculate aggregations
    results = {
        'total_projects': len(projects_df),
        'total_transactions': filtered_df.shape[0],
        'top_5_projects': projects_df.nlargest(5, 'cagr')[['project_name', 'cagr', 'transactions']].round(2),
    }
    
    # Area aggregations (if multiple areas)
    if len(projects_df['area'].unique()) > 1:
        area_agg = projects_df.groupby('area').agg({
            'cagr': ['count', 'mean'],
            'transactions': 'sum'
        }).round(2)
        area_agg.columns = ['project_count', 'simple_avg_cagr', 'total_transactions']
        
        # Volume-weighted average
        area_agg['weighted_avg_cagr'] = projects_df.groupby('area').apply(
            lambda x: np.average(x['cagr'], weights=x['transactions'])
        ).round(2)
        
        results['area_aggregations'] = area_agg.sort_values('weighted_avg_cagr', ascending=False)
    
    # Developer aggregations (if multiple developers)  
    if len(projects_df['developer'].unique()) > 1:
        dev_agg = projects_df.groupby('developer').agg({
            'cagr': ['count', 'mean'],
            'transactions': 'sum'
        }).round(2)
        dev_agg.columns = ['project_count', 'simple_avg_cagr', 'total_transactions']
        
        # Volume-weighted average
        dev_agg['weighted_avg_cagr'] = projects_df.groupby('developer').apply(
            lambda x: np.average(x['cagr'], weights=x['transactions'])
        ).round(2)
        
        results['developer_aggregations'] = dev_agg.sort_values('weighted_avg_cagr', ascending=False)
    
    return results

def print_clean_results(filters, results):
    """Print results in clean, easy-to-read format"""
    print("=" * 60)
    print(f"MANUAL CALCULATION RESULTS")
    print(f"Filters: {filters}")
    print("=" * 60)
    
    if results is None:
        print("❌ No data found for these filters")
        return
    
    print(f"📊 OVERVIEW:")
    print(f"   Total Projects: {results['total_projects']}")
    print(f"   Total Transactions: {results['total_transactions']:,}")
    
    print(f"\n🏆 TOP 5 PROJECTS BY CAGR:")
    for i, (_, project) in enumerate(results['top_5_projects'].iterrows(), 1):
        print(f"   {i}. {project['project_name'][:40]:40} | {project['cagr']:6.1f}% | {project['transactions']:4.0f} txns")
    
    if 'area_aggregations' in results:
        print(f"\n🌍 AREA AGGREGATIONS:")
        print(f"   {'Area':<30} | {'Projects':<8} | {'Simple':<7} | {'Weighted':<8}")
        print(f"   {'-'*30} | {'-'*8} | {'-'*7} | {'-'*8}")
        for area, row in results['area_aggregations'].iterrows():
            print(f"   {area[:30]:<30} | {row['project_count']:8.0f} | {row['simple_avg_cagr']:7.1f}% | {row['weighted_avg_cagr']:8.1f}%")
    
    if 'developer_aggregations' in results:
        print(f"\n🏢 DEVELOPER AGGREGATIONS:")
        print(f"   {'Developer':<25} | {'Projects':<8} | {'Simple':<7} | {'Weighted':<8}")
        print(f"   {'-'*25} | {'-'*8} | {'-'*7} | {'-'*8}")
        for dev, row in results['developer_aggregations'].iterrows():
            print(f"   {dev[:25]:<25} | {row['project_count']:8.0f} | {row['simple_avg_cagr']:7.1f}% | {row['weighted_avg_cagr']:8.1f}%")

def main():
    """Run validation for specific test cases"""
    
    test_cases = [
        {
            'name': 'Sobha LLC Developer',
            'filters': {'developer': 'Sobha LLC'}
        },
        {
            'name': 'Al Barsha South Fifth Area',  
            'filters': {'area': 'Al Barsha South Fifth'}
        },
        {
            'name': '1 B/R Room Type',
            'filters': {'room_type': '1 B/R'}
        }
    ]
    
    print("DASHBOARD CALCULATION VERIFICATION")
    print("Set the same filters in your dashboard and compare these numbers:")
    print()
    
    for test_case in test_cases:
        filters = test_case['filters']
        filter_str = ', '.join([f"{k.title()}={v}" for k, v in filters.items()])
        
        results = calculate_metrics_for_filters(
            property_type=filters.get('property_type', 'All'),
            area=filters.get('area', 'All'), 
            developer=filters.get('developer', 'All'),
            room_type=filters.get('room_type', 'All')
        )
        
        print_clean_results(filter_str, results)
        print()

if __name__ == "__main__":
    main()