import pandas as pd
import numpy as np
import os
import json
import traceback
from config import get_path, ENVIRONMENT, IS_PRODUCTION, OUTPUT_DIR

# Import data loader
from scripts.data_loader import load_data

# Import analysis modules with error handling
try:
    from scripts.geographic_analysis import prepare_geographic_data, load_kml_file
    has_geo_analysis = True
except ImportError as e:
    print(f"Warning: Geographic analysis import error: {e}")
    has_geo_analysis = False

try:
    from scripts.investment_analysis import perform_microsegment_analysis
    has_investment_analysis = True
except ImportError as e:
    print(f"Warning: Investment analysis import error: {e}")
    has_investment_analysis = False

try:
    from scripts.project_analysis import prepare_project_data
    has_project_analysis = True
except ImportError as e:
    print(f"Warning: Project analysis import error: {e}")
    has_project_analysis = False

try:
    from scripts.segmentation_analysis import perform_micro_segmentation_analysis
    has_segmentation_analysis = True
except ImportError as e:
    print(f"Warning: Segmentation analysis import error: {e}")
    has_segmentation_analysis = False

try:
    from scripts.comparative_analysis import perform_comparative_analysis
    has_comparative_analysis = True
except ImportError as e:
    print(f"Warning: Comparative analysis import error: {e}")
    has_comparative_analysis = False

# Import time series analysis module
try:
    from scripts.time_series_analysis import (
        calculate_growth_rates, 
        extract_market_summary,
        create_price_trends_chart,
        detect_market_cycles,
        create_price_forecast
    )
    has_time_series_analysis = True
except ImportError as e:
    print(f"Warning: Time series analysis import error: {e}")
    has_time_series_analysis = False

def ensure_output_dir(output_dir='output'):
    """Ensure output directory exists"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def run_analysis():
    """Run the full analysis pipeline"""
    print("=" * 60)
    print("Starting Dubai Real Estate Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data_path = get_path('dashboard_data')
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return False
        
    df = load_data(data_path, validate_schema=True)
    if df is None:
        print("Error loading data or critical schema validation issues")
        return False
        
    # Log schema validation passed
    print("Schema validation complete - proceeding with analysis")
        
    print(f"Successfully loaded {len(df):,} records with {len(df.columns)} columns")
    
    # Create output directory
    output_dir = ensure_output_dir()
    
    # Check for date columns and fix if needed
    date_columns = ['project_start_date', 'completion_date']
    for col in date_columns:
        if col in df.columns:
            if df[col].dtype != 'datetime64[ns]':
                print(f"Converting {col} to datetime...")
                df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            print(f"Warning: {col} not found in dataset")
            
    # Save processed data for dashboard to use
    processed_path = os.path.join(output_dir, 'processed_data.csv')
    print(f"\nSaving processed data to {processed_path}")
    df.to_csv(processed_path, index=False)
    
    # Save dataset info as JSON for dashboard to use
    dataset_info = {
        'record_count': len(df),
        'column_count': len(df.columns),
        'property_types': sorted(df['property_type_en'].unique().tolist()),
        'registration_types': sorted(df['reg_type_en'].unique().tolist()),
        'room_types': sorted(df['rooms_en'].unique().tolist()),
        'area_count': df['area_name_en'].nunique(),
        'year_range': [
            int(min([int(col) for col in df.columns if str(col).isdigit() and 2000 <= int(col) <= 2025])),
            int(max([int(col) for col in df.columns if str(col).isdigit() and 2000 <= int(col) <= 2025]))
        ] if any(str(col).isdigit() and 2000 <= int(col) <= 2025 for col in df.columns) else [2000, 2024]
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f)
    
    # 1. Investment Analysis
    if has_investment_analysis:
        print("\nPerforming investment analysis...")
        try:
            # Run for all data - unpack the tuple properly
            investment_results, metadata = perform_microsegment_analysis(df)
            investment_results.to_csv(os.path.join(output_dir, 'investment_opportunities.csv'), index=False)
            print(f"Created investment analysis with {len(investment_results)} microsegments")
            
            # Run for each property type
            for prop_type in df['property_type_en'].unique():
                print(f"  - Analyzing {prop_type}...")
                prop_results, prop_metadata = perform_microsegment_analysis(df, {'property_type_en': prop_type})
                prop_results.to_csv(
                    os.path.join(output_dir, f'investment_{prop_type.lower().replace(" ", "_")}.csv'), 
                    index=False
                )
            
            # Run for each registration type
            for reg_type in df['reg_type_en'].unique():
                print(f"  - Analyzing {reg_type}...")
                reg_results, reg_metadata = perform_microsegment_analysis(df, {'reg_type_en': reg_type})
                reg_results.to_csv(
                    os.path.join(output_dir, f'investment_{reg_type.lower().replace(" ", "_").replace("-", "_")}.csv'), 
                    index=False
                )
                
        except Exception as e:
            print(f"Error in investment analysis: {e}")
    
    # 2. Geographic Analysis
    if has_geo_analysis:
        print("\nPerforming geographic analysis...")
        try:
            # Directly use the custom GeoJSON file
            custom_geojson_path = get_path('geojson_file')
            if os.path.exists(custom_geojson_path):
                print(f"Using custom GeoJSON file: {custom_geojson_path}")
                geo_data = load_kml_file(custom_geojson_path)
                if geo_data is not None:
                    # Process and save geo data
                    areas = prepare_geographic_data(df, geo_df=geo_data)
                    if areas is not None:
                        areas.to_csv(os.path.join(output_dir, 'geographic_analysis.csv'), index=False)
                        print(f"Created geographic analysis for {len(areas)} areas")
                    else:
                        print("Error: prepare_geographic_data returned None")
                else:
                    print("Error: GeoJSON data could not be loaded")
            else:
                print("Custom GeoJSON file not found")
        except Exception as e:
            print(f"Error in geographic analysis: {e}")
        
    # 3. Project Analysis (replacing Launch to Completion)
    if has_project_analysis:
        print("\nPerforming project analysis...")
        try:
            # Load the new project dataset
            project_data_path = get_path('launch_completion_data')
            if not os.path.exists(project_data_path):
                # Try alternative path
                project_data_path = os.path.join(os.path.dirname(data_path), 'df_launch_completion_analysis.csv')
            
            if os.path.exists(project_data_path):
                print(f"Loading project data from {project_data_path}")
                project_df = pd.read_csv(project_data_path)
                print(f"Loaded {len(project_df)} project segments")
                
                # Enhance the project data
                enhanced_project_df = prepare_project_data(project_df)
                print(f"Enhanced project data with peer volatility and market outperformance metrics")
                
                # Save the enhanced data
                output_path = os.path.join(output_dir, 'launch_completion.csv')
                enhanced_project_df.to_csv(output_path, index=False)
                print(f"Saved enhanced project data with {len(enhanced_project_df)} project segments")
                
                # Additional analysis by property type
                for prop_type in enhanced_project_df['property_type_en'].unique():
                    prop_df = enhanced_project_df[enhanced_project_df['property_type_en'] == prop_type]
                    if len(prop_df) >= 10:  # Only save if meaningful sample size
                        prop_path = os.path.join(output_dir, f'project_analysis_{prop_type.lower().replace(" ", "_")}.csv')
                        prop_df.to_csv(prop_path, index=False)
                        print(f"  - Saved {prop_type} project analysis with {len(prop_df)} projects")
            else:
                print(f"Project data file not found at {project_data_path}")
                print("Skipping project analysis")
        except Exception as e:
            print(f"Error in project analysis: {e}")
            traceback.print_exc()
    
    # 4. Micro-Segmentation Analysis (New)
    if has_segmentation_analysis:
        print("\nPerforming micro-segmentation analysis...")
        try:
            # Run the micro-segmentation analysis
            micro_segment_df = perform_micro_segmentation_analysis(df)
            
            # Save the micro-segmentation results
            output_path = os.path.join(output_dir, 'micro_segment_analysis.csv')
            micro_segment_df.to_csv(output_path, index=False)
            print(f"Saved micro-segmentation analysis with {len(micro_segment_df)} rows")
            
            # Save a separate file for emerging segments if available
            if 'is_emerging' in micro_segment_df.columns:
                emerging_segments = micro_segment_df[micro_segment_df['is_emerging']]
                if len(emerging_segments) > 0:
                    emerging_path = os.path.join(output_dir, 'emerging_segments.csv')
                    emerging_segments.to_csv(emerging_path, index=False)
                    print(f"Saved {len(emerging_segments)} emerging segments")
        except Exception as e:
            print(f"Error in micro-segmentation analysis: {e}")
    
    # 5. Comparative Analysis 
    if has_comparative_analysis:
        print("\nPerforming comparative analysis...")
        try:
            # Run the comparative analysis
            comparative_df = perform_comparative_analysis(df)
            
            # Save the comparative analysis results
            output_path = os.path.join(output_dir, 'comparative_analysis.csv')
            comparative_df.to_csv(output_path, index=False)
            print(f"Saved comparative analysis with {len(comparative_df)} rows")
            
            # Save a separate file for consistent outperformers if available
            if 'is_consistent_outperformer' in comparative_df.columns:
                outperformers = comparative_df[comparative_df['is_consistent_outperformer']]
                if len(outperformers) > 0:
                    outperformers_path = os.path.join(output_dir, 'consistent_outperformers.csv')
                    outperformers.to_csv(outperformers_path, index=False)
                    print(f"Saved {len(outperformers)} consistent outperformers")
        except Exception as e:
            print(f"Error in comparative analysis: {e}")

    # 6. Time Series Analysis - Enhanced implementation
    if has_time_series_analysis:
        print("\nPerforming time series analysis...")
        try:
            # Calculate standardized growth rates
            df_with_growth = calculate_growth_rates(df)
            print(f"Calculated and standardized growth rates")
            
            # Generate market summary
            market_summary = extract_market_summary(df_with_growth)
            if len(market_summary) > 0:
                market_summary_path = os.path.join(output_dir, 'market_summary.csv')
                market_summary.to_csv(market_summary_path, index=False)
                print(f"Saved market summary with {len(market_summary)} years of data")
            else:
                print("Warning: No market summary data generated")
            
            # Create segmented market summaries for key segments
            print("Generating segment-specific market summaries...")
            
            # By property type
            for prop_type in df['property_type_en'].unique():
                filtered_df = df[df['property_type_en'] == prop_type]
                segment_summary = extract_market_summary(filtered_df)
                
                if len(segment_summary) > 0:
                    segment_path = os.path.join(output_dir, f'market_summary_{prop_type.lower().replace(" ", "_")}.csv')
                    segment_summary.to_csv(segment_path, index=False)
                    print(f"  - Saved market summary for {prop_type} with {len(segment_summary)} years")
            
            # By registration type
            for reg_type in df['reg_type_en'].unique():
                filtered_df = df[df['reg_type_en'] == reg_type]
                segment_summary = extract_market_summary(filtered_df)
                
                if len(segment_summary) > 0:
                    segment_path = os.path.join(output_dir, f'market_summary_{reg_type.lower().replace(" ", "_").replace("-", "_")}.csv')
                    segment_summary.to_csv(segment_path, index=False)
                    print(f"  - Saved market summary for {reg_type} with {len(segment_summary)} years")
            
            # Create baseline visualizations for documentation
            try:
                print("Generating baseline time series visualizations...")
                
                # Create price trends chart
                fig_trends = create_price_trends_chart(df_with_growth)
                
                # Create market cycles chart
                fig_cycles, breakpoints = detect_market_cycles(df_with_growth)
                
                # Create price forecast
                fig_forecast = create_price_forecast(df_with_growth)
                
                print("Successfully generated baseline time series visualizations")
                
                # Save breakpoints data for reference
                if breakpoints:
                    bp_data = pd.DataFrame({'breakpoint_year': breakpoints})
                    bp_path = os.path.join(output_dir, 'market_breakpoints.csv')
                    bp_data.to_csv(bp_path, index=False)
                    print(f"Saved {len(breakpoints)} market breakpoints")
            except Exception as e:
                print(f"Error generating baseline visualizations: {e}")
            
            # Save processed data with standardized growth rates
            df_with_growth.to_csv(os.path.join(output_dir, 'processed_data_with_growth.csv'), index=False)
            print("Saved processed data with standardized growth rates")
            
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            traceback.print_exc()

    return True

if __name__ == "__main__":
    run_analysis()