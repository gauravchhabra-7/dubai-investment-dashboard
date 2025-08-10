import os
import sys
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import json

# Import from enhanced config module
from config import get_path, OUTPUT_DIR, IS_PRODUCTION, ENVIRONMENT, REQUIRED_COLUMNS

# Import from scripts
from scripts.data_loader import load_data

# Initialize Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
app.title = "Dubai Real Estate Analysis Dashboard"

server = app.server

def load_processed_data(fallback_to_raw=True):
    """
    Load processed data from output directory, or fall back to raw data
    
    Args:
        fallback_to_raw (bool): Whether to try loading raw data if processed isn't available
        
    Returns:
        tuple: (df, geo_df, launch_completion_df, micro_segment_df, comparative_df, time_series_df)
    """
    output_dir = OUTPUT_DIR
    processed_path = get_path('processed_data')
    
    if os.path.exists(processed_path):
        print(f"Loading processed data from {processed_path}")
        df = load_data(processed_path)
        
        # Try to load other processed datasets
        geo_df = None
        launch_completion_df = None
        micro_segment_df = None
        comparative_df = None
        time_series_df = None
        
        # Try to load geographic data if available
        try:
            geo_path = os.path.join(output_dir, 'geographic_analysis.csv')
            if os.path.exists(geo_path):
                geo_df = pd.read_csv(geo_path)
                print(f"Loaded geographic data ({len(geo_df)} areas)")
        except Exception as e:
            print(f"Error loading geographic data: {e}")
            
        # Try to load project analysis data if available
        if os.path.exists(output_dir):
            try:
                # Check for project analysis files
                project_files = ['project_analysis_apartment.csv', 'project_analysis_villa.csv', 'project_txn_analysis.csv']
                project_data_frames = []
                
                for file in project_files:
                    file_path = os.path.join(output_dir, file)
                    if os.path.exists(file_path):
                        temp_df = pd.read_csv(file_path)
                        project_data_frames.append(temp_df)
                        print(f"Loaded {file} ({len(temp_df)} records)")
                
                if project_data_frames:
                    launch_completion_df = pd.concat(project_data_frames, ignore_index=True)
                    print(f"Combined project analysis data: {len(launch_completion_df)} total records")
                    print(f"CAGR range: {launch_completion_df['cagr'].min():.1f}% to {launch_completion_df['cagr'].max():.1f}%")
                    print(f"Projects with positive CAGR: {(launch_completion_df['cagr'] > 0).sum()}")
                    
            except Exception as e:
                print(f"Error loading project analysis data: {e}")
                print(f"Please ensure main.py has been run to generate project analysis files")
                launch_completion_df = None
                
        # Try to load micro segment data if available
        micro_segment_path = os.path.join(output_dir, 'micro_segment_analysis.csv')
        if os.path.exists(micro_segment_path):
            try:
                micro_segment_df = pd.read_csv(micro_segment_path)
                print(f"Loaded micro segment data ({len(micro_segment_df)} segments)")
            except Exception as e:
                print(f"Error loading micro segment data: {e}")

        # Load time series data if available
        time_series_path = os.path.join(output_dir, 'market_summary.csv')
        if os.path.exists(time_series_path):
            try:
                time_series_df = pd.read_csv(time_series_path)
                print(f"Loaded time series market summary with {len(time_series_df)} years of data")
            except Exception as e:
                print(f"Error loading time series market summary: {e}")

        # Try to load comparative analysis data if available
        comparative_path = os.path.join(output_dir, 'comparative_analysis.csv')
        if os.path.exists(comparative_path):
            try:
                comparative_df = pd.read_csv(comparative_path)
                print(f"Loaded comparative analysis data ({len(comparative_df)} records)")
            except Exception as e:
                print(f"Error loading comparative analysis data: {e}")
        
        return df, geo_df, launch_completion_df, micro_segment_df, comparative_df, time_series_df
        
    elif fallback_to_raw:
        # Fall back to raw data
        print("Processed data not found, trying to load raw data...")
        data_dir = get_path('dashboard_data')
        
        if os.path.exists(data_dir):
            df = load_data(data_dir)
            
            # Attempt to load geographic data
            geo_df = None
            launch_completion_df = None
            micro_segment_df = None
            comparative_df = None
            time_series_df = None
            
            return df, geo_df, launch_completion_df, micro_segment_df, comparative_df, time_series_df
        else:
            print(f"Raw data not found at {data_dir}")
            return None, None, None, None, None, None
    else:
        print(f"Processed data not found at {processed_path}")
        return None, None, None, None, None, None

# Load dataset info if available
dataset_info = {}
info_path = get_path('dataset_info')
try:
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)
    else:
        print(f"Dataset info file not found at {info_path}")
except Exception as e:
    print(f"Error loading dataset info: {e}")
    
# Load data
print("Loading data for dashboard...")
df, geo_df, launch_completion_df, micro_segment_df, comparative_df, time_series_df = load_processed_data()

# TEMPORARY DEBUG CODE
print("\n" + "="*50)
print("VOLUME DEBUG")
print("="*50)
import os

# Check volume environment variables
print(f"RAILWAY_VOLUME_MOUNT_PATH: {os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')}")

# Check what's actually in /app
if os.path.exists("/app"):
    print(f"/app contents: {os.listdir('/app')}")

# Check volumes directory
if os.path.exists("/app/volumes"):
    print(f"/app/volumes contents: {os.listdir('/app/volumes')}")
    if os.path.exists("/app/volumes/data"):
        print(f"/app/volumes/data contents: {os.listdir('/app/volumes/data')}")

# Search for the CSV file
print("Searching for project_txn.csv...")
for root, dirs, files in os.walk("/app"):
    for file in files:
        if "project_txn" in file.lower():
            print(f"Found: {os.path.join(root, file)}")
print("="*50)

if df is None:
    # Create a minimal dataframe with required columns
    df = pd.DataFrame({
        'property_type_en': ['Apartment', 'Villa'],
        'area_name_en': ['Downtown Dubai', 'Dubai Marina', 'Palm Jumeirah'],
        'rooms_en': ['Studio', '1 B/R', '2 B/R', '3 B/R'],
        'reg_type_en': ['Existing Properties', 'Off-Plan Properties']
    })
    print("Using dummy data due to loading error")

# Enhanced debugging for project analysis data
if launch_completion_df is not None:
    print(f"\n=== PROJECT ANALYSIS DATA LOADED ===")
    print(f"Shape: {launch_completion_df.shape}")
    if len(launch_completion_df) > 0:
        print(f"Columns: {list(launch_completion_df.columns)}")
        print(f"Sample CAGR values: {launch_completion_df['cagr'].head().tolist()}")
        print(f"CAGR range: {launch_completion_df['cagr'].min():.1f}% to {launch_completion_df['cagr'].max():.1f}%")
    print("=====================================\n")
else:
    print(f"\n❌ PROJECT ANALYSIS DATA NOT LOADED")
    print("   This will cause the Project Analysis tab to be blank")
    print("   Please run 'python main.py' to generate the required files")
    print("=====================================\n")

# Check for asset directories and create if needed
asset_dir = 'assets'
if not os.path.exists(asset_dir):
    os.makedirs(asset_dir, exist_ok=True)
    print(f"Created assets directory at {asset_dir}")

# Import dashboard components
try:
    from dashboard.layouts import create_layout
    app.layout = create_layout(df)
except ImportError as e:
    print(f"Error importing layout: {e}")
    # Create a basic layout if the import fails
    app.layout = html.Div([
        html.H1("Dubai Real Estate Dashboard"),
        html.P("Error loading dashboard layout. Please check your installation."),
        html.Pre(str(e))
    ])

# Import and register callbacks
try:
    from dashboard.callbacks import register_callbacks
    register_callbacks(app, df, geo_df, launch_completion_df, micro_segment_df, comparative_df, time_series_df)
except ImportError as e:
    print(f"Error importing callbacks: {e}")
    print("Dashboard will have limited interactivity")

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug = not IS_PRODUCTION
    
    print("\n" + "=" * 60)
    print("Dubai Real Estate Analysis Dashboard")
    print("=" * 60)
    print(f"\nDataset: {dataset_info.get('record_count', 'Unknown')} records")
    print(f"Areas: {dataset_info.get('area_count', 'Unknown')}")
    print(f"Years: {'-'.join(map(str, dataset_info.get('year_range', ['Unknown', 'Unknown'])))}") 
    print(f"Property Types: {', '.join(dataset_info.get('property_types', ['Unknown']))}")
    print(f"Registration Types: {', '.join(dataset_info.get('registration_types', ['Unknown']))}")
    
    # Enhanced startup diagnostics
    print(f"\n=== DATA STATUS ===")
    print(f"Main DataFrame: {'✅ Loaded' if df is not None else '❌ Missing'}")
    print(f"Geographic Data: {'✅ Loaded' if geo_df is not None else '❌ Missing'}")
    print(f"Project Analysis: {'✅ Loaded' if launch_completion_df is not None else '❌ Missing'}")
    print(f"Micro Segments: {'✅ Loaded' if micro_segment_df is not None else '❌ Missing'}")
    print(f"Comparative Data: {'✅ Loaded' if comparative_df is not None else '❌ Missing'}")
    print(f"Time Series: {'✅ Loaded' if time_series_df is not None else '❌ Missing'}")
    print("===================")
    
    print("\nStarting Dashboard Server...")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
