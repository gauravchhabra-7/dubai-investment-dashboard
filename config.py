# config.py
import os
import json
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (for local development only)
load_dotenv()

def get_project_root():
    """Get project root path that works in Railway and other environments"""
    # Start with the directory of this file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For files that import from subdirectories
    if os.path.basename(root_dir) in ['scripts', 'dashboard']:
        root_dir = os.path.dirname(root_dir)
    
    # Railway-specific adjustments
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        # In Railway, use the deployment directory
        root_dir = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '/app')
    elif os.environ.get('DUBAI_DASHBOARD_ENV', 'development') == 'production':
        # Other production environments
        root_dir = os.environ.get('DUBAI_DASHBOARD_ROOT', root_dir)
    
    return root_dir

# Environment detection and configuration
class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    # Environment settings
    ENV = os.environ.get('DUBAI_DASHBOARD_ENV', 'development')
    IS_RAILWAY = bool(os.environ.get('RAILWAY_ENVIRONMENT'))
    IS_PRODUCTION = ENV == 'production' or IS_RAILWAY
    IS_STAGING = ENV == 'staging'
    IS_DEVELOPMENT = ENV == 'development' and not IS_RAILWAY
    DEBUG = os.environ.get('DUBAI_DASHBOARD_DEBUG', 'false').lower() == 'true'
    
    # Feature flags
    ENABLE_CACHING = os.environ.get('DUBAI_DASHBOARD_ENABLE_CACHING', 'true').lower() == 'true'
    USE_PRECOMPUTED_DATA = os.environ.get('DUBAI_DASHBOARD_USE_PRECOMPUTED', 'true').lower() == 'true'
    DISABLE_ERROR_DETAILS = os.environ.get('DUBAI_DASHBOARD_DISABLE_ERROR_DETAILS', 'false').lower() == 'true' or IS_PRODUCTION
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('DUBAI_DASHBOARD_LOG_LEVEL', 'INFO').upper()
    VERBOSE_DEBUG = os.environ.get('DUBAI_DASHBOARD_VERBOSE_DEBUG', 'false').lower() == 'true'
    
    # API configuration
    MAPBOX_TOKEN = os.environ.get('DUBAI_DASHBOARD_MAPBOX_TOKEN')
    
    if not MAPBOX_TOKEN and IS_DEVELOPMENT:
        print("⚠️  MAPBOX_TOKEN not found. Create .env file for local development.")
        MAPBOX_TOKEN = ""
    elif not MAPBOX_TOKEN and IS_PRODUCTION:
        print("⚠️  MAPBOX_TOKEN not found in production environment variables.")

    # Content limits
    MAX_ROWS_PER_REQUEST = int(os.environ.get('DUBAI_DASHBOARD_MAX_ROWS', '5000'))
    
    # Project paths - Railway-optimized
    PROJECT_ROOT = get_project_root()
    
    # Railway-specific path handling
    if IS_RAILWAY:
        # In Railway, use app directory and volume mounts
        DATA_DIR = os.environ.get('DUBAI_DASHBOARD_DATA_DIR', '/app/data')
        OUTPUT_DIR = os.environ.get('DUBAI_DASHBOARD_OUTPUT_DIR', '/app/output')
        LARGE_DATA_DIR = os.environ.get('DUBAI_DASHBOARD_LARGE_DATA_DIR', '/app/volumes/large_data')
    else:
        # Local development paths
        DATA_DIR = os.environ.get('DUBAI_DASHBOARD_DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
        OUTPUT_DIR = os.environ.get('DUBAI_DASHBOARD_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'output'))
        LARGE_DATA_DIR = os.environ.get('DUBAI_DASHBOARD_LARGE_DATA_DIR', os.path.join(PROJECT_ROOT, 'large_data'))
    
    # Ensure directories exist in development
    if IS_DEVELOPMENT:
        for directory in [DATA_DIR, OUTPUT_DIR, LARGE_DATA_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    # Standard file paths
    DEFAULT_PATHS = {
        'project_txn_data': os.path.join(DATA_DIR, 'project_txn.csv'),
        'geojson_file': os.path.join(DATA_DIR, 'complete_community_with_csv_names.geojson'),
        'dashboard_data': os.path.join(DATA_DIR, 'dashboard_merging_2.csv'),
        
        # Large file paths (for Railway volume)
        'large_project_data': os.path.join(LARGE_DATA_DIR, 'large_project_analysis.csv'),
        
        # Output paths
        'processed_data': os.path.join(OUTPUT_DIR, 'processed_data.csv'),
        'dataset_info': os.path.join(OUTPUT_DIR, 'dataset_info.json'),
        'project_analysis_output': os.path.join(OUTPUT_DIR, 'project_txn_analysis.csv'),
        'area_analysis_output': os.path.join(OUTPUT_DIR, 'area_txn_analysis.csv'),
        'developer_analysis_output': os.path.join(OUTPUT_DIR, 'developer_txn_analysis.csv'),
        
        # Project analysis files
        'project_analysis_apartment': os.path.join(OUTPUT_DIR, 'project_analysis_apartment.csv'),
        'project_analysis_villa': os.path.join(OUTPUT_DIR, 'project_analysis_villa.csv'),
    }
    
    @classmethod
    def get_path(cls, path_key):
        """Get file path with environment variable override support"""
        env_var = f"DUBAI_DASHBOARD_{path_key.upper()}"
        path_value = os.environ.get(env_var)
        
        # If environment variable set, use it
        if path_value:
            return path_value
        
        # If not found in environment, use default paths
        if path_key in cls.DEFAULT_PATHS:
            return cls.DEFAULT_PATHS[path_key]
        
        # If not found, raise error
        raise ValueError(f"No path defined for key: {path_key}")
    
    @classmethod
    def setup_logging(cls):
        """Configure logging based on environment settings"""
        log_level = getattr(logging, cls.LOG_LEVEL)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Create app logger
        logger = logging.getLogger("dubai_dashboard")
        
        # Set more restrictive level for third-party libraries in production
        if cls.IS_PRODUCTION or not cls.VERBOSE_DEBUG:
            for module in ['matplotlib', 'pandas', 'dash', 'plotly', 'urllib3', 'requests']:
                logging.getLogger(module).setLevel(logging.WARNING)
        
        return logger
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)"""
        if not cls.DEBUG and cls.IS_PRODUCTION:
            return
            
        print(f"Environment: {cls.ENV}")
        print(f"Railway: {cls.IS_RAILWAY}")
        print(f"Production: {cls.IS_PRODUCTION}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Cache Enabled: {cls.ENABLE_CACHING}")
        print(f"Using Precomputed Data: {cls.USE_PRECOMPUTED_DATA}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Max Rows: {cls.MAX_ROWS_PER_REQUEST}")
        
        # Print important paths
        important_paths = ['data_dir', 'output_dir', 'large_data_dir']
        for key in important_paths:
            if hasattr(cls, key.upper()):
                print(f"Path '{key}': {getattr(cls, key.upper())}")

# For backward compatibility
MAPBOX_TOKEN = EnvironmentConfig.MAPBOX_TOKEN
ENVIRONMENT = EnvironmentConfig.ENV
IS_PRODUCTION = EnvironmentConfig.IS_PRODUCTION
IS_RAILWAY = EnvironmentConfig.IS_RAILWAY
PROJECT_ROOT = EnvironmentConfig.PROJECT_ROOT
DATA_DIR = EnvironmentConfig.DATA_DIR
OUTPUT_DIR = EnvironmentConfig.OUTPUT_DIR
LARGE_DATA_DIR = EnvironmentConfig.LARGE_DATA_DIR

# Export the get_path function for backward compatibility
get_path = EnvironmentConfig.get_path

# Initialize logger
logger = EnvironmentConfig.setup_logging()

# Data schema validation config
REQUIRED_COLUMNS = [
    'property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en',
    'median_price_sqft', 'transaction_count'
]

# Print configuration in development mode
if EnvironmentConfig.DEBUG or EnvironmentConfig.IS_DEVELOPMENT:
    EnvironmentConfig.print_config()

# Railway-specific startup message
if EnvironmentConfig.IS_RAILWAY:
    print("🚂 Running on Railway!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Large data directory: {LARGE_DATA_DIR}")
