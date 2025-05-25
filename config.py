# config.py
import os
import json
import sys
import logging
from dotenv import load_dotenv
load_dotenv()

# Determine project root (keeping your existing function)
def get_project_root():
    """Get project root path that works in any environment"""
    # Start with the directory of this file
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # For files that import from subdirectories
    if os.path.basename(root_dir) in ['scripts', 'dashboard']:
        root_dir = os.path.dirname(root_dir)
        
    # In production (Netlify), use the base directory
    if os.environ.get('DUBAI_DASHBOARD_ENV', 'development') == 'production':
        # Override with environment variable if provided
        root_dir = os.environ.get('DUBAI_DASHBOARD_ROOT', root_dir)
    
    return root_dir

# Environment detection and configuration
class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    # Environment settings
    ENV = os.environ.get('DUBAI_DASHBOARD_ENV', 'development')
    IS_PRODUCTION = ENV == 'production'
    IS_STAGING = ENV == 'staging'
    IS_DEVELOPMENT = ENV == 'development'
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
    
    if not MAPBOX_TOKEN:
        print("⚠️  MAPBOX_TOKEN not found. Create .env file for local development.")
        MAPBOX_TOKEN = ""

    

    # Content limits
    MAX_ROWS_PER_REQUEST = int(os.environ.get('DUBAI_DASHBOARD_MAX_ROWS', '5000'))
    
    # Project paths
    PROJECT_ROOT = get_project_root()
    DATA_DIR = os.environ.get('DUBAI_DASHBOARD_DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
    OUTPUT_DIR = os.environ.get('DUBAI_DASHBOARD_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'output'))
    
    # Ensure directories exist in development
    if IS_DEVELOPMENT:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Standard file paths
    DEFAULT_PATHS = {
        'launch_completion_data': os.path.join(DATA_DIR, 'data/df_launch_completion_analysis.csv'),
        'geojson_file': os.path.join(DATA_DIR, 'complete_community_with_csv_names.geojson'),
        'dashboard_data': os.path.join(DATA_DIR, 'dashboard_merging_2.csv'),
        'processed_data': os.path.join(OUTPUT_DIR, 'processed_data.csv'),
        'dataset_info': os.path.join(OUTPUT_DIR, 'dataset_info.json')
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
        
        # Set more restrictive level for third-party libraries
        if not cls.VERBOSE_DEBUG:
            for module in ['matplotlib', 'pandas', 'dash', 'plotly']:
                logging.getLogger(module).setLevel(logging.WARNING)
        
        return logger
    
    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)"""
        if not cls.DEBUG and not cls.IS_DEVELOPMENT:
            return
            
        print(f"Environment: {cls.ENV}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Cache Enabled: {cls.ENABLE_CACHING}")
        print(f"Using Precomputed Data: {cls.USE_PRECOMPUTED_DATA}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Max Rows: {cls.MAX_ROWS_PER_REQUEST}")
        
        # Print file paths
        for key, path in cls.DEFAULT_PATHS.items():
            print(f"Path '{key}': {cls.get_path(key)}")

# For backward compatibility
MAPBOX_TOKEN = EnvironmentConfig.MAPBOX_TOKEN
ENVIRONMENT = EnvironmentConfig.ENV
IS_PRODUCTION = EnvironmentConfig.IS_PRODUCTION
PROJECT_ROOT = EnvironmentConfig.PROJECT_ROOT
DATA_DIR = EnvironmentConfig.DATA_DIR
OUTPUT_DIR = EnvironmentConfig.OUTPUT_DIR

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