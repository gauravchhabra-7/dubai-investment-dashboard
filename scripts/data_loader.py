import os 
import pandas as pd
from config import REQUIRED_COLUMNS

def load_data(file_path, validate_schema=True):
    """
    Load and preprocess the dataset
    
    Args:
        file_path (str): Path to the CSV file
        validate_schema (bool): Whether to validate required columns
        
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
        
        # Validate schema if requested
        if validate_schema:
            validation_result = validate_dataframe_schema(df)
            if not validation_result['valid']:
                print(f"WARNING: Schema validation issues detected:")
                for issue in validation_result['issues']:
                    print(f"  - {issue}")
                print("Some analyses may not function correctly.")
                if validation_result['critical']:
                    print("CRITICAL: Missing columns will prevent core functionality.")
                    return None
        
        # Basic preprocessing
        # Convert year columns to float (these are the columns that are numeric years)
        year_columns = sorted([col for col in df.columns 
                    if (isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2025) or
                    (isinstance(col, (int, float)) and 2000 <= col <= 2025)])

        for col in year_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure registration type column exists
        if 'reg_type_en' not in df.columns:
            print("Warning: Registration type column 'reg_type_en' not found.")
        
        # Ensure date columns are datetime
        date_columns = ['completion_date', 'project_start_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def validate_dataframe_schema(df):
    """
    Validate that the dataframe has the required columns and structure
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        dict: Validation result with keys:
            - valid (bool): Whether validation passed
            - issues (list): List of validation issues
            - critical (bool): Whether any critical issues were found
    """
    issues = []
    critical = False
    
    # Check for required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        critical = True
    
    # Check for year columns (at least 2 needed for growth calculation)
    year_columns = [col for col in df.columns if 
                (isinstance(col, (int, float)) and 2000 <= col <= 2025) or
                (isinstance(col, str) and col.isdigit() and 2000 <= int(col) <= 2025)]
    if len(year_columns) < 2:
        issues.append(f"Insufficient year columns for growth calculation. Found {len(year_columns)}, need at least 2.")
        critical = True
    
    # Check for date columns
    date_columns = ['completion_date', 'project_start_date']
    missing_date_columns = [col for col in date_columns if col not in df.columns]
    if missing_date_columns:
        issues.append(f"Missing date columns: {', '.join(missing_date_columns)}")
    
    # Check for growth columns
    growth_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('growth_')]
    if not growth_columns:
        issues.append("No growth columns found. Time series analysis may be limited.")
    
    # Check data types (basic check)
    if 'median_price_sqft' in df.columns and not pd.api.types.is_numeric_dtype(df['median_price_sqft']):
        issues.append("Column 'median_price_sqft' should be numeric.")
        
    if 'transaction_count' in df.columns and not pd.api.types.is_numeric_dtype(df['transaction_count']):
        issues.append("Column 'transaction_count' should be numeric.")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'critical': critical
    }