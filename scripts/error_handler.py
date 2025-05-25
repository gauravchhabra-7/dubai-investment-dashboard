# scripts/error_handler.py

import traceback
import logging
import os
from datetime import datetime
from dash import html, dcc
import plotly.graph_objects as go

# Set up logging
LOG_LEVEL = os.environ.get("DUBAI_DASHBOARD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Create a logger for the application
logger = logging.getLogger("dubai_dashboard")

# Define error severity levels
ERROR_CRITICAL = "critical"  # Prevents functionality completely
ERROR_WARNING = "warning"    # Functionality degraded but usable
ERROR_INFO = "info"          # Informational issues only

def log_error(error, module_name, function_name, severity=ERROR_WARNING, additional_info=None):
    """
    Log an error with standardized formatting
    
    Args:
        error (Exception): The exception that occurred
        module_name (str): Name of the module where the error occurred
        function_name (str): Name of the function where the error occurred
        severity (str): Error severity level
        additional_info (dict, optional): Additional context information
        
    Returns:
        str: Error ID for reference
    """
    # Generate a unique error ID for reference
    error_id = f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Format the error message
    error_message = f"{error_id} | {module_name}.{function_name}: {str(error)}"
    
    # Log the error with appropriate severity
    if severity == ERROR_CRITICAL:
        logger.error(error_message)
    elif severity == ERROR_WARNING:
        logger.warning(error_message)
    else:
        logger.info(error_message)
    
    # Log the stack trace for debug level and above
    if logger.getEffectiveLevel() <= logging.DEBUG:
        logger.debug(f"Stack trace for {error_id}:\n{traceback.format_exc()}")
    
    # Log additional context information if provided
    if additional_info and isinstance(additional_info, dict):
        logger.debug(f"Additional context for {error_id}: {additional_info}")
    
    return error_id

def create_error_figure(title="Error", message="An error occurred while generating this visualization", error_id=None, error_details=None, height=400):
    """
    Create a standardized error figure for visualizations
    
    Args:
        title (str): Title to display
        message (str): Main error message
        error_id (str, optional): Error ID for reference
        error_details (str, optional): Detailed error information (only shown in development)
        height (int): Height of the figure
        
    Returns:
        go.Figure: Plotly figure with error message
    """
    fig = go.Figure()
    
    # Main error message
    annotations = [
        {
            "text": message,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 16}
        }
    ]
    
    # Add error ID if provided
    if error_id:
        annotations.append({
            "text": f"Reference: {error_id}",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.4,
            "showarrow": False,
            "font": {"size": 12}
        })
    
    # Add detailed error in development environment
    if error_details and os.environ.get("DUBAI_DASHBOARD_ENV", "development") == "development":
        annotations.append({
            "text": f"Details: {error_details}",
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.3,
            "showarrow": False,
            "font": {"size": 10}
        })
    
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(240, 240, 240, 0.8)",
        height=height
    )
    
    return fig

def create_error_component(title="Error", message="An error occurred", error_id=None, error_details=None, severity=ERROR_WARNING):
    """
    Create a standardized HTML error component
    
    Args:
        title (str): Error component title
        message (str): Main error message
        error_id (str, optional): Error ID for reference
        error_details (str, optional): Detailed error information (only shown in development)
        severity (str): Error severity level
        
    Returns:
        html.Div: Dash component with error message
    """
    # Select color based on severity
    color_map = {
        ERROR_CRITICAL: "danger",
        ERROR_WARNING: "warning",
        ERROR_INFO: "info"
    }
    color = color_map.get(severity, "warning")
    
    # Create the component
    component = [
        html.H5(title, className=f"text-{color}"),
        html.P(message, className="mb-2")
    ]
    
    # Add error ID if provided
    if error_id:
        component.append(html.Small(f"Reference: {error_id}", className="text-muted d-block"))
    
    # Add detailed error in development environment
    if error_details and os.environ.get("DUBAI_DASHBOARD_ENV", "development") == "development":
        component.append(html.Pre(error_details, 
                                className="border p-2 mt-2 text-small",
                                style={"fontSize": "11px", "maxHeight": "150px", "overflow": "auto"}))
    
    return html.Div(component, className=f"alert alert-{color}")

def handle_chart_error(error, module_name, function_name, title="Visualization Error", message=None, height=400):
    """
    Standard error handler for chart visualization errors
    
    Args:
        error (Exception): The exception that occurred
        module_name (str): Name of the module where the error occurred
        function_name (str): Name of the function where the error occurred
        title (str): Title to display on the error figure
        message (str, optional): Custom error message (uses default if None)
        height (int): Height of the error figure
        
    Returns:
        tuple: (dcc.Graph component with error figure, error_id)
    """
    error_message = message or f"An error occurred while generating this visualization"
    error_id = log_error(error, module_name, function_name)
    
    # Create error figure
    error_fig = create_error_figure(
        title=title,
        message=error_message,
        error_id=error_id,
        error_details=str(error),
        height=height
    )
    
    return dcc.Graph(figure=error_fig), error_id

def handle_data_error(error, module_name, function_name, title="Data Processing Error", message=None, fallback_data=None):
    """
    Standard error handler for data processing errors
    
    Args:
        error (Exception): The exception that occurred
        module_name (str): Name of the module where the error occurred
        function_name (str): Name of the function where the error occurred
        title (str): Title to display on the error component
        message (str, optional): Custom error message (uses default if None)
        fallback_data: Default data to return if error handling fails
        
    Returns:
        tuple: (data or fallback_data, error_component, error_id)
    """
    error_message = message or f"An error occurred while processing data"
    error_id = log_error(error, module_name, function_name, severity=ERROR_CRITICAL)
    
    # Create error component
    error_component = create_error_component(
        title=title,
        message=error_message,
        error_id=error_id,
        error_details=str(error),
        severity=ERROR_CRITICAL
    )
    
    return fallback_data, error_component, error_id

def handle_callback_error(error, module_name, callback_name, outputs, default_values=None):
    """
    Helper function to handle errors in callbacks with multiple outputs
    
    Args:
        error (Exception): The exception that occurred
        module_name (str): Name of the module where the error occurred
        callback_name (str): Name of the callback function
        outputs (list): List of output names that the callback updates
        default_values (dict, optional): Dictionary mapping outputs to default values
        
    Returns:
        tuple: Returns default values for each output, with error components where appropriate
    """
    error_id = log_error(error, module_name, callback_name)
    
    results = []
    default_values = default_values or {}
    
    # For each output, provide an appropriate default or error component
    for output_name in outputs:
        if output_name.endswith("-graph"):
            # For graph outputs, return error figure
            error_fig = create_error_figure(
                title="Visualization Error",
                message=f"Could not generate visualization",
                error_id=error_id,
                error_details=str(error)
            )
            results.append(dcc.Graph(figure=error_fig))
        
        elif output_name.endswith("-insights") or output_name.endswith("-text"):
            # For insight components, return error component
            results.append(create_error_component(
                title="Analysis Error",
                message=f"Could not generate insights",
                error_id=error_id,
                error_details=str(error)
            ))
        
        else:
            # For other outputs, return default value or None
            results.append(default_values.get(output_name, None))
    
    return tuple(results)