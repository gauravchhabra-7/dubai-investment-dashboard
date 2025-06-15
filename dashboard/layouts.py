import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
import os

def create_sidebar_filters(df, tab_id="investment"):
    """
    Create sidebar filter section for specific tab
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        tab_id (str): ID prefix for the current tab
        
    Returns:
        dash component: A sidebar component containing all filters
    """
    # Get available columns and their unique values
    available_columns = df.columns.tolist()
    
    # Prepare filter options with safeguards for missing columns
    property_types = sorted(df['property_type_en'].unique()) if 'property_type_en' in available_columns else ['Apartment', 'Villa']
    areas = sorted(df['area_name_en'].unique()) if 'area_name_en' in available_columns else ['Downtown Dubai', 'Dubai Marina', 'Palm Jumeirah']
    room_types = sorted(df['rooms_en'].unique()) if 'rooms_en' in available_columns else ['Studio', '1 B/R', '2 B/R', '3 B/R', '4 B/R', '5 B/R']
    reg_types = sorted(df['reg_type_en'].unique()) if 'reg_type_en' in available_columns else ['Existing Properties', 'Off-Plan Properties']
    
    # Get developer names if available
    developers = []
    if 'developer_name' in available_columns:
        developers = sorted(df['developer_name'].unique())
    else:
        developers = ['Emaar', 'Nakheel', 'Dubai Properties', 'Damac', 'Meraas']
    
    # Find year columns
    year_columns = [col for col in df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025]
    has_years = len(year_columns) > 0
    
    # Define standardized filter IDs with tab-specific prefixes
    property_type_id = f"{tab_id}-property-type-filter"
    area_id = f"{tab_id}-area-filter"
    room_id = f"{tab_id}-room-type-filter"
    registration_id = f"{tab_id}-registration-type-filter"
    developer_id = f"{tab_id}-developer-filter"
    year_id = f"{tab_id}-year-filter"
    
    # Common filter components
    common_filters = []
    
    # Property Type Filter (all tabs have this)
    common_filters.append(html.Div([
        html.Label("Property Type", className="fw-bold mb-2"),
        dcc.Dropdown(
            id=property_type_id,
            options=[{'label': 'All', 'value': 'All'}] + 
                    [{'label': p, 'value': p} for p in property_types],
            value='All',
            placeholder="Select property type...",
            className="mb-3 expandable-dropdown",
            style={'position': 'relative', 'zIndex': 1000}
        )
    ]))
    
    # Tab-specific filter combinations
    if tab_id == "investment":
        # Investment Analysis: Property Type, Area, Room Type, Registration Type, Time Horizon
        common_filters.extend([
            html.Div([
                html.Label("Area", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=area_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': a, 'value': a} for a in areas[:100]],
                    value='All',
                    placeholder="Select areas...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 999}
                )
            ]),
            html.Div([
                html.Label("Room Configuration", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=room_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in room_types],
                    value='All',
                    placeholder="Select room types...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 998}
                )
            ]),
            html.Div([
                html.Label("Registration Type", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=registration_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in reg_types],
                    value='All',
                    placeholder="Select registration types...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 997}
                )
            ]),
            html.Div([
                html.Label("Time Horizon", className="fw-bold mb-2"),
                dcc.RadioItems(
                    id=f"{tab_id}-time-horizon-filter",
                    options=[
                        {"label": "Short-term (1 year)", "value": "short_term_growth"},
                        {"label": "Medium-term (3 years)", "value": "medium_term_growth"},
                        {"label": "Long-term (5+ years)", "value": "long_term_cagr"}
                    ],
                    value="short_term_growth",
                    className="mb-3 vertical-radio",
                    style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}
                )
            ])
        ])
        
    elif tab_id == "ts":
        # Time Series: Property Type, Area, Room Type, Registration Type
        common_filters.extend([
            html.Div([
                html.Label("Area", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=area_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': a, 'value': a} for a in areas],
                    value='All',
                    placeholder="Select area...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 999}
                )
            ]),
            html.Div([
                html.Label("Room Configuration", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=room_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in room_types],
                    value='All',
                    placeholder="Select room type...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 998}
                )
            ]),
            html.Div([
                html.Label("Registration Type", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=registration_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in reg_types],
                    value='All',
                    placeholder="Select registration type...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 997}
                )
            ])
        ])
        
    elif tab_id == "geo":
        # Geographic: Property Type, Room Type, Registration Type, Year
        common_filters.extend([
            html.Div([
                html.Label("Room Configuration", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=room_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in room_types],
                    value='All',
                    placeholder="Select room type...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 999}
                )
            ]),
            html.Div([
                html.Label("Registration Type", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=registration_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in reg_types],
                    value='All',
                    placeholder="Select registration type...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 998}
                )
            ]),
            html.Div([
                html.Label("Analysis Year", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=year_id,
                    options=[{'label': str(int(y)), 'value': y} for y in sorted(year_columns)] if has_years else [{'label': 'Current', 'value': 'current'}],
                    value=max(year_columns) if has_years and year_columns else 'current',
                    placeholder="Select year...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 997}
                )
            ])
        ])
        
    elif tab_id == "ltc":
        # Project Analysis: Property Type, Area, Developer, Room Type (ALL FOUR FILTERS)
        common_filters.extend([
            html.Div([
                html.Label("Area", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=area_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': a, 'value': a} for a in areas],
                    value='All',
                    placeholder="Select area...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 999}
                )
            ]),
            html.Div([
                html.Label("Developer", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=developer_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': d, 'value': d} for d in developers],
                    value='All',
                    placeholder="Select developer...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 998}
                )
            ]),
            html.Div([
                html.Label("Room Configuration", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=room_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in room_types],
                    value='All',
                    placeholder="Select room type...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 997}
                )
            ])
        ])

    elif tab_id == "comp":
        # Comparative: Property Type, Area, Room Type, Registration Type, Developer, Time Horizon
        common_filters.extend([
            html.Div([
                html.Label("Area", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=area_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': a, 'value': a} for a in areas[:100]],
                    value='All',
                    placeholder="Select areas...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 999}
                )
            ]),
            html.Div([
                html.Label("Room Configuration", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=room_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in room_types],
                    value='All',
                    placeholder="Select room types...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 998}
                )
            ]),
            html.Div([
                html.Label("Registration Type", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=registration_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': r, 'value': r} for r in reg_types],
                    value='All',
                    placeholder="Select registration types...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 997}
                )
            ]),
            html.Div([
                html.Label("Developer", className="fw-bold mb-2"),
                dcc.Dropdown(
                    id=developer_id,
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': d, 'value': d} for d in developers[:50]],
                    value='All',
                    placeholder="Select developer...",
                    className="mb-3 expandable-dropdown",
                    style={'position': 'relative', 'zIndex': 996}
                )
            ]),
            html.Div([
                html.Label("Time Horizon", className="fw-bold mb-2"),
                dcc.RadioItems(
                    id=f"{tab_id}-time-horizon-filter",
                    options=[
                        {"label": "Short-term (1 year)", "value": "short_term_growth"},
                        {"label": "Medium-term (3 years)", "value": "medium_term_growth"},
                        {"label": "Long-term (5+ years)", "value": "long_term_cagr"}
                    ],
                    value="short_term_growth",
                    className="mb-3 vertical-radio",
                    style={'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}
                )
            ])
        ])
    
    # Create the sidebar
    sidebar = html.Div([
        html.H5("Filters", className="mb-3 text-center fw-bold"),
        html.Hr(style={'margin': '10px 0'}),
        *common_filters
    ], style={
        'position': 'sticky',
        'top': '20px',
        'height': 'fit-content',
        'maxHeight': '85vh',
        'overflowY': 'auto',
        'backgroundColor': '#f8f9fa',
        'padding': '20px 15px',
        'borderRadius': '12px',
        'border': '1px solid #dee2e6',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })
    
    return sidebar

def create_graph_container(id_base, fallback_message="No data available", min_height="500px"):
    """Create a consistent graph container with optimized sizing and spacing"""
    return html.Div([
        dcc.Loading(
            id=f"{id_base}-loading",
            type="circle",
            children=html.Div(
                id=f"{id_base}-graph", 
                children=create_fallback_content(fallback_message),
                style={
                    'width': '100%',
                    'minHeight': min_height,
                    'height': 'auto',
                    'paddingBottom': '40px'
                }
            )
        ),
        html.Div(
            id=f"{id_base}-insights", 
            className="mt-4",
            style={'width': '100%', 'padding': '15px 0', 'clear': 'both'}
        )
    ], style={
        'width': '100%',
        'height': 'auto',
        'marginBottom': '40px'
    })

def create_fallback_content(message="Data not available", height=400, show_data_quality=False):
    """Create fallback content for when data or visualizations are not available"""
    content = [
        html.H5(message, className="text-muted"),
        html.P("Please check filters or data availability"),
    ]
    
    if show_data_quality:
        content.append(html.Div(className="data-quality-badge mt-2"))
    
    return html.Div([
        html.Div(content, className="text-center")
    ], style={"height": f"{height}px", "display": "flex", "align-items": "center", "justify-content": "center"})

def check_asset_exists(asset_path):
    """Check if an asset file exists, handle both absolute and relative paths"""
    if os.path.isabs(asset_path):
        return os.path.exists(asset_path)
    
    # Check if we're in dashboard directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Try multiple possible locations
    possible_paths = [
        os.path.join(current_dir, asset_path),  # Current directory
        os.path.join(parent_dir, asset_path),   # Parent directory
        asset_path                              # Direct path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return True
    
    return False

def create_layout(df=None):
    """Create the dashboard layout with sidebar design"""
    # Handle empty dataframe
    if df is None or len(df) == 0:
        df = pd.DataFrame({
            'property_type_en': ['Apartment', 'Villa'],
            'area_name_en': ['Downtown Dubai', 'Dubai Marina', 'Palm Jumeirah'],
            'rooms_en': ['Studio', '1 B/R', '2 B/R', '3 B/R'],
            'reg_type_en': ['Existing Properties', 'Off-Plan Properties']
        })
        data_status = "Warning: Using sample data (no data provided)"
    else:
        data_status = f"Data loaded successfully: {len(df):,} records analyzed"
    
    # Create dashboard layout
    layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Dubai Real Estate Investment Dashboard", className="text-center pt-4 pb-2"),
                html.H5("Micro-Segment Analysis & Investment Opportunities", 
                        className="text-center text-muted pb-2"),
                html.Div(data_status, className="text-center text-muted small pb-3"),
            ], width=12)
        ]),
        
        # Main Navigation Tabs
        dbc.Tabs([
            # ===== Investment Analysis Tab =====
            dbc.Tab([
                dbc.Row([
                    # Sidebar with filters
                    dbc.Col([
                        create_sidebar_filters(df, "investment")
                    ], width=12, lg=3, className="mb-4 mb-lg-0"),
                    
                    # Main content area with sub-tabs
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Investment Analysis", className="mb-0")),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("investment-heatmap", "Investment heatmap not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Heatmap Analysis", tab_id="heatmap-analysis-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("investment-opportunities", "Investment opportunities not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Opportunities", tab_id="opportunities-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            dbc.Tabs([
                                                dbc.Tab([
                                                    create_graph_container("microsegment", "Microsegment data not available", "500px")
                                                ], label="Top Segments", tab_id="top-segments-tab"),
                                                
                                                dbc.Tab([
                                                    create_graph_container("emerging-segments", "Emerging segments not available", "500px")
                                                ], label="Emerging Segments", tab_id="emerging-segments-tab"),
                                            ], id="microsegment-tabs")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Microsegments", tab_id="microsegments-main-tab"),
                                ], id="investment-main-tabs")
                            ], style={'padding': '0', 'minHeight': '700px'})
                        ], style={'minHeight': '750px'})
                    ], width=12, lg=9)
                ], className="mt-3")
            ], label="Investment Analysis", tab_id="investment-analysis"),
            
            # ===== Comparative Analysis Tab =====
            dbc.Tab([
                dbc.Row([
                    # Sidebar with filters
                    dbc.Col([
                        create_sidebar_filters(df, "comp")
                    ], width=12, lg=3, className="mb-4 mb-lg-0"),
                    
                    # Main content area
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Comparative Analysis", className="mb-0")),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("segment-premium", "Premium analysis not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Segment Premium/Discount", tab_id="segment-premium-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("consistent-outperformers", "Outperformers analysis not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Consistent Outperformers", tab_id="consistent-outperformers-tab"),
                                ], id="comp-subtabs")
                            ], style={'padding': '0', 'minHeight': '700px'})
                        ], style={'minHeight': '750px'})
                    ], width=12, lg=9)
                ], className="mt-3")
            ], label="Comparative Analysis", tab_id="comparative-analysis"),
            
            # ===== Geographic Analysis Tab =====
            dbc.Tab([
                dbc.Row([
                    # Sidebar with filters
                    dbc.Col([
                        create_sidebar_filters(df, "geo")
                    ], width=12, lg=3, className="mb-4 mb-lg-0"),
                    
                    # Main content area
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Geographic Analysis", className="mb-0")),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("price-heatmap", "Price heatmap not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Price Map", tab_id="price-map-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("growth-heatmap", "Growth heatmap not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Growth Map", tab_id="growth-map-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("investment-hotspot", "Investment hotspots not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Investment Hotspots", tab_id="investment-hotspot-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("transaction-volume", "Transaction volume not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Transaction Volume", tab_id="transaction-volume-tab"),
                                ], id="geo-subtabs")
                            ], style={'padding': '0', 'minHeight': '700px'})
                        ], style={'minHeight': '750px'})
                    ], width=12, lg=9)
                ], className="mt-3")
            ], label="Geographic Analysis", tab_id="geographic-analysis"),
            
            # ===== Time Series Analysis Tab =====
            dbc.Tab([
                dbc.Row([
                    # Sidebar with filters
                    dbc.Col([
                        create_sidebar_filters(df, "ts")
                    ], width=12, lg=3, className="mb-4 mb-lg-0"),
                    
                    # Main content area
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Time Series Analysis", className="mb-0")),
                            dbc.CardBody([
                                dbc.Tabs([
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("price-trends", "Price trends not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Price Trends", tab_id="price-trends-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("market-cycles", "Market cycles not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Market Cycles", tab_id="market-cycles-tab"),
                                    
                                    dbc.Tab([
                                        html.Div([
                                            create_graph_container("price-forecast", "Price forecast not available", "600px")
                                        ], style={'width': '100%', 'padding': '20px 0'})
                                    ], label="Price Forecast", tab_id="price-forecast-tab"),
                                ], id="ts-subtabs")
                            ], style={'padding': '0', 'minHeight': '700px'})
                        ], style={'minHeight': '750px'})
                    ], width=12, lg=9)
                ], className="mt-3")
            ], label="Time Series Analysis", tab_id="time-series-analysis"),
            
            # ===== Project Analysis Tab =====
            dbc.Tab([
                dbc.Row([
                    # Sidebar with filters
                    dbc.Col([
                        create_sidebar_filters(df, "ltc")
                    ], width=12, lg=3, className="mb-4 mb-lg-0"),
                    
                    # Main content area
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Project Analysis", className="mb-0")),
                            dbc.CardBody([
                                dbc.Tabs([
                                    # ===== SUB-TAB 1: Individual Projects =====
                                    dbc.Tab([
                                        html.Div([
                                            # Main Chart with Absolute Legend
                                            html.Div([
                                                dcc.Loading(
                                                    id="project-appreciation-loading",
                                                    type="circle",
                                                    children=html.Div(
                                                        id="project-appreciation-chart",
                                                        style={
                                                            'width': '100%',
                                                            'minHeight': '500px',  # REDUCED from 600px
                                                            'paddingBottom': '5px',  # REDUCED from 20px
                                                            'position': 'relative'  # ADDED for absolute positioning
                                                        }
                                                    )
                                                ),
                                                # Absolute positioned legend
                                                html.Div([
                                                    html.H6("Legend", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '8px'}),
                                                    html.Div([
                                                        html.Div([html.Span("●", style={'color': 'darkblue', 'marginRight': '6px'}), "Normal"], style={'fontSize': '10px', 'marginBottom': '2px'}),
                                                        html.Div([html.Span("●", style={'color': 'lightblue', 'marginRight': '6px'}), "Early Launch < 9mo"], style={'fontSize': '10px', 'marginBottom': '2px'}),
                                                        html.Div([html.Span("●", style={'color': 'orange', 'marginRight': '6px'}), "Thin Data"], style={'fontSize': '10px', 'marginBottom': '2px'}),
                                                        #html.Div([html.Span("●", style={'color': 'red', 'marginRight': '6px'}), "Review"], style={'fontSize': '10px', 'marginBottom': '2px'}),
                                                        #html.Div([html.Span("●", style={'color': 'lightgray', 'marginRight': '6px'}), "Single"], style={'fontSize': '10px'})
                                                    ])
                                                ], style={
                                                    'position': 'absolute',
                                                    'top': '50px',
                                                    'right': '20px',
                                                    'backgroundColor': 'rgba(248, 249, 250, 0.9)',
                                                    'padding': '10px',
                                                    'borderRadius': '6px',
                                                    'border': '1px solid #dee2e6',
                                                    'width': '110px',
                                                    'zIndex': 1000
                                                })
                                            ], style={'width': '100%', 'textAlign': 'center', 'position': 'relative'}),
                                            
                                            html.Div([
                                                html.Div([
                                                    html.I(className="fas fa-info-circle", style={'marginRight': '8px', 'color': '#1976d2'}),
                                                    html.Strong("Interactive Feature: "),
                                                    "Click on any project bar above to view detailed room-wise breakdown below."
                                                ], style={
                                                    'backgroundColor': '#e3f2fd',
                                                    'padding': '12px',
                                                    'borderRadius': '6px',
                                                    'margin': '15px 0',
                                                    'border': '1px solid #1976d2',
                                                    'textAlign': 'center'
                                                })
                                            ]),
                                            
                                            # Drill-down Table (appears below chart on click)
                                            html.Div(
                                                id="project-drilldown-table",
                                                style={
                                                    'width': '100%',
                                                    'padding': '10px 0',
                                                    'clear': 'both'
                                                }
                                            ),
                                            
                                            # Insights Section
                                            html.Div(
                                                id="project-appreciation-insights",
                                                className="mt-4",
                                                style={
                                                    'width': '100%',
                                                    'padding': '20px 0',
                                                    'clear': 'both',
                                                    'borderTop': '1px solid #dee2e6'
                                                }
                                            )
                                        ], style={'width': '100%', 'padding': '20px'})
                                    ], label="Individual Projects", tab_id="individual-projects-tab"),
                                    
                                    # ===== SUB-TAB 2: Area Comparison =====
                                    dbc.Tab([
                                        html.Div([
                                            # Main Chart - Center  
                                            html.Div([
                                                dcc.Loading(
                                                    id="area-comparison-loading",
                                                    type="circle",
                                                    children=html.Div(
                                                        id="area-comparison-graph",
                                                        style={
                                                            'width': '100%',
                                                            'minHeight': '600px',
                                                            'paddingBottom': '20px'
                                                        }
                                                    )
                                                )
                                            ], style={'width': '100%', 'textAlign': 'center'}),
                                            
                                            # Insights Section
                                            html.Div(
                                                id="area-comparison-insights",
                                                className="mt-4",
                                                style={
                                                    'width': '100%',
                                                    'padding': '20px 0',
                                                    'clear': 'both',
                                                    'borderTop': '1px solid #dee2e6'
                                                }
                                            )
                                        ], style={'width': '100%', 'padding': '20px'})
                                    ], label="Area Comparison", tab_id="area-comparison-tab"),
                                    
                                    # ===== SUB-TAB 3: Developer Analysis =====
                                    dbc.Tab([
                                        html.Div([
                                            # Main Chart - Center
                                            html.Div([
                                                dcc.Loading(
                                                    id="developer-comparison-loading",
                                                    type="circle",
                                                    children=html.Div(
                                                        id="developer-comparison-graph",
                                                        style={
                                                            'width': '100%',
                                                            'minHeight': '600px',
                                                            'paddingBottom': '20px'
                                                        }
                                                    )
                                                )
                                            ], style={'width': '100%', 'textAlign': 'center'}),
                                            
                                            # Insights Section
                                            html.Div(
                                                id="developer-comparison-insights",
                                                className="mt-4",
                                                style={
                                                    'width': '100%',
                                                    'padding': '20px 0',
                                                    'clear': 'both',
                                                    'borderTop': '1px solid #dee2e6'
                                                }
                                            )
                                        ], style={'width': '100%', 'padding': '20px'})
                                    ], label="Developer Analysis", tab_id="developer-analysis-tab"),
                                    
                                ], id="project-main-tabs")
                            ], style={'padding': '0', 'minHeight': '700px'})
                        ], style={'minHeight': '750px'})
                    ], width=12, lg=9)
                ], className="mt-3")
            ], label="Project Analysis", tab_id="launch-completion-tab"),
            
            # ===== Investment Methodology Tab =====
            dbc.Tab([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H5("Investment Score Methodology", className="mb-0")),
                            dbc.CardBody([
                                html.Div([
                                    html.Img(
                                        src="assets/investment_score_explanation.png",
                                        width="100%",
                                        style={"max-height": "800px", "object-fit": "contain"}
                                    ) if check_asset_exists("assets/investment_score_explanation.png") else
                                    html.Div([
                                        dcc.Markdown("""
                                        # Investment Score Methodology
                                        
                                        The investment score evaluates properties on a scale of 0-100 based on multiple weighted factors:
                                        
                                        ## Core Factors (75%)
                                        - **Price Growth (40%)**: Historical price appreciation rates
                                          - Recent 1-year growth (20%)
                                          - 3-year growth trend (15%)
                                          - 5-year CAGR (5%)
                                        
                                        - **Liquidity (20%)**: Market activity metrics
                                          - Transaction volume (10%)
                                          - Time on market (5%)
                                          - Consistency of transactions (5%)
                                        
                                        - **Growth Acceleration (15%)**: Change in growth rate
                                          - Recent growth acceleration (10%)
                                          - Seasonal pattern strength (5%)
                                        
                                        ## Supplementary Factors (25%)
                                        - **Supply-Demand Balance (10%)**: Ratio of demand to new supply
                                        - **Price Volatility (5%)**: Stability of prices over time
                                        - **Rental Yield (5%)**: Current rental return potential
                                        - **Developer Premium (5%)**: Brand value premium
                                        
                                        ## Investment Categories
                                        - **Exceptional (80-100)**: Outstanding performance across multiple factors
                                        - **Strong (65-79)**: Robust growth and strong fundamentals
                                        - **Good (50-64)**: Positive outlook with some strong indicators
                                        - **Average (35-49)**: Moderate performance, may have specific strengths
                                        - **Below Average (<35)**: Underperforming relative to market
                                        """, className="methodology-content")
                                    ], style={'padding': '30px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
                                ], style={'marginBottom': '30px'}),
                                
                                html.Div([
                                    html.H5("How to Use This Dashboard", className="mt-4 mb-3"),
                                    html.Div([
                                        html.P([
                                            "This dashboard provides a comprehensive analysis of Dubai real estate investments through multiple specialized modules:"
                                        ], className="mb-3"),
                                        html.Div([
                                            html.Div([
                                                html.Strong("💼 Investment Analysis"), " - Performance metrics, emerging opportunities, and top investment segments"
                                            ], className="methodology-item"),
                                            html.Div([
                                                html.Strong("📊 Comparative Analysis"), " - Benchmarks segments against market averages and identifies consistent outperformers"
                                            ], className="methodology-item"),
                                            html.Div([
                                                html.Strong("🗺️ Geographic Analysis"), " - Area-based visualizations, price maps, and location-specific insights"
                                            ], className="methodology-item"),
                                            html.Div([
                                                html.Strong("📈 Time Series Analysis"), " - Price trends, market cycles, and forecasting models"
                                            ], className="methodology-item"),
                                            html.Div([
                                                html.Strong("🏗️ Project Analysis"), " - Actual project returns from first to last transaction"
                                            ], className="methodology-item")
                                        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
                                    ])
                                ], style={'marginTop': '30px'})
                            ], style={'padding': '30px'})
                        ], style={'minHeight': '600px'})
                    ], width=12)
                ], className="mt-4 mb-5")
            ], label="Investment Methodology", tab_id="investment-methodology"),
            
        ], id="main-tabs", active_tab="investment-analysis")
    ], fluid=True)
    
    return layout