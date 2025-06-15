# This file contains the callbacks for the dashboard interactivity
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import os

def register_callbacks(app, df, geo_df, launch_completion_df, micro_segment_df=None, comparative_df=None, time_series_df=None):
    """
    Register all callbacks for dashboard interactivity
    
    Args:
        app: Dash app instance
        df: Main dataframe
        geo_df: Geographic dataframe with area boundaries
        launch_completion_df: Launch to completion dataframe
        micro_segment_df: Micro-segmentation analysis dataframe
        comparative_df: Comparative analysis dataframe
        time_series_df: Pre-aggregated time series dataframe
    """

    # Print diagnostic information about available datasets
    print("\n===== CALLBACK REGISTRATION - DATA DIAGNOSTICS =====")
    print(f"Main dataframe: {len(df) if df is not None else 'None'} rows")
    print(f"Geographic dataframe: {'Available' if geo_df is not None else 'None'}")
    print(f"Launch completion dataframe: {len(launch_completion_df) if launch_completion_df is not None else 'None'} rows")
    print(f"Micro segment dataframe: {len(micro_segment_df) if micro_segment_df is not None else 'None'} rows")
    print(f"Comparative dataframe: {len(comparative_df) if comparative_df is not None else 'None'} rows")
    
             
    # Special diagnostic for time series data
    if time_series_df is not None:
        ts_years = sorted(time_series_df['year'].unique().tolist()) if 'year' in time_series_df.columns else []
        print(f"Time series dataframe: {len(time_series_df)} rows covering {len(ts_years)} years: {ts_years}")
    else:
        print("Time series dataframe: None")
    print("===================================================\n")
    
    # CRITICAL FIX: Helper function to access time_series_df from inside callbacks
    def get_time_series_data():
        """Helper function to access time_series_df from inside callbacks"""
        return time_series_df
    
    #--------------------------
    # Investment Analysis Tab Callbacks
    #--------------------------
    
    #--------------------------
    # Investment Heatmap Callback
    #--------------------------
    @app.callback(
        [Output("investment-heatmap-graph", "children"),
        Output("investment-heatmap-insights", "children")],
        [Input("investment-property-type-filter", "value"),
        Input("investment-area-filter", "value"),
        Input("investment-room-type-filter", "value"),
        Input("investment-registration-type-filter", "value"),
        Input("investment-time-horizon-filter", "value")]
    )
    def update_investment_heatmap(property_type, area, room_type, registration_type, time_horizon):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Log debug info
            print(f"Investment heatmap callback triggered with filters: {filters}")
            print(f"Time horizon: {time_horizon}")
            
            print(f"DEBUG TIME FILTER: Selected time horizon: {time_horizon}")

            # Apply progressive filtering - start with all filters
            filtered_df = progressive_filtering(df, filters)
            
            # Generate heatmap figure using improved function
            try:
                from scripts.investment_analysis import create_investment_heatmap, select_best_growth_column
                fig = create_investment_heatmap(filtered_df, time_horizon=time_horizon)

                # Get data quality info for the insights
                growth_column, coverage_pct, data_quality = select_best_growth_column(
                    filtered_df, 
                    requested_horizon=time_horizon
                )
                
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'growth_column': growth_column
                }
            
            except Exception as e:
                print(f"Error creating investment heatmap: {e}")
                fig = create_default_figure("Investment Heatmap", f"Error generating heatmap: {str(e)}")
                metadata = {}

            # Generate insights with the new insights module
            try:
                from scripts.insights import create_insights_component
                insights_component = create_insights_component(
                    filtered_df,
                    'investment',
                    'heatmap',
                    {
                        'property_type_en': property_type,
                        'area_name_en': area,
                        'rooms_en': room_type,
                        'reg_type_en': registration_type,
                        'time_horizon': time_horizon
                    },
                    metadata
                )
            except Exception as e:
                print(f"Error creating investment insights: {e}")
                insights_component = html.Div("Investment insights not available")
            
            return dcc.Graph(figure=fig), insights_component
        except Exception as e:
            print(f"Error in investment heatmap callback: {e}")
            return dcc.Graph(figure=create_default_figure("Investment Heatmap", str(e))), html.Div("Error generating insights")
        
    #--------------------------
    # Investment Opportunities Callback
    #--------------------------
    @app.callback(
    [Output("investment-opportunities-graph", "children"),
    Output("investment-opportunities-insights", "children")],
    [Input("investment-property-type-filter", "value"),
    Input("investment-area-filter", "value"),
    Input("investment-room-type-filter", "value"),
    Input("investment-registration-type-filter", "value"),
    Input("investment-time-horizon-filter", "value")]
    )
    def update_investment_opportunities(property_type, area, room_type, registration_type, time_horizon):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Log debug info
            print(f"Investment opportunities callback triggered with filters: {filters}")
            print(f"Time horizon: {time_horizon}")
            print(f"DEBUG TIME FILTER: Selected time horizon: {time_horizon}")
            # Apply progressive filtering - start with all filters
            filtered_df = progressive_filtering(df, filters)
            
            # Generate opportunities scatter plot using improved function
            try:
                from scripts.investment_analysis import create_opportunity_scatter, select_best_growth_column
                fig = create_opportunity_scatter(filtered_df, time_horizon=time_horizon)
                
                # Get data quality info for the insights
                growth_column, coverage_pct, data_quality = select_best_growth_column(
                    filtered_df, 
                    requested_horizon=time_horizon
                )
                
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'growth_column': growth_column
                }
                
            except Exception as e:
                print(f"Error creating opportunity scatter: {e}")
                fig = create_default_figure("Investment Opportunities", f"Error generating scatter plot: {str(e)}")
                metadata = {}
            
            # Generate insights with data quality information
            try:
                # Try to import insights module
                from scripts.insights import create_insights_component
                insights_component = create_insights_component(
                    filtered_df, 
                    'investment',
                    'opportunities',
                    {
                        'property_type': property_type,
                        'area': area,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'time_horizon': time_horizon
                    },
                    metadata
                )
            except Exception as e:
                print(f"Error creating opportunity insights: {e}")
                insights_component = html.Div("Investment opportunity insights not available")
            
            return dcc.Graph(figure=fig), insights_component
        except Exception as e:
            print(f"Error in investment opportunities callback: {e}")
            error_figure = create_default_figure("Investment Opportunities", f"Error: {str(e)}")
            return dcc.Graph(figure=error_figure), html.Div("Error generating insights")
        
    #--------------------------
    # Microsegment Analysis Callback
    #--------------------------
    @app.callback(
    Output("microsegment-graph", "children"),
    [Input("investment-property-type-filter", "value"),
    Input("investment-area-filter", "value"),
    Input("investment-room-type-filter", "value"),
    Input("investment-registration-type-filter", "value"),
    Input("investment-time-horizon-filter", "value"),
    Input("microsegment-tabs", "active_tab")]
    )
    def update_microsegment_analysis(property_type, area, room_type, registration_type, time_horizon, active_tab):
        # Default to top-segments-tab if None
        active_tab = active_tab or "top-segments-tab"
        
        # Only process this callback if the top segments tab is active
        if active_tab != "top-segments-tab":
            return html.Div()  # Return empty div for other tabs
        
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Log debug info
            print(f"Microsegment callback triggered with filters: {filters}")
            print(f"Time horizon: {time_horizon}")
            print(f"Active tab: {active_tab}")
            
            # Apply progressive filtering - start with all filters
            filtered_df = progressive_filtering(df, filters)
            
            # Generate microsegment table
            try:
                # Check if pre-calculated data is available
                if micro_segment_df is not None:
                    print("Using pre-calculated micro_segment_df")
                    filtered_micro_df = apply_filters(micro_segment_df, filters)
                    
                    if len(filtered_micro_df) > 0:
                        from scripts.investment_analysis import create_microsegment_table, perform_microsegment_analysis
                        from scripts.insights import create_insights_component
                        
                        # Get metadata even with pre-calculated data
                        _, metadata = perform_microsegment_analysis(filtered_df, filters=filters, growth_column=time_horizon)
                        
                        # Create the table component
                        table = create_microsegment_table(filtered_micro_df, metadata)
                        
                        # Add insights component
                        insights = create_insights_component(
                            filtered_df,
                            'investment',
                            'microsegment',
                            {
                                'property_type': property_type,
                                'area': area,
                                'room_type': room_type,
                                'registration_type': registration_type,
                                'time_horizon': time_horizon
                            },
                            metadata
                        )
                        
                        # Combine table and insights
                        return html.Div([
                            table,
                            html.Div(className="mt-4"),
                            insights
                        ])
                    
                    # If no data after filtering, do live analysis
                    print("No data after filtering pre-calculated data")
                
                # If no pre-calculated data or after filtering it's empty, do live analysis
                print("Performing live microsegment analysis")
                from scripts.investment_analysis import perform_microsegment_analysis, create_microsegment_table
                from scripts.insights import create_insights_component
                
                micro_segments, metadata = perform_microsegment_analysis(filtered_df, filters=filters, growth_column=time_horizon)
                table = create_microsegment_table(micro_segments, metadata)
                
                # Add insights component
                insights = create_insights_component(
                    filtered_df,
                    'investment',
                    'microsegment',
                    {
                        'property_type': property_type,
                        'area': area,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'time_horizon': time_horizon
                    },
                    metadata
                )
                
                # Combine table and insights
                return html.Div([
                    table,
                    html.Div(className="mt-4"),
                    insights
                ])
                    
            except Exception as e:
                print(f"Error creating microsegment table: {e}")
                return html.Div([
                    html.H5("Error in Microsegment Analysis"),
                    html.P(f"Details: {str(e)}"),
                    html.P("Please try adjusting your filters or check the data quality.")
                ])
        except Exception as e:
            print(f"Error in microsegment analysis callback: {e}")
            return html.Div([
                html.H5("Error in Analysis"),
                html.P(f"Details: {str(e)}"),
                html.P("Please try adjusting your filters.")
            ])

    # Separate callback for emerging segments tab
    @app.callback(
    Output("emerging-segments-graph", "children"),
    [Input("investment-property-type-filter", "value"),
    Input("investment-area-filter", "value"),
    Input("investment-room-type-filter", "value"),
    Input("investment-registration-type-filter", "value"),
    Input("microsegment-tabs", "active_tab")]
    )
    def update_emerging_segments(property_type, area, room_type, registration_type, active_tab):
        # Only process if emerging segments tab is active
        if active_tab != "emerging-segments-tab":
            # Return empty div for other tabs
            return html.Div(style={'display': 'none'})
        
        print(f"Emerging segments callback triggered with filters: {property_type}, {area}, {room_type}, {registration_type}")
        
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Apply progressive filtering - start with all filters
            filtered_df = progressive_filtering(df, filters)
            
            # Try using pre-calculated micro_segment_df if available
            if micro_segment_df is not None and 'is_emerging' in micro_segment_df.columns:
                print("Using pre-calculated emerging segments")
                filtered_micro_df = apply_filters(micro_segment_df, filters)
                emerging_segments = filtered_micro_df[filtered_micro_df['is_emerging'] == True]
                
                if len(emerging_segments) > 0:
                    print(f"Found {len(emerging_segments)} emerging segments")
                    from scripts.investment_analysis import perform_microsegment_analysis
                    from scripts.insights import create_insights_component
                    
                    # Get metadata even with pre-calculated data
                    _, metadata = perform_microsegment_analysis(filtered_df)
                    
                    # Create table
                    table = create_emerging_segments_table(emerging_segments, metadata)
                    
                    # Generate insights
                    insights = create_insights_component(
                        filtered_df,
                        'investment',
                        'emerging',
                        {
                            'property_type': property_type,
                            'area': area,
                            'room_type': room_type,
                            'registration_type': registration_type
                        },
                        metadata
                    )
                    
                    # Combine table and insights
                    return html.Div([
                        table,
                        html.Div(className="mt-4"),
                        insights
                    ])
                else:
                    return html.Div([
                        html.H5("No Emerging Segments Found"),
                        html.P("No segments with accelerating growth found for the selected filters.")
                    ])
            
            # Fallback to live analysis
            print("Performing live emerging segments analysis")
            try:
                from scripts.segmentation_analysis import perform_micro_segmentation_analysis
                from scripts.investment_analysis import perform_microsegment_analysis
                from scripts.insights import create_insights_component
                
                # Get metadata for data quality information
                _, metadata = perform_microsegment_analysis(filtered_df)
                
                micro_segments = perform_micro_segmentation_analysis(filtered_df)
                
                if 'is_emerging' in micro_segments.columns:
                    emerging_segments = micro_segments[micro_segments['is_emerging'] == True]
                    if len(emerging_segments) > 0:
                        # Create table
                        table = create_emerging_segments_table(emerging_segments, metadata)
                        
                        # Generate insights
                        insights = create_insights_component(
                            filtered_df,
                            'investment',
                            'emerging',
                            {
                                'property_type': property_type,
                                'area': area,
                                'room_type': room_type,
                                'registration_type': registration_type
                            },
                            metadata
                        )
                        
                        # Combine table and insights
                        return html.Div([
                            table,
                            html.Div(className="mt-4"),
                            insights
                        ])
                    else:
                        return html.Div([
                            html.H5("No Emerging Segments Found"),
                            html.P("No segments with accelerating growth found for the selected filters.")
                        ])
                else:
                    return html.Div([
                        html.H5("Emerging Segments Analysis Unavailable"),
                        html.P("Unable to identify emerging segments with the available data.")
                    ])
            except Exception as e:
                print(f"Error in emerging segments analysis: {e}")
                return html.Div([
                    html.H5("Emerging Segments Analysis Unavailable"),
                    html.P(f"Error: {str(e)}")
                ])
        except Exception as e:
            print(f"Error in emerging segments callback: {e}")
            return html.Div([
                html.H5("Error in Analysis"),
                html.P(f"Details: {str(e)}"),
                html.P("Please try adjusting your filters.")
            ])
        
    #--------------------------
    # Time Series Analysis Tab Callbacks
    #--------------------------
    
    @app.callback(
        [Output("price-trends-graph", "children"),
        Output("price-trends-insights", "children")],
        [Input("ts-property-type-filter", "value"),
        Input("ts-area-filter", "value"),
        Input("ts-room-type-filter", "value"),
        Input("ts-registration-type-filter", "value")]
    )
    def update_price_trends(property_type, area, room_type, registration_type):
        """
        Update the price trends visualization based on selected filters
        
        Args:
            property_type (str): Selected property type
            area (str): Selected area
            room_type (str): Selected room type
            registration_type (str): Selected registration type
            
        Returns:
            tuple: (visualization component, insights component)
        """
        try:
            print(f"Updating price trends with filters: property_type={property_type}, area={area}, "
                f"room_type={room_type}, registration_type={registration_type}")
            
            # Import time series analysis functions
            from scripts.time_series_analysis import create_price_trends_chart, generate_time_series_insights
            
            # Create the price trends chart
            fig = create_price_trends_chart(df, property_type, area, room_type, registration_type)
            
            # Generate insights
            try:
                from scripts.insights import create_insights_component
                
                # Get insights data for metadata
                insights_data = generate_time_series_insights(
                    df, 'price_trends', property_type, area, room_type, registration_type
                )

                # Create filter dictionary for insights
                filters = {
                    'property_type': property_type,
                    'area': area,
                    'room_type': room_type,
                    'registration_type': registration_type
                }
                
                # Generate insights
                insights = create_insights_component(
                    df, 
                    'time_series',
                    'price_trends',  # Specific visualization type
                    filters,
                    {
                    'data_quality': insights_data.get('data_quality', 'unknown'),
                    'coverage_pct': insights_data.get('coverage_pct', 0)
                    }
                )
            except (ImportError, AttributeError) as e:
                print(f"Error creating insights from insights.py: {e}")
                
                # Fallback to direct insights generation
                insights_data = generate_time_series_insights(
                    df, 'price_trends', property_type, area, room_type, registration_type
                )
                
                # Create a basic insights component
                insights = html.Div([
                    html.H5("Price Trends Insights"),
                    html.Div([html.P(text) for text in insights_data['text']])
                ])
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in price trends callback: {e}")
            # Create a default figure with the error message
            error_fig = create_default_figure("Price Trends", f"Error: {str(e)}")
            error_insights = html.Div(["Error generating insights: ", str(e)])
            return dcc.Graph(figure=error_fig), error_insights


    @app.callback(
        [Output("market-cycles-graph", "children"),
        Output("market-cycles-insights", "children")],
        [Input("ts-property-type-filter", "value"),
        Input("ts-area-filter", "value"),
        Input("ts-room-type-filter", "value"),
        Input("ts-registration-type-filter", "value")]
    )
    def update_market_cycles(property_type, area, room_type, registration_type):
        """Update the market cycles visualization based on selected filters"""
        try:
            print(f"Updating market cycles with filters: property_type={property_type}, area={area}, "
                f"room_type={room_type}, registration_type={registration_type}")
            
            # Import time series analysis functions
            from scripts.time_series_analysis import detect_market_cycles, generate_time_series_insights
            
            # Create the market cycles chart
            fig, breakpoints = detect_market_cycles(df, property_type, area, room_type, registration_type)
            
            # Generate insights with new insights module
            try:
                from scripts.insights import create_insights_component
                
                # Get insights data for metadata
                insights_data = generate_time_series_insights(
                    df, 'market_cycles', property_type, area, room_type, registration_type
                )
                
                # Create filter dictionary for insights
                filters = {
                    'property_type': property_type,
                    'area': area,
                    'room_type': room_type,
                    'registration_type': registration_type
                }
                
                # Generate insights using the proper insights component
                insights = create_insights_component(
                    df, 
                    'time_series',
                    'market_cycles',  # Specific visualization type
                    filters,
                    {
                        'data_quality': insights_data.get('data_quality', 'unknown'),
                        'coverage_pct': insights_data.get('coverage_pct', 0)
                    }
                )
                
                return dcc.Graph(figure=fig), insights
                
            except Exception as e:
                print(f"Error creating market cycles insights: {e}")
                import traceback
                traceback.print_exc()
                
                # Create a fallback insights component
                insights = html.Div([
                    html.H5("Market Cycles Insights"),
                    html.P(f"We encountered an issue generating detailed insights."),
                    html.Div([
                        html.P(text) for text in insights_data.get('text', [
                            "Current Market Phase: " + insights_data.get('metrics', {}).get('current_phase', 'Unknown'),
                            "Market cycles typically include expansion, peak, contraction, and recovery phases.",
                            "Understanding the current market phase helps with investment timing decisions."
                        ])
                    ])
                ])
                
                return dcc.Graph(figure=fig), insights
            
        except Exception as e:
            print(f"Error in market cycles callback: {e}")
            import traceback
            traceback.print_exc()
            
            error_fig = create_default_figure("Market Cycles", f"Error: {str(e)}")
            error_insights = html.Div([
                html.H5("Market Cycles Analysis Error"),
                html.P(f"An error occurred: {str(e)}"),
                html.P("Please try adjusting your filters or contact support if the issue persists.")
            ])
            
            return dcc.Graph(figure=error_fig), error_insights
        
    @app.callback(
        [Output("price-forecast-graph", "children"),
        Output("price-forecast-insights", "children")],
        [Input("ts-property-type-filter", "value"),
        Input("ts-area-filter", "value"),
        Input("ts-room-type-filter", "value"),
        Input("ts-registration-type-filter", "value")]
    )
    def update_price_forecast(property_type, area, room_type, registration_type):
        """
        Update the price forecast visualization based on selected filters
        
        Args:
            property_type (str): Selected property type
            area (str): Selected area
            room_type (str): Selected room type
            registration_type (str): Selected registration type
            
        Returns:
            tuple: (visualization component, insights component)
        """
        try:
            print(f"Updating price forecast with filters: property_type={property_type}, area={area}, "
                f"room_type={room_type}, registration_type={registration_type}")
            
            # Import time series analysis functions
            from scripts.time_series_analysis import create_price_forecast, generate_time_series_insights
            
            # Create the price forecast chart
            fig = create_price_forecast(df, property_type, area, room_type, registration_type)
            
            # Generate insights
            try:
                from scripts.insights import create_insights_component

                # Get insights data for metadata
                insights_data = generate_time_series_insights(
                    df, 'price_forecast', property_type, area, room_type, registration_type
                )
                
                # Create filter dictionary for insights
                filters = {
                    'property_type': property_type,
                    'area': area,
                    'room_type': room_type,
                    'registration_type': registration_type
                }
                
                # Generate insights
                insights = create_insights_component(
                    df, 
                    'time_series',
                    'price_forecast',  # Specific visualization type
                    filters,
                    {
                    'data_quality': insights_data.get('data_quality', 'unknown'),
                    'coverage_pct': insights_data.get('coverage_pct', 0)
                    }
                )
            except (ImportError, AttributeError) as e:
                print(f"Error creating insights from insights.py: {e}")
                
                # Fallback to direct insights generation
                insights_data = generate_time_series_insights(
                    df, 'price_forecast', property_type, area, room_type, registration_type
                )
                
                # Create a basic insights component
                insights = html.Div([
                    html.H5("Price Forecast Insights"),
                    html.Div([html.P(text) for text in insights_data['text']])
                ])
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in price forecast callback: {e}")
            # Create a default figure with the error message
            error_fig = create_default_figure("Price Forecast", f"Error: {str(e)}")
            error_insights = html.Div(["Error generating insights: ", str(e)])
            return dcc.Graph(figure=error_fig), error_insights


    # Helper function for default figures (if not already defined in callbacks.py)
    def create_default_figure(title="No Data", message="No data available for the selected filters"):
        """Create a default figure with a message when data is not available"""
        fig = go.Figure()
        
        fig.update_layout(
            title=title,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": message,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 18
                    }
                }
            ]
        )
        
        return fig

    #--------------------------
    # Comparative Analysis Tab Callbacks
    #--------------------------
    
    @app.callback(
    [Output("segment-premium-graph", "children"),
     Output("segment-premium-insights", "children")],
    [Input("comp-property-type-filter", "value"),
     Input("comp-area-filter", "value"),
     Input("comp-room-type-filter", "value"),
     Input("comp-registration-type-filter", "value"),
     Input("comp-developer-filter", "value")]
    )
    def update_segment_premium(property_type, area, room_type, registration_type, developer):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        if developer != 'All' and developer is not None:
            filters['developer_name'] = developer
        
        try:
            # Use pre-calculated comparative data if available
            from scripts.insights import create_insights_component
            
            
            if comparative_df is not None:
                # Apply filters to comparative dataframe
                filtered_comp_df = apply_filters(comparative_df, filters)
                
                # Check for premium columns
                premium_columns = [col for col in filtered_comp_df.columns if col.endswith('_premium')]
                
                if len(premium_columns) > 0 and len(filtered_comp_df) > 0:
                    # Create premium visualization
                    fig = create_premium_discount_chart(filtered_comp_df, premium_columns)
                else:
                    # Perform live analysis
                    filtered_df = apply_filters(df, filters)
                    try:
                        from scripts.comparative_analysis import calculate_segment_premium
                        # Fix: Reduce complexity since the original function is failing
                        premium_df = calculate_simple_premium(filtered_df)
                        fig = create_premium_discount_chart(premium_df, [col for col in premium_df.columns if col.endswith('_premium')])
                    except Exception as e:
                        print(f"Error calculating segment premiums: {e}")
                        fig = create_default_figure("Segment Premium/Discount", "Error generating premium analysis")
            else:
                # Perform live analysis using the comparative_analysis module
                filtered_df = apply_filters(df, filters)
                try:
                    from scripts.comparative_analysis import calculate_segment_premium
                    # Fix: Use a simplified version
                    premium_df = calculate_simple_premium(filtered_df)
                    fig = create_premium_discount_chart(premium_df, [col for col in premium_df.columns if col.endswith('_premium')])
                except Exception as e:
                    print(f"Error calculating segment premiums: {e}")
                    fig = create_default_figure("Segment Premium/Discount", "Error generating premium analysis")
            
            # Generate insights
            filtered_df = apply_filters(df, filters)
            insights = create_insights_component(
                filtered_df, 
                'comparative',  # Changed from 'investment' to 'comparative'
                'segment_premium',  # Specific visualization type
                {
                    'property_type': property_type,
                    'area': area,
                    'room_type': room_type,
                    'registration_type': registration_type,
                    'developer': developer
                },
                {
                    'data_quality': 'medium',  # Default for comparative analysis
                    'coverage_pct': 60  # Default coverage for premium data
                }
            )
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in segment premium callback: {e}")
            return dcc.Graph(figure=create_default_figure("Segment Premium/Discount", str(e))), html.Div("Error generating insights")
        
    # Comparative Analysis Tab Callbacks - Updated with Dynamic Insights

    @app.callback(
        [Output("relative-performance-graph", "children"),
        Output("relative-performance-insights", "children")],
        [Input("comp-property-type-filter", "value"),
        Input("comp-area-filter", "value"),
        Input("comp-room-type-filter", "value"),
        Input("comp-registration-type-filter", "value"),
        Input("comp-developer-filter", "value"),
        Input("comp-time-horizon-filter", "value")]
    )
    def update_relative_performance(property_type, area, room_type, registration_type, developer, time_horizon):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        if developer != 'All' and developer is not None:
            filters['developer_name'] = developer
        
        try:
            # Use pre-calculated comparative data if available
            if comparative_df is not None:
                # Apply filters to comparative dataframe
                filtered_comp_df = apply_filters(comparative_df, filters)
                
                # Check for relative performance columns
                rel_perf_columns = [col for col in filtered_comp_df.columns if col.startswith('rel_')]
                
                if len(rel_perf_columns) > 0 and len(filtered_comp_df) > 0:
                    # Create relative performance visualization
                    rel_column = f'rel_{time_horizon}' if f'rel_{time_horizon}' in rel_perf_columns else rel_perf_columns[0]
                    fig = create_relative_performance_chart(filtered_comp_df, rel_column)
                    
                    # Calculate data quality metrics
                    data_quality = "medium"  # Default for comparative analysis
                    coverage_pct = filtered_comp_df[rel_column].notna().mean() * 100 if rel_column in filtered_comp_df.columns else 50
                    data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                else:
                    # Perform live analysis
                    filtered_df = apply_filters(df, filters)
                    try:
                        from scripts.comparative_analysis import create_performance_benchmarks
                        benchmark_df = create_performance_benchmarks(filtered_df)
                        rel_column = f'rel_{time_horizon}' if f'rel_{time_horizon}' in benchmark_df.columns else benchmark_df.columns[0]
                        fig = create_relative_performance_chart(benchmark_df, rel_column)
                        
                        # Calculate data quality metrics
                        data_quality = "medium"  # Default for freshly calculated data
                        coverage_pct = benchmark_df[rel_column].notna().mean() * 100 if rel_column in benchmark_df.columns else 50
                        data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                    except (ImportError, AttributeError) as e:
                        print(f"Error creating performance benchmarks: {e}")
                        fig = create_default_figure("Relative Performance", "Error generating relative performance analysis")
                        data_quality = "low"
                        coverage_pct = 0
            else:
                # Perform live analysis using the comparative_analysis module
                filtered_df = apply_filters(df, filters)
                try:
                    from scripts.comparative_analysis import create_performance_benchmarks
                    benchmark_df = create_performance_benchmarks(filtered_df)
                    rel_column = f'rel_{time_horizon}' if f'rel_{time_horizon}' in benchmark_df.columns else [col for col in benchmark_df.columns if col.startswith('rel_')][0]
                    fig = create_relative_performance_chart(benchmark_df, rel_column)
                    
                    # Calculate data quality metrics
                    data_quality = "medium"  # Default for live analysis
                    coverage_pct = benchmark_df[rel_column].notna().mean() * 100 if rel_column in benchmark_df.columns else 50
                    data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                except (ImportError, AttributeError) as e:
                    print(f"Error creating performance benchmarks: {e}")
                    fig = create_default_figure("Relative Performance", "Error generating relative performance analysis")
                    data_quality = "low"
                    coverage_pct = 0
            
            # Generate insights using new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'time_horizon': time_horizon
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'comparative',
                    'relative_performance',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'area': area,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'developer': developer,
                        'time_horizon': time_horizon
                    },
                    metadata
                )
            except (ImportError, AttributeError) as e:
                print(f"Error creating relative performance insights: {e}")
                insights = html.Div("Relative performance insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in relative performance callback: {e}")
            return dcc.Graph(figure=create_default_figure("Relative Performance", str(e))), html.Div("Error generating insights")


    @app.callback(
        [Output("consistent-outperformers-graph", "children"),
        Output("consistent-outperformers-insights", "children")],
        [Input("comp-property-type-filter", "value"),
        Input("comp-area-filter", "value"),
        Input("comp-room-type-filter", "value"),
        Input("comp-registration-type-filter", "value"),
        Input("comp-developer-filter", "value")]
    )
    def update_consistent_outperformers(property_type, area, room_type, registration_type, developer):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        if developer != 'All' and developer is not None:
            filters['developer_name'] = developer
        
        try:
            # Use pre-calculated comparative data if available
            outperformers = None
            data_quality = "medium"  # Default for comparative analysis
            coverage_pct = 50  # Default coverage
            
            if comparative_df is not None and 'is_consistent_outperformer' in comparative_df.columns:
                # Apply filters to comparative dataframe
                filtered_comp_df = apply_filters(comparative_df, filters)
                
                if len(filtered_comp_df) > 0:
                    # Create consistent outperformers visualization
                    outperformers = filtered_comp_df[filtered_comp_df['is_consistent_outperformer'] == True]
                    if len(outperformers) > 0:
                        table = create_outperformers_table(outperformers)
                        # Calculate data quality metrics
                        coverage_pct = filtered_comp_df['outperformance_ratio'].notna().mean() * 100 if 'outperformance_ratio' in filtered_comp_df.columns else 50
                        data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                    else:
                        table = html.Div([
                            html.H5("No Consistent Outperformers Found"),
                            html.P("No segments consistently outperforming the market for the selected filters.")
                        ])
                else:
                    # Fall back to live analysis
                    filtered_df = apply_filters(df, filters)
                    try:
                        from scripts.comparative_analysis import perform_comparative_analysis
                        comp_df = perform_comparative_analysis(filtered_df)
                        outperformers = comp_df[comp_df['is_consistent_outperformer'] == True]
                        if len(outperformers) > 0:
                            table = create_outperformers_table(outperformers)
                            # Calculate data quality metrics
                            coverage_pct = comp_df['outperformance_ratio'].notna().mean() * 100 if 'outperformance_ratio' in comp_df.columns else 50
                            data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                        else:
                            table = html.Div([
                                html.H5("No Consistent Outperformers Found"),
                                html.P("No segments consistently outperforming the market for the selected filters.")
                            ])
                    except (ImportError, AttributeError) as e:
                        print(f"Error performing comparative analysis: {e}")
                        table = html.Div("Consistent outperformers analysis not available")
                        data_quality = "low"
                        coverage_pct = 0
            else:
                # Perform live analysis using the comparative_analysis module
                filtered_df = apply_filters(df, filters)
                try:
                    from scripts.comparative_analysis import identify_consistent_outperformers, create_performance_benchmarks
                    benchmark_df = create_performance_benchmarks(filtered_df)
                    outperformer_df = identify_consistent_outperformers(benchmark_df)
                    outperformers = outperformer_df[outperformer_df['is_consistent_outperformer'] == True]
                    if len(outperformers) > 0:
                        table = create_outperformers_table(outperformers)
                        # Calculate data quality metrics
                        coverage_pct = outperformer_df['outperformance_ratio'].notna().mean() * 100 if 'outperformance_ratio' in outperformer_df.columns else 50
                        data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                    else:
                        table = html.Div([
                            html.H5("No Consistent Outperformers Found"),
                            html.P("No segments consistently outperforming the market for the selected filters.")
                        ])
                except (ImportError, AttributeError) as e:
                    print(f"Error identifying consistent outperformers: {e}")
                    table = html.Div("Consistent outperformers analysis not available")
                    data_quality = "low"
                    coverage_pct = 0
            
            # Generate insights using new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'outperformer_count': len(outperformers) if outperformers is not None else 0
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'comparative',
                    'consistent_outperformers',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'area': area,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'developer': developer
                    },
                    metadata
                )
            except (ImportError, AttributeError) as e:
                print(f"Error creating outperformers insights: {e}")
                insights = html.Div("Consistent outperformers insights not available")
            
            return table, insights
        except Exception as e:
            print(f"Error in consistent outperformers callback: {e}")
            return html.Div(f"Error in consistent outperformers analysis: {str(e)}"), html.Div("Error generating insights")


    @app.callback(
        [Output("statistical-significance-graph", "children"),
        Output("statistical-significance-insights", "children")],
        [Input("comp-property-type-filter", "value"),
        Input("comp-area-filter", "value"),
        Input("comp-room-type-filter", "value"),
        Input("comp-registration-type-filter", "value"),
        Input("comp-developer-filter", "value"),
        Input("comp-time-horizon-filter", "value")]
    )
    def update_statistical_significance(property_type, area, room_type, registration_type, developer, time_horizon):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        if developer != 'All' and developer is not None:
            filters['developer_name'] = developer
        
        try:
            data_quality = "medium"  # Default for statistical analysis
            coverage_pct = 50  # Default coverage
            stat_significant_count = 0
            
            # Use pre-calculated comparative data if available
            if comparative_df is not None and f'{time_horizon}_significant' in comparative_df.columns:
                # Apply filters to comparative dataframe
                filtered_comp_df = apply_filters(comparative_df, filters)
                
                if len(filtered_comp_df) > 0:
                    # Create statistical significance visualization
                    fig = create_statistical_significance_chart(filtered_comp_df, time_horizon)
                    
                    # Calculate data quality metrics
                    ci_lower = f'{time_horizon}_ci_lower'
                    ci_upper = f'{time_horizon}_ci_upper'
                    significant = f'{time_horizon}_significant'
                    
                    if all(col in filtered_comp_df.columns for col in [ci_lower, ci_upper, significant]):
                        coverage_pct = filtered_comp_df[significant].notna().mean() * 100
                        data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                        stat_significant_count = filtered_comp_df[significant].sum()
                else:
                    # Fall back to live analysis
                    filtered_df = apply_filters(df, filters)
                    try:
                        from scripts.comparative_analysis import add_statistical_significance
                        stat_df = add_statistical_significance(filtered_df)
                        fig = create_statistical_significance_chart(stat_df, time_horizon)
                        
                        # Calculate data quality metrics
                        significant = f'{time_horizon}_significant'
                        if significant in stat_df.columns:
                            coverage_pct = stat_df[significant].notna().mean() * 100
                            data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                            stat_significant_count = stat_df[significant].sum()
                    except (ImportError, AttributeError) as e:
                        print(f"Error adding statistical significance: {e}")
                        fig = create_default_figure("Statistical Significance", "Error generating statistical significance analysis")
                        data_quality = "low"
                        coverage_pct = 0
            else:
                # Perform live analysis using the comparative_analysis module
                filtered_df = apply_filters(df, filters)
                try:
                    from scripts.comparative_analysis import add_statistical_significance
                    stat_df = add_statistical_significance(filtered_df)
                    fig = create_statistical_significance_chart(stat_df, time_horizon)
                    
                    # Calculate data quality metrics
                    significant = f'{time_horizon}_significant'
                    if significant in stat_df.columns:
                        coverage_pct = stat_df[significant].notna().mean() * 100
                        data_quality = "high" if coverage_pct >= 70 else "medium" if coverage_pct >= 40 else "low"
                        stat_significant_count = stat_df[significant].sum()
                except (ImportError, AttributeError) as e:
                    print(f"Error adding statistical significance: {e}")
                    fig = create_default_figure("Statistical Significance", "Error generating statistical significance analysis")
                    data_quality = "low"
                    coverage_pct = 0
            
            # Generate insights using new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'time_horizon': time_horizon,
                    'significant_count': stat_significant_count
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'comparative',
                    'statistical_significance',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'area': area,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'developer': developer,
                        'time_horizon': time_horizon
                    },
                    metadata
                )
            except (ImportError, AttributeError) as e:
                print(f"Error creating statistical significance insights: {e}")
                insights = html.Div("Statistical significance insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in statistical significance callback: {e}")
            return dcc.Graph(figure=create_default_figure("Statistical Significance", str(e))), html.Div("Error generating insights")
        
    #--------------------------
    # Geographic Analysis Tab Callbacks
    #--------------------------
    
    @app.callback(
    [Output("price-heatmap-graph", "children"),
     Output("price-heatmap-insights", "children")],
    [Input("geo-property-type-filter", "value"),
     Input("geo-room-type-filter", "value"),
     Input("geo-registration-type-filter", "value")]
    )
    def update_price_heatmap(property_type, room_type, registration_type):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Try to load geographic data directly
            from scripts.geographic_analysis import load_kml_file, prepare_geographic_data, create_price_heatmap
            from scripts.insights import create_insights_component
            
            current_dir = os.path.dirname(__file__)  # dashboard/
            parent_dir = os.path.dirname(current_dir)  # project root  
            custom_geojson_path = os.path.join(parent_dir, 'data', 'complete_community_with_csv_names.geojson')

            if os.path.exists(custom_geojson_path):
                geo_data = load_kml_file(custom_geojson_path)
                
                if geo_data is not None:
                    # Filter the main dataframe
                    filtered_df = apply_filters(df, filters)
                    
                    # Generate area_data with proper coordinates
                    area_data = prepare_geographic_data(filtered_df, property_type, room_type, registration_type, geo_data)
                    
                    if area_data is not None and len(area_data) > 0:
                        # Generate price heatmap
                        try:
                            fig = create_price_heatmap(area_data)
                            
                            # Create metadata for insights
                            metadata = {
                                'data_quality': 'medium',  # Default to medium since geographic data has varying quality
                                'coverage_pct': (len(area_data) / len(filtered_df.groupby('area_name_en'))) * 100 if 'area_name_en' in filtered_df.columns else 50,
                            }
                            
                            # Generate insights
                            insights = create_insights_component(
                                filtered_df,
                                'geographic',
                                'price_map',
                                {
                                    'property_type': property_type,
                                    'room_type': room_type,
                                    'registration_type': registration_type
                                },
                                metadata
                            )
                            
                            return dcc.Graph(figure=fig), insights
                            
                        except Exception as e:
                            print(f"Error creating price heatmap: {e}")
                            fig = create_default_figure("Price Heatmap", f"Error generating price heatmap: {str(e)}")
                    else:
                        fig = create_default_figure("Price Heatmap", "No geographic data available for the selected filters")
                else:
                    fig = create_default_figure("Price Heatmap", "Geographic data could not be loaded")
            else:
                # Fall back to loading geo_df from the parameter
                area_data = prepare_geographic_data(df, property_type, room_type, registration_type, geo_df)
                
                if area_data is None or len(area_data) == 0:
                    fig = create_default_figure("Price Heatmap", "No geographic data available")
                else:
                    try:
                        fig = create_price_heatmap(area_data)
                    except Exception as e:
                        print(f"Error creating price heatmap: {e}")
                        fig = create_default_figure("Price Heatmap", f"Error generating price heatmap: {str(e)}")
            
            # Generate insights even if visualization failed
            try:
                # Filter the dataframe for insights
                filtered_df = apply_filters(df, filters)
                
                # Generate insights
                insights = create_insights_component(
                    filtered_df,
                    'geographic',
                    'price_map',
                    {
                        'property_type': property_type,
                        'room_type': room_type,
                        'registration_type': registration_type
                    },
                    {
                        'data_quality': 'medium',  # Default to medium since geographic data has varying quality
                        'coverage_pct': 50,  # Default coverage
                    }
                )
            except Exception as e:
                print(f"Error creating geographic insights: {e}")
                insights = html.Div("Geographic insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in price heatmap callback: {e}")
            return dcc.Graph(figure=create_default_figure("Price Heatmap", str(e))), html.Div("Error generating insights")
        
    @app.callback(
        [Output("growth-heatmap-graph", "children"),
        Output("growth-heatmap-insights", "children")],
        [Input("geo-property-type-filter", "value"),
        Input("geo-room-type-filter", "value"),
        Input("geo-registration-type-filter", "value"),
        Input("geo-year-filter", "value")]
    )
    def update_growth_heatmap(property_type, room_type, registration_type, year):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Try to load geographic data directly
            from scripts.geographic_analysis import load_kml_file, prepare_geographic_data
            current_dir = os.path.dirname(__file__)  # dashboard/
            parent_dir = os.path.dirname(current_dir)  # project root  
            custom_geojson_path = os.path.join(parent_dir, 'data', 'complete_community_with_csv_names.geojson')
            
            if os.path.exists(custom_geojson_path):
                geo_data = load_kml_file(custom_geojson_path)
                
                if geo_data is not None:
                    # Filter the main dataframe
                    filtered_df = apply_filters(df, filters)
                    
                    # Generate area_data with proper coordinates
                    area_data = prepare_geographic_data(filtered_df, property_type, room_type, registration_type, geo_data)
                    
                    if area_data is not None and len(area_data) > 0:
                        # Determine growth column based on selected year
                        growth_column = None
                        
                        # Handle case where year is not a valid year (e.g., it's 'current')
                        if year is not None and year != 'current' and isinstance(year, (int, float)):
                            years = sorted([col for col in df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025])
                            if year in years and years.index(year) > 0:
                                prev_year = years[years.index(year) - 1]
                                growth_column = f'growth_{prev_year}_to_{year}'
                        
                        if growth_column is None:
                            # Default to the most recent growth period
                            growth_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('growth_')]
                            growth_column = growth_columns[-1] if growth_columns else None
                        
                        # Generate growth heatmap
                        try:
                            from scripts.geographic_analysis import create_growth_heatmap
                            fig = create_growth_heatmap(area_data, growth_column)
                        except Exception as e:
                            print(f"Error creating growth heatmap: {e}")
                            fig = create_default_figure("Growth Heatmap", f"Error generating growth heatmap: {str(e)}")
                    else:
                        fig = create_default_figure("Growth Heatmap", "No geographic data available for the selected filters")
                else:
                    fig = create_default_figure("Growth Heatmap", "Geographic data could not be loaded")
            else:
                # Fall back to using geo_df from the parameter
                area_data = prepare_geographic_data(df, property_type, room_type, registration_type, geo_df)
                
                if area_data is None or len(area_data) == 0:
                    fig = create_default_figure("Growth Heatmap", "No geographic data available")
                else:
                    # Determine growth column based on selected year
                    growth_column = None
                    
                    if year is not None and year != 'current' and isinstance(year, (int, float)):
                        years = sorted([col for col in df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025])
                        if year in years and years.index(year) > 0:
                            prev_year = years[years.index(year) - 1]
                            growth_column = f'growth_{prev_year}_to_{year}'
                    
                    if growth_column is None:
                        # Default to the most recent growth period
                        growth_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('growth_')]
                        growth_column = growth_columns[-1] if growth_columns else None
                    
                    # Generate growth heatmap
                    try:
                        from scripts.geographic_analysis import create_growth_heatmap
                        fig = create_growth_heatmap(area_data, growth_column)
                    except Exception as e:
                        print(f"Error creating growth heatmap: {e}")
                        fig = create_default_figure("Growth Heatmap", f"Error generating growth heatmap: {str(e)}")
            
            # Generate insights using the new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Calculate data quality metrics
                data_quality = "medium"  # Default for geographic data
                coverage_pct = 0
                
                if area_data is not None and len(area_data) > 0:
                    coverage_pct = (len(area_data) / len(filtered_df['area_name_en'].unique())) * 100 if 'area_name_en' in filtered_df.columns else 50
                    
                    # Check growth column coverage if available
                    if growth_column in area_data.columns:
                        growth_coverage = area_data[growth_column].notna().mean() * 100
                        coverage_pct = growth_coverage  # Use the growth column coverage
                        data_quality = "high" if growth_coverage >= 70 else "medium" if growth_coverage >= 40 else "low"
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct,
                    'growth_column': growth_column
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'geographic',
                    'growth_map',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'room_type': room_type,
                        'registration_type': registration_type,
                        'year': year
                    },
                    metadata
                )
            except Exception as e:
                print(f"Error creating growth insights: {e}")
                insights = html.Div("Growth insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in growth heatmap callback: {e}")
            return dcc.Graph(figure=create_default_figure("Growth Heatmap", str(e))), html.Div("Error generating insights")

    @app.callback(
    [Output("investment-hotspot-graph", "children"),
     Output("investment-hotspot-insights", "children")],
    [Input("geo-property-type-filter", "value"),
     Input("geo-room-type-filter", "value"),
     Input("geo-registration-type-filter", "value")]
    )
    def update_investment_hotspot(property_type, room_type, registration_type):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Try to load geographic data directly
            from scripts.geographic_analysis import load_kml_file, prepare_geographic_data
            
            current_dir = os.path.dirname(__file__)  # dashboard/
            parent_dir = os.path.dirname(current_dir)  # project root  
            custom_geojson_path = os.path.join(parent_dir, 'data', 'complete_community_with_csv_names.geojson')

            if os.path.exists(custom_geojson_path):
                geo_data = load_kml_file(custom_geojson_path)
                
                if geo_data is not None:
                    # Filter the main dataframe
                    filtered_df = apply_filters(df, filters)
                    
                    # Generate area_data with proper coordinates
                    area_data = prepare_geographic_data(filtered_df, property_type, room_type, registration_type, geo_data)
                    
                    if area_data is not None and len(area_data) > 0:
                        # Generate investment hotspot map
                        try:
                            from scripts.geographic_analysis import create_investment_hotspot_map
                            fig = create_investment_hotspot_map(area_data)
                        except Exception as e:
                            print(f"Error creating investment hotspot map: {e}")
                            fig = create_default_figure("Investment Hotspots", f"Error generating investment hotspot map: {str(e)}")
                    else:
                        fig = create_default_figure("Investment Hotspots", "No geographic data available for the selected filters")
                else:
                    fig = create_default_figure("Investment Hotspots", "Geographic data could not be loaded")
            else:
                # Fall back to using geo_df from the parameter
                area_data = prepare_geographic_data(df, property_type, room_type, registration_type, geo_df)
                
                if area_data is None or len(area_data) == 0:
                    fig = create_default_figure("Investment Hotspots", "No geographic data available")
                else:
                    try:
                        from scripts.geographic_analysis import create_investment_hotspot_map
                        fig = create_investment_hotspot_map(area_data)
                    except Exception as e:
                        print(f"Error creating investment hotspot map: {e}")
                        fig = create_default_figure("Investment Hotspots", f"Error generating investment hotspot map: {str(e)}")
            
            # Generate insights using the new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Calculate data quality metrics
                data_quality = "medium"  # Default for geographic data
                coverage_pct = 0
                
                if area_data is not None and len(area_data) > 0:
                    coverage_pct = (len(area_data) / len(filtered_df['area_name_en'].unique())) * 100 if 'area_name_en' in filtered_df.columns else 50
                    
                    # Check investment score coverage if available
                    if 'investment_score' in area_data.columns:
                        score_coverage = area_data['investment_score'].notna().mean() * 100
                        coverage_pct = score_coverage  # Use the investment score coverage
                        data_quality = "high" if score_coverage >= 70 else "medium" if score_coverage >= 40 else "low"
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'geographic',
                    'investment_hotspot',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'room_type': room_type,
                        'registration_type': registration_type
                    },
                    metadata
                )
            except Exception as e:
                print(f"Error creating hotspot insights: {e}")
                insights = html.Div("Investment hotspot insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in investment hotspot callback: {e}")
            return dcc.Graph(figure=create_default_figure("Investment Hotspots", str(e))), html.Div("Error generating insights")

    @app.callback(
    [Output("transaction-volume-graph", "children"),
     Output("transaction-volume-insights", "children")],
    [Input("geo-property-type-filter", "value"),
     Input("geo-room-type-filter", "value"),
     Input("geo-registration-type-filter", "value")]
    )
    def update_transaction_volume(property_type, room_type, registration_type):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Try to load geographic data directly
            from scripts.geographic_analysis import load_kml_file, prepare_geographic_data
            
            current_dir = os.path.dirname(__file__)  # dashboard/
            parent_dir = os.path.dirname(current_dir)  # project root  
            custom_geojson_path = os.path.join(parent_dir, 'data', 'complete_community_with_csv_names.geojson')

            if os.path.exists(custom_geojson_path):
                geo_data = load_kml_file(custom_geojson_path)
                
                if geo_data is not None:
                    # Filter the main dataframe
                    filtered_df = apply_filters(df, filters)
                    
                    # Generate area_data with proper coordinates
                    area_data = prepare_geographic_data(filtered_df, property_type, room_type, registration_type, geo_data)
                    
                    if area_data is not None and len(area_data) > 0:
                        # Generate transaction volume map
                        try:
                            from scripts.geographic_analysis import create_transaction_volume_map
                            fig = create_transaction_volume_map(area_data)
                        except Exception as e:
                            print(f"Error creating transaction volume map: {e}")
                            fig = create_default_figure("Transaction Volume", f"Error generating transaction volume map: {str(e)}")
                    else:
                        fig = create_default_figure("Transaction Volume", "No geographic data available for the selected filters")
                else:
                    fig = create_default_figure("Transaction Volume", "Geographic data could not be loaded")
            else:
                # Fall back to using geo_df from the parameter
                area_data = prepare_geographic_data(df, property_type, room_type, registration_type, geo_df)
                
                if area_data is None or len(area_data) == 0:
                    fig = create_default_figure("Transaction Volume", "No geographic data available")
                else:
                    try:
                        from scripts.geographic_analysis import create_transaction_volume_map
                        fig = create_transaction_volume_map(area_data)
                    except Exception as e:
                        print(f"Error creating transaction volume map: {e}")
                        fig = create_default_figure("Transaction Volume", f"Error generating transaction volume map: {str(e)}")
            
            # Generate insights using the new insights module
            try:
                from scripts.insights import create_insights_component
                filtered_df = apply_filters(df, filters)
                
                # Calculate data quality metrics
                data_quality = "high"  # Transaction data typically has high quality
                coverage_pct = 0
                
                if area_data is not None and len(area_data) > 0:
                    coverage_pct = (len(area_data) / len(filtered_df['area_name_en'].unique())) * 100 if 'area_name_en' in filtered_df.columns else 50
                    
                    # Check transaction count coverage if available
                    if 'transaction_count' in area_data.columns:
                        transaction_coverage = area_data['transaction_count'].notna().mean() * 100
                        coverage_pct = transaction_coverage  # Use the transaction count coverage
                
                # Create metadata dictionary
                metadata = {
                    'data_quality': data_quality,
                    'coverage_pct': coverage_pct
                }
                
                insights = create_insights_component(
                    filtered_df, 
                    'geographic',
                    'transaction_volume',  # Specific visualization type
                    {
                        'property_type': property_type,
                        'room_type': room_type,
                        'registration_type': registration_type
                    },
                    metadata
                )
            except Exception as e:
                print(f"Error creating volume insights: {e}")
                insights = html.Div("Transaction volume insights not available")
            
            return dcc.Graph(figure=fig), insights
        except Exception as e:
            print(f"Error in transaction volume callback: {e}")
            return dcc.Graph(figure=create_default_figure("Transaction Volume", str(e))), html.Div("Error generating insights")
    
#--------------------------
    # Launch to Completion Tab Callbacks (Project Analysis)
    #--------------------------
    
    @app.callback(
        [Output("project-appreciation-chart", "children"),
         Output("project-appreciation-insights", "children")],
        [Input("ltc-property-type-filter", "value"),
         Input("ltc-area-filter", "value"),
         Input("ltc-developer-filter", "value"),
         Input("ltc-room-type-filter", "value")]  # Added room type filter
    )
    def update_individual_projects(property_type, area, developer, room_type):
        from scripts.project_analysis import prepare_project_data, create_individual_project_analysis, prepare_insights_metadata, get_peer_group_info
        from scripts.insights import create_insights_component
        """Update individual project performance visualization"""
        
        # Handle empty filter selections - default to 'All'
        property_type = property_type if property_type else 'All'
        area = area if area else 'All'
        developer = developer if developer else 'All'
        room_type = room_type if room_type else 'All'
        
        try:
            # Re-process data with deed-level filtering
            # Note: prepare_project_data now handles all filtering internally
            filtered_df = prepare_project_data(
                property_type=property_type,
                area=area,
                developer=developer,
                room_type=room_type
            )
            
            if len(filtered_df) == 0:
                return dcc.Graph(
                    id="project-appreciation-chart",  # Add ID to the graph component
                    figure=create_default_figure("Individual Project Analysis", "No projects match your current filters. Try adjusting your selection."),
                    config={'responsive': True},
                    style={'width': '100%', 'height': '100%'}
                ), html.Div("No data available for insights")
            
            # Create visualization
            fig = create_individual_project_analysis(filtered_df)
            
            # Prepare metadata for insights
            metadata = prepare_insights_metadata(filtered_df)
            
            # Add context-specific metadata if specific filters are applied
            peer_info = get_peer_group_info(filtered_df, property_type, area)
            metadata.update(peer_info)
            
            # Generate insights
            insights = create_insights_component(
                filtered_df, 
                'project_analysis',
                'individual_projects',
                {
                    'property_type': property_type,
                    'area': area,
                    'developer': developer,
                    'room_type': room_type  # Added room type to filters
                },
                metadata
            )
            
            return dcc.Graph(
                id="project-appreciation-graph",  # Add ID to the graph component
                figure=fig,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                },
                style={
                    'width': '100%', 
                    'height': '100%',
                    'minHeight': '500px',
                    'marginBottom': '30px'  # Add margin to prevent overlap
                },
                className="responsive-chart"
            ), insights
            
        except Exception as e:
            print(f"Error in individual projects callback: {e}")
            import traceback
            traceback.print_exc()
            return dcc.Graph(
                id="project-appreciation-chart",
                figure=create_default_figure("Individual Project Analysis", str(e)),
                config={'responsive': True},
                style={'width': '100%', 'height': '100%'}
            ), html.Div("Error generating insights")

    # Add new callback for project drill-down (room type breakdown)
    @app.callback(
        Output("project-drilldown-table", "children"),
        [Input("project-appreciation-graph", "clickData"),
         State("ltc-property-type-filter", "value"),
         State("ltc-area-filter", "value"),
         State("ltc-developer-filter", "value"),
         State("ltc-room-type-filter", "value"),  # Add room type state
         State("project-appreciation-graph", "children")]  # Get the graph component to access data
    )
    def update_project_drilldown(clickData, property_type, area, developer, room_type, graph_component):
        """Show room-type breakdown when a project is clicked"""
        from scripts.project_analysis import get_project_room_breakdown, prepare_project_data
        
        if clickData is None:
            return html.Div()  # Return empty div when no project is clicked
        
        try:
            # Get the clicked project name
            clicked_point = clickData['points'][0]
            project_name = clicked_point.get('y', '')  # Project name is on y-axis
            
            if not project_name:
                return html.Div("Unable to identify the clicked project")
            
            # Re-load the filtered data to get project IDs
            filtered_df = prepare_project_data(
                property_type=property_type if property_type else 'All',
                area=area if area else 'All',
                developer=developer if developer else 'All',
                room_type=room_type if room_type else 'All'
            )
            
            # Find the project ID by matching the project name
            project_row = filtered_df[filtered_df['project_name_en'] == project_name]
            
            if len(project_row) == 0:
                return html.Div(f"Project '{project_name}' not found in filtered data")
            
            project_id = int(project_row.iloc[0]['project_number_int'])
            
            # Get room breakdown for this project
            room_breakdown = get_project_room_breakdown(
                project_id, 
                property_type=property_type if property_type else 'All',
                area=area if area else 'All',
                developer=developer if developer else 'All'
            )
            
            if len(room_breakdown) == 0:
                # If no breakdown available, show a message based on current filter
                if room_type and room_type != 'All':
                    return html.Div([
                        html.H5(f"Room Type Breakdown: {project_name}", className="mt-3 mb-2"),
                        html.P(f"Currently filtered to show only {room_type} units. No additional breakdown available.", 
                               className="text-muted")
                    ])
                else:
                    return html.Div([
                        html.H5(f"Room Type Breakdown: {project_name}", className="mt-3 mb-2"),
                        html.P("Insufficient data to calculate room-type breakdown for this project.", 
                               className="text-muted")
                    ])
            
            # Prepare data for display table
            room_breakdown = room_breakdown.rename(columns={
                'room_type': 'Room Type',
                'cagr': 'CAGR (%)',
                'launch_price_sqft': 'Launch Price',
                'recent_price_sqft': 'Recent Price',
                'transaction_count': 'Total Transactions',
                'recent_deeds': 'Recent Deeds'
            })
            
            # Round numeric columns
            room_breakdown['CAGR (%)'] = room_breakdown['CAGR (%)'].round(1)
            room_breakdown['Launch Price'] = room_breakdown['Launch Price'].round(0)
            room_breakdown['Recent Price'] = room_breakdown['Recent Price'].round(0)
            
            # Create quality flag column for display
            room_breakdown['Quality Flag'] = room_breakdown.apply(
                lambda x: 'Thin Data' if x.get('Recent Deeds', 0) < 3 else 
                         'Review Needed' if abs(x.get('CAGR (%)', 0)) > 400 else 'Good', 
                axis=1
            )
            
            # Create dash table
            table = dash_table.DataTable(
                data=room_breakdown.to_dict('records'),
                columns=[
                    {"name": "Room Type", "id": "Room Type"},
                    {"name": "CAGR (%)", "id": "CAGR (%)", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "Launch Price", "id": "Launch Price", "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "Recent Price", "id": "Recent Price", "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "Total Txns", "id": "Total Transactions", "type": "numeric"},
                    {"name": "Quality", "id": "Quality Flag"}
                ],
                style_cell={'textAlign': 'left', 'padding': '8px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'column_id': 'CAGR (%)', 'filter_query': '{CAGR (%)} > 0'},
                        'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                    },
                    {
                        'if': {'column_id': 'CAGR (%)', 'filter_query': '{CAGR (%)} < 0'},
                        'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                    },
                    {
                        'if': {'column_id': 'Quality Flag', 'filter_query': '{Quality Flag} contains "Review"'},
                        'backgroundColor': 'rgba(255, 0, 0, 0.3)',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Quality Flag', 'filter_query': '{Quality Flag} contains "Thin"'},
                        'backgroundColor': 'rgba(255, 165, 0, 0.2)',
                    }
                ]
            )
            
            # Return table with a title
            return html.Div([
                html.H5(f"Room Type Breakdown: {project_name}", className="mt-3 mb-2"),
                table
            ], className="mb-4")
            
        except Exception as e:
            print(f"Error in project drilldown callback: {e}")
            import traceback
            traceback.print_exc()
            return html.Div(f"Error loading room breakdown: {str(e)}")
        
    @app.callback(
        [Output("area-comparison-graph", "children"),
         Output("area-comparison-insights", "children")],
        [Input("ltc-property-type-filter", "value"),
         Input("ltc-area-filter", "value"),
         Input("ltc-developer-filter", "value"),
         Input("ltc-room-type-filter", "value")]  # Added room type filter
    )
    def update_project_area_comparison(property_type, area, developer, room_type):
        from scripts.project_analysis import prepare_project_data, create_area_performance_comparison, prepare_insights_metadata
        from scripts.insights import create_insights_component
        """Update area performance comparison visualization"""
        
        # Handle empty filter selections - default to 'All'
        property_type = property_type if property_type else 'All'
        area = area if area else 'All'
        developer = developer if developer else 'All'
        room_type = room_type if room_type else 'All'
        
        try:
            # Re-process data with deed-level filtering
            filtered_df = prepare_project_data(
                property_type=property_type,
                area=area,
                developer=developer,
                room_type=room_type
            )
            
            if len(filtered_df) == 0:
                return dcc.Graph(
                    figure=create_default_figure("Area Performance Comparison", "No projects match your current filters. Try adjusting your selection."),
                    config={'responsive': True},
                    style={'width': '100%', 'height': '100%'}
                ), html.Div("No data available for insights")
            
            # Create visualization
            fig = create_area_performance_comparison(filtered_df)
            
            # Prepare metadata for insights
            metadata = prepare_insights_metadata(filtered_df)
            
            # Add area-specific metrics
            if len(filtered_df) > 0:
                area_stats = filtered_df.groupby('area_name_en')['cagr'].agg(['mean', 'count']).reset_index()
                area_stats = area_stats[area_stats['count'] >= 3]  # Only areas with 3+ projects
                
                if len(area_stats) > 0:
                    area_stats = area_stats.sort_values('mean', ascending=False)
                    top_area = area_stats.iloc[0]
                    
                    metadata.update({
                        'area_count': len(area_stats),
                        'top_performing_area': top_area['area_name_en'],
                        'top_area_cagr': top_area['mean'],
                        'top_area_project_count': top_area['count'],
                        'area_diversity_score': len(filtered_df['area_name_en'].unique()),
                        'premium_area_multiplier': 1.5,  # This could be calculated from actual data
                        'location_value_premium': 25,  # This could be calculated from actual data
                        'top5_area_market_share': 42,  # This could be calculated from actual data
                        'is_room_filtered': room_type != 'All',
                        'filtered_room_type': room_type if room_type != 'All' else None
                    })
            
            # Generate insights
            insights = create_insights_component(
                filtered_df, 
                'project_analysis',
                'area_comparison',
                {
                    'property_type': property_type,
                    'area': area,
                    'developer': developer,
                    'room_type': room_type  # Added room type to filters
                },
                metadata
            )
            
            return dcc.Graph(
                figure=fig,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                },
                style={
                    'width': '100%', 
                    'height': '100%',
                    'minHeight': '500px',
                    'marginBottom': '30px'  # Add margin to prevent overlap
                },
                className="responsive-chart"
            ), insights
            
        except Exception as e:
            print(f"Error in area comparison callback: {e}")
            import traceback
            traceback.print_exc()
            return dcc.Graph(
                figure=create_default_figure("Area Performance Comparison", str(e)),
                config={'responsive': True},
                style={'width': '100%', 'height': '100%'}
            ), html.Div("Error generating insights")

    @app.callback(
        [Output("developer-comparison-graph", "children"),
         Output("developer-comparison-insights", "children")],
        [Input("ltc-property-type-filter", "value"),
         Input("ltc-area-filter", "value"),
         Input("ltc-developer-filter", "value"),
         Input("ltc-room-type-filter", "value")]  # Added room type filter
    )
    def update_project_developer_comparison(property_type, area, developer, room_type):
        from scripts.project_analysis import prepare_project_data, create_developer_track_record_analysis, prepare_insights_metadata
        from scripts.insights import create_insights_component
        """Update developer track record comparison visualization"""
        
        # Handle empty filter selections - default to 'All'
        property_type = property_type if property_type else 'All'
        area = area if area else 'All'
        developer = developer if developer else 'All'
        room_type = room_type if room_type else 'All'
        
        try:
            # Re-process data with deed-level filtering
            filtered_df = prepare_project_data(
                property_type=property_type,
                area=area,
                developer=developer,
                room_type=room_type
            )
            
            if len(filtered_df) == 0:
                return dcc.Graph(
                    figure=create_default_figure("Developer Track Record", "No projects match your current filters. Try adjusting your selection."),
                    config={'responsive': True},
                    style={'width': '100%', 'height': '100%'}
                ), html.Div("No data available for insights")
            
            # Create visualization
            fig = create_developer_track_record_analysis(filtered_df)
            
            # Prepare metadata for insights
            metadata = prepare_insights_metadata(filtered_df)
            
            # Add developer-specific metrics
            if len(filtered_df) > 0:
                dev_stats = filtered_df.groupby('developer_name')['cagr'].agg(['mean', 'count']).reset_index()
                dev_stats = dev_stats[dev_stats['count'] >= 2]  # Only developers with 2+ projects
                
                if len(dev_stats) > 0:
                    dev_stats = dev_stats.sort_values('mean', ascending=False)
                    top_dev = dev_stats.iloc[0]
                    
                    # Calculate consistency metric
                    top_dev_projects = filtered_df[filtered_df['developer_name'] == top_dev['developer_name']]
                    consistency = (top_dev_projects['cagr'] > filtered_df['cagr'].mean()).mean() * 100
                    
                    metadata.update({
                        'developer_count': len(dev_stats),
                        'top_performing_developer': top_dev['developer_name'],
                        'top_developer_cagr': top_dev['mean'],
                        'top_developer_project_count': top_dev['count'],
                        'top_developer_consistency': consistency,
                        'premium_developer_volatility_reduction': 35,  # This could be calculated from actual data
                        'experienced_developer_premium': 1.2,  # This could be calculated from actual data
                        'top5_market_share': 42,  # This could be calculated from actual data
                        'is_room_filtered': room_type != 'All',
                        'filtered_room_type': room_type if room_type != 'All' else None
                    })
            
            # Generate insights
            insights = create_insights_component(
                filtered_df, 
                'project_analysis',
                'developer_comparison',
                {
                    'property_type': property_type,
                    'area': area,
                    'developer': developer,
                    'room_type': room_type  # Added room type to filters
                },
                metadata
            )
            
            return dcc.Graph(
                figure=fig,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                },
                style={
                    'width': '100%', 
                    'height': '100%',
                    'minHeight': '500px',
                    'marginBottom': '30px'  # Add margin to prevent overlap
                },
                className="responsive-chart"
            ), insights
            
        except Exception as e:
            print(f"Error in developer comparison callback: {e}")
            import traceback
            traceback.print_exc()
            return dcc.Graph(
                figure=create_default_figure("Developer Track Record", str(e)),
                config={'responsive': True},
                style={'width': '100%', 'height': '100%'}
            ), html.Div("Error generating insights")
                
    #--------------------------
    # Investment Metrics Callback
    #--------------------------
    @app.callback(
        [Output("investment-avg-growth-metric", "children"),
        Output("investment-top-area-metric", "children"),
        Output("investment-opportunities-count", "children"),
        Output("investment-median-price-metric", "children")],
        [Input("investment-property-type-filter", "value"),
        Input("investment-area-filter", "value"),
        Input("investment-room-type-filter", "value"),
        Input("investment-registration-type-filter", "value"),
        Input("investment-time-horizon-filter", "value")]
    )
    def update_investment_metrics(property_type, area, room_type, registration_type, time_horizon):
        # Create filter dictionary
        filters = {
            'property_type_en': property_type if property_type != 'All' else None,
            'area_name_en': area if area != 'All' else None,
            'rooms_en': room_type if room_type != 'All' else None,
            'reg_type_en': registration_type if registration_type != 'All' else None
        }
        
        try:
            # Apply progressive filtering - start with all filters
            filtered_df = progressive_filtering(df, filters)
            
            # Use improved selection of growth column
            try:
                from scripts.investment_analysis import select_best_growth_column, perform_microsegment_analysis
                
                # Get growth column and its coverage metrics
                growth_column, coverage_pct, data_quality = select_best_growth_column(
                    filtered_df, 
                    requested_horizon=time_horizon
                )
                
                # Calculate avg growth - handle special case of calculated growth
                if growth_column and isinstance(growth_column, str) and growth_column.startswith('calc_growth_'):
                    _, prev_year, _, latest_year = growth_column.split('_')
                    prev_year = int(prev_year)
                    latest_year = int(latest_year)
                    
                    # Calculate growth for valid rows only
                    mask = (filtered_df[prev_year].notna() & filtered_df[latest_year].notna() & (filtered_df[prev_year] > 0))
                    
                    if mask.sum() > 0:
                        growth_values = ((filtered_df.loc[mask, latest_year] / filtered_df.loc[mask, prev_year]) - 1) * 100
                        avg_growth = growth_values.median()
                        avg_growth_display = f"{avg_growth:.1f}%" if not pd.isna(avg_growth) else "--"
                    else:
                        avg_growth_display = "--"
                elif growth_column and growth_column in filtered_df.columns:
                    avg_growth = filtered_df[growth_column].median()
                    avg_growth_display = f"{avg_growth:.1f}%" if not pd.isna(avg_growth) else "--"
                else:
                    avg_growth_display = "--"
                
                # Add data quality indicator to growth display
                if data_quality != "none" and coverage_pct > 0:
                    quality_indicator = "★★★" if data_quality == "high" else "★★" if data_quality == "medium" else "★"
                    avg_growth_display = f"{avg_growth_display} {quality_indicator}"
            except Exception as e:
                print(f"Error calculating growth metrics: {e}")
                avg_growth_display = "--"
            
            # Get top area using available growth column
            top_area_display = "--"
            try:
                if growth_column and 'area_name_en' in filtered_df.columns:
                    # Handle calculated growth case
                    if isinstance(growth_column, str) and growth_column.startswith('calc_growth_'):
                        _, prev_year, _, latest_year = growth_column.split('_')
                        prev_year = int(prev_year)
                        latest_year = int(latest_year)
                        
                        # Only consider areas with sufficient data
                        valid_areas = []
                        for area_name in filtered_df['area_name_en'].unique():
                            area_df = filtered_df[filtered_df['area_name_en'] == area_name]
                            mask = (area_df[prev_year].notna() & area_df[latest_year].notna() & (area_df[prev_year] > 0))
                            if mask.sum() >= 5:  # Require at least 5 data points
                                growth = ((area_df.loc[mask, latest_year] / area_df.loc[mask, prev_year]) - 1) * 100
                                valid_areas.append((area_name, growth.median()))
                                
                        if valid_areas:
                            # Sort by growth rate and get top area
                            valid_areas.sort(key=lambda x: x[1], reverse=True)
                            top_area = valid_areas[0][0]
                            top_area_display = top_area.split(' ')[0] if len(top_area) > 10 else top_area
                    elif growth_column in filtered_df.columns:
                        # Group by area and calculate median growth
                        area_growth = filtered_df.groupby('area_name_en')[growth_column].median().reset_index()
                        area_growth = area_growth.dropna()  # Remove areas with NaN growth
                        
                        if len(area_growth) > 0:
                            top_area = area_growth.sort_values(growth_column, ascending=False).iloc[0]['area_name_en']
                            top_area_display = top_area.split(' ')[0] if len(top_area) > 10 else top_area
            except Exception as e:
                print(f"Error finding top area: {e}")
            
            # Count investment opportunities using improved microsegment analysis
            opportunities_display = "--"
            try:
                # Use our enhanced function that handles sparse data better
                micro_segments, _ = perform_microsegment_analysis(filtered_df, filters=filters, growth_column=time_horizon)
                
                if 'investment_score' in micro_segments.columns:
                    opportunities_count = (micro_segments['investment_score'] >= 70).sum()
                    opportunities_display = str(opportunities_count) if opportunities_count > 0 else "--"
            except Exception as e:
                print(f"Error counting investment opportunities: {e}")
            
            # Calculate median price
            if 'median_price_sqft' in filtered_df.columns:
                median_price = filtered_df['median_price_sqft'].median()
                median_price_display = f"{int(median_price):,}" if not pd.isna(median_price) else "--"
            else:
                median_price_display = "--"
            
            return avg_growth_display, top_area_display, opportunities_display, median_price_display
        except Exception as e:
            print(f"Error in investment metrics callback: {e}")
            return "--", "--", "--", "--"

# Helper function for progressive filtering
def progressive_filtering(df, filters, min_rows=10):
    """
    Apply filters progressively, relaxing constraints if results are too sparse
    
    Args:
        df (pd.DataFrame): The original dataframe
        filters (dict): Dictionary mapping column names to filter values
        min_rows (int): Minimum number of rows required
        
    Returns:
        pd.DataFrame: Filtered dataframe with sufficient data
    """
    # Apply initial filters
    filtered_df = apply_filters(df, filters)
    
    # If we have enough data, return it
    if len(filtered_df) >= min_rows:
        return filtered_df
    
    print(f"Initial filtering returned only {len(filtered_df)} rows, trying progressive relaxation")
    
    # Define filter relaxation order (from most to least important to keep)
    relaxation_order = ['property_type_en', 'reg_type_en', 'area_name_en', 'rooms_en']
    
    # Iteratively relax filters
    remaining_filters = dict(filters)
    
    for filter_to_relax in relaxation_order:
        # Skip if this filter isn't active
        if filter_to_relax not in remaining_filters or remaining_filters[filter_to_relax] is None:
            continue
        
        # Create new filter set without this filter
        relaxed_filters = dict(remaining_filters)
        del relaxed_filters[filter_to_relax]
        
        # Apply relaxed filters
        relaxed_df = apply_filters(df, relaxed_filters)
        
        print(f"Relaxed {filter_to_relax} filter, got {len(relaxed_df)} rows")
        
        # If we have enough data now, return it
        if len(relaxed_df) >= min_rows:
            return relaxed_df
            
        # Update remaining filters for next iteration
        remaining_filters = relaxed_filters
    
    # If we've relaxed all filters and still don't have enough data,
    # return whatever we have from the most relaxed filtering
    return apply_filters(df, remaining_filters)

# Helper function for applying filters
def apply_filters(df, filters):
    """
    Apply a dictionary of filters to a dataframe
    
    Args:
        df (pd.DataFrame): The dataframe to filter
        filters (dict): Dictionary mapping column names to filter values
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    for col, value in filters.items():
        if value is not None and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == value]
    
    return filtered_df
    
# Fix emerging segments table creation function
def create_emerging_segments_table(emerging_segments_df):
    """
    Create a table visualization for emerging segments
    
    Args:
        emerging_segments_df (pd.DataFrame): DataFrame with emerging segments
        
    Returns:
        dash component: Dash data table with emerging segments
    """
    print(f"Creating emerging segments table with {len(emerging_segments_df)} rows")
    
    # Select columns to display
    display_columns = []
    
    # Core columns that should always be included if available
    core_columns = ['property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en', 
                   'median_price_sqft', 'transaction_count']
    
    # Growth metrics in order of preference
    growth_metrics = ['short_term_growth', 'recent_growth', 'growth_acceleration']
    
    # Add core columns if available
    for col in core_columns:
        if col in emerging_segments_df.columns:
            display_columns.append(col)
    
    # Add growth metrics if available
    for metric in growth_metrics:
        if metric in emerging_segments_df.columns:
            display_columns.append(metric)
    
    # Prepare display DataFrame
    if len(display_columns) > 0:
        display_df = emerging_segments_df[display_columns].copy()
        
        # Sort by short-term growth if available
        if 'short_term_growth' in display_df.columns:
            display_df = display_df.sort_values('short_term_growth', ascending=False)
        elif 'recent_growth' in display_df.columns:
            display_df = display_df.sort_values('recent_growth', ascending=False)
        
        # Rename columns for display
        column_map = {
            'property_type_en': 'Property Type',
            'rooms_en': 'Room Config',
            'reg_type_en': 'Registration Type',
            'area_name_en': 'Area',
            'median_price_sqft': 'Price (AED/sqft)',
            'transaction_count': 'Transactions',
            'short_term_growth': 'Current Growth (%)',
            'recent_growth': 'Current Growth (%)',
            'growth_acceleration': 'Acceleration (%)'
        }
        
        display_df = display_df.rename(columns={col: column_map.get(col, col) for col in display_df.columns})
        
        # Format numeric columns
        for col in display_df.columns:
            if col == 'Price (AED/sqft)' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif col == 'Transactions' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif 'Growth' in col or 'Acceleration' in col and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(1)
        
        # Limit to top 15 rows
        display_df = display_df.head(15)
        
        # Create style conditions for highlighting
        style_conditions = []
        
        # Highlight growth
        if 'Current Growth (%)' in display_df.columns:
            style_conditions.append({
                'if': {'column_id': 'Current Growth (%)', 'filter_query': '{Current Growth (%)} >= 10'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            })
        
        # Highlight acceleration
        if 'Acceleration (%)' in display_df.columns:
            style_conditions.append({
                'if': {'column_id': 'Acceleration (%)', 'filter_query': '{Acceleration (%)} >= 5'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            })
        
        # Create the table
        table = dash_table.DataTable(
            id='emerging-segments-table',
            columns=[
                {"name": i, "id": i} for i in display_df.columns
            ],
            data=display_df.to_dict('records'),
            sort_action="native",
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '8px'
            },
            style_data_conditional=style_conditions
        )
        
        return html.Div([
            html.H6("Emerging Segments with Accelerating Growth", className="mb-2"),
            html.P("Segments showing strong recent growth and positive acceleration compared to previous periods."),
            table
        ], className="table-container")
    else:
        return html.Div([
            html.H5("Limited Data Available"),
            html.P("The selected filters don't provide enough data for emerging segments analysis.")
        ])

def create_microsegment_table(microsegment_df):
    """
    Create an improved table visualization for microsegment analysis
    
    Args:
        microsegment_df (pd.DataFrame): DataFrame with microsegment analysis
        
    Returns:
        dash component: Dash data table
    """
    # Print debug info to see what's happening
    print(f"Creating microsegment table with {len(microsegment_df) if microsegment_df is not None else 'None'} rows")
    
    # Handle empty dataframe
    if microsegment_df is None or len(microsegment_df) == 0:
        return html.Div([
            html.H5("No Data Available"),
            html.P("No microsegment data available for the selected filters.")
        ])
    
    # Make a deep copy to avoid modifying the original dataframe
    micro_df = microsegment_df.copy(deep=True)
    
    # Select columns to display - limit to the most important ones for better fit
    display_columns = []
    
    # Core columns that should always be included if available
    core_columns = ['property_type_en', 'rooms_en', 'area_name_en', 
                   'median_price_sqft', 'transaction_count']
    
    # Growth metrics in order of preference
    growth_metrics = ['short_term_growth', 'recent_growth', 'medium_term_growth', 'long_term_cagr']
    
    # Score metrics in order of preference
    score_metrics = ['investment_score', 'outperformance_ratio']
    
    # Add core columns if available
    for col in core_columns:
        if col in micro_df.columns:
            display_columns.append(col)
    
    # Add a growth metric if available
    growth_col_found = False
    for metric in growth_metrics:
        if metric in micro_df.columns:
            display_columns.append(metric)
            growth_col_found = True
            break
    
    # Add a score metric if available
    score_col_found = False
    for metric in score_metrics:
        if metric in micro_df.columns:
            display_columns.append(metric)
            score_col_found = True
            break
    
    # Prepare display DataFrame
    if len(display_columns) > 0:
        print(f"Selected columns for display: {display_columns}")
        # Only include columns that exist in the dataframe
        valid_columns = [col for col in display_columns if col in micro_df.columns]
        display_df = micro_df[valid_columns].copy()
        
        # Sort by score or growth if available
        if score_col_found:
            for metric in score_metrics:
                if metric in display_df.columns:
                    display_df = display_df.sort_values(metric, ascending=False)
                    break
        elif growth_col_found:
            for metric in growth_metrics:
                if metric in display_df.columns:
                    display_df = display_df.sort_values(metric, ascending=False)
                    break
        
        # Rename columns for display
        column_map = {
            'property_type_en': 'Property Type',
            'rooms_en': 'Room Config',
            'reg_type_en': 'Registration Type',
            'area_name_en': 'Area',
            'median_price_sqft': 'Price (AED/sqft)',
            'transaction_count': 'Transactions',
            'short_term_growth': 'Growth (%)',
            'recent_growth': 'Growth (%)',
            'medium_term_growth': 'Medium-Term Growth (%)',
            'long_term_cagr': 'Long-Term CAGR (%)',
            'investment_score': 'Investment Score',
            'outperformance_ratio': 'Outperformance Ratio'
        }
        
        display_df = display_df.rename(columns={col: column_map.get(col, col) for col in display_df.columns})
        
        # Format numeric columns with proper type handling
        for col in display_df.columns:
            if col == 'Price (AED/sqft)' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif col == 'Transactions' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif 'Growth' in col and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(1)
            elif col == 'Investment Score' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(1)
            elif col == 'Outperformance Ratio' and col in display_df.columns:
                display_df[col] = (display_df[col].astype(float) * 100).round(0).astype('Int64')
        
        # Limit to top 15 rows
        display_df = display_df.head(15)
        
        # Create style conditions for highlighting
        style_conditions = []
        
        # Highlight investment score
        if 'Investment Score' in display_df.columns:
            style_conditions.extend([
                {
                    'if': {'column_id': 'Investment Score', 'filter_query': '{Investment Score} >= 80'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'Investment Score', 'filter_query': '{Investment Score} >= 65 && {Investment Score} < 80'},
                    'backgroundColor': 'rgba(255, 255, 0, 0.2)'
                }
            ])
        
        # Highlight growth
        growth_col = next((col for col in display_df.columns if 'Growth' in col), None)
        if growth_col:
            style_conditions.extend([
                {
                    'if': {'column_id': growth_col, 'filter_query': f'{{{growth_col}}} >= 10'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.2)'
                },
                {
                    'if': {'column_id': growth_col, 'filter_query': f'{{{growth_col}}} < 0'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.2)'
                }
            ])
        
        # Define column widths - using style_cell_conditional
        column_widths = {
            'Property Type': '15%',
            'Room Config': '12%',
            'Area': '20%',
            'Price (AED/sqft)': '13%',
            'Growth (%)': '13%',
            'Transactions': '12%',
            'Investment Score': '15%'
        }
        
        # Create style_cell_conditional for column widths
        style_cell_conditional = [
            {'if': {'column_id': col}, 'width': width} 
            for col, width in column_widths.items() 
            if col in display_df.columns
        ]
        
        # Create columns configuration with dropdown properties
        table_columns = []
        for col in display_df.columns:
            column_config = {
                "name": col,
                "id": col,
                "type": "numeric" if col not in ['Property Type', 'Room Config', 'Area'] else "text",
            }
            
            table_columns.append(column_config)
        
        # Convert the data to records format carefully
        try:
            records = display_df.to_dict('records')
            print(f"Successfully created {len(records)} records for table")
        except Exception as e:
            print(f"Error converting dataframe to records: {e}")
            records = []
        
        # Create the table
        table = dash_table.DataTable(
            id='microsegment-table',
            columns=table_columns,
            data=records,
            sort_action="native",
            page_size=10,
            style_table={
                'overflowX': 'auto',
                'width': '100%',
                'minWidth': '100%',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'whiteSpace': 'normal',
                'height': 'auto',
                'padding': '5px',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'whiteSpace': 'normal',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': '0',
            },
            style_cell_conditional=style_cell_conditional,
            style_data_conditional=style_conditions,
            tooltip_delay=0,
            tooltip_duration=None
        )
        
        # Return the table wrapped in a div
        return html.Div([
            html.H6("Top Investment Opportunities by Micro-Segment", className="mt-2 mb-2"),
            html.P("Segments are ranked by investment potential, combining growth, price level, and liquidity.", className="mb-2"),
            table
        ], className="table-container", id="microsegment-table-container")
    else:
        # Return a message if no data is available
        return html.Div([
            html.H5("Limited Data Available"),
            html.P("The selected filters don't provide enough data for microsegment analysis.")
        ])
    
# Enhanced version with data quality indicators
def create_emerging_segments_table(emerging_segments_df, metadata=None):
    """
    Create a table visualization for emerging segments with data quality indicators
    
    Args:
        emerging_segments_df (pd.DataFrame): DataFrame with emerging segments
        metadata (dict): Dictionary with data quality information
        
    Returns:
        dash component: Dash data table with emerging segments
    """
    print(f"Creating emerging segments table with {len(emerging_segments_df)} rows")
    
    # Process metadata
    data_quality = "unknown"
    coverage_pct = 0
    estimation_method = None
    if metadata is not None:
        data_quality = metadata.get('data_quality', 'unknown')
        coverage_pct = metadata.get('coverage_pct', 0)
        estimation_method = metadata.get('estimation_method')
    
    # Select columns to display
    display_columns = []
    
    # Core columns that should always be included if available
    core_columns = ['property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en', 
                   'median_price_sqft', 'transaction_count']
    
    # Growth metrics in order of preference
    growth_metrics = ['short_term_growth', 'recent_growth', 'growth_acceleration']
    
    # Add core columns if available
    for col in core_columns:
        if col in emerging_segments_df.columns:
            display_columns.append(col)
    
    # Add growth metrics if available
    for metric in growth_metrics:
        if metric in emerging_segments_df.columns:
            display_columns.append(metric)
    
    # Prepare display DataFrame
    if len(display_columns) > 0:
        display_df = emerging_segments_df[display_columns].copy()
        
        # Sort by short-term growth if available
        if 'short_term_growth' in display_df.columns:
            display_df = display_df.sort_values('short_term_growth', ascending=False)
        elif 'recent_growth' in display_df.columns:
            display_df = display_df.sort_values('recent_growth', ascending=False)
        
        # Rename columns for display
        column_map = {
            'property_type_en': 'Property Type',
            'rooms_en': 'Room Config',
            'reg_type_en': 'Registration Type',
            'area_name_en': 'Area',
            'median_price_sqft': 'Price (AED/sqft)',
            'transaction_count': 'Transactions',
            'short_term_growth': 'Current Growth (%)',
            'recent_growth': 'Current Growth (%)',
            'growth_acceleration': 'Acceleration (%)'
        }
        
        display_df = display_df.rename(columns={col: column_map.get(col, col) for col in display_df.columns})
        
        # Format numeric columns
        for col in display_df.columns:
            if col == 'Price (AED/sqft)' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif col == 'Transactions' and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(0).astype('Int64')
            elif 'Growth' in col or 'Acceleration' in col and col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(1)
        
        # Limit to top 15 rows
        display_df = display_df.head(15)
        
        # Create style conditions for highlighting
        style_conditions = []
        
        # Highlight growth
        if 'Current Growth (%)' in display_df.columns:
            style_conditions.append({
                'if': {'column_id': 'Current Growth (%)', 'filter_query': '{Current Growth (%)} >= 10'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            })
        
        # Highlight acceleration
        if 'Acceleration (%)' in display_df.columns:
            style_conditions.append({
                'if': {'column_id': 'Acceleration (%)', 'filter_query': '{Acceleration (%)} >= 5'},
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            })
        
        # Create the table
        table = dash_table.DataTable(
            id='emerging-segments-table',
            columns=[
                {"name": i, "id": i} for i in display_df.columns
            ],
            data=display_df.to_dict('records'),
            sort_action="native",
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '8px'
            },
            style_data_conditional=style_conditions
        )
        
        # Data quality indicator colors
        quality_colors = {
            'high': 'green',
            'medium': 'orange',
            'low': 'red',
            'none': 'red',
            'unknown': 'gray'
        }
        
        # Note about data estimation if applicable
        estimation_note = None
        if estimation_method is not None:
            method_text = "price levels" if estimation_method == 'price_level' else "yearly prices"
            estimation_note = html.Div([
                html.Span(
                    f"Note: Growth values are estimated from {method_text} due to limited historical data",
                    style={
                        'backgroundColor': 'rgba(255, 255, 200, 0.8)',
                        'color': 'black',
                        'padding': '4px 8px',
                        'borderRadius': '4px',
                        'fontSize': '12px',
                        'marginTop': '5px',
                        'display': 'inline-block'
                    }
                )
            ])
        
        # Build the complete component
        header_components = [
            html.H6("Emerging Segments with Accelerating Growth", className="mb-2")
        ]
        
        if estimation_note:
            header_components.append(estimation_note)
        
        return html.Div([
            html.Div(header_components, style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '5px'}),
            html.P("Segments showing strong recent growth and positive acceleration compared to previous periods."),
            table
        ], className="table-container")
    else:
        return html.Div([
            html.H5("Limited Data Available"),
            html.P("The selected filters don't provide enough data for emerging segments analysis.")
        ])


def create_outperformers_table(outperformers_df):
    """
    Create a table visualization for consistent outperformers
    
    Args:
        outperformers_df (pd.DataFrame): DataFrame with consistent outperformers
        
    Returns:
        dash component: Dash data table with consistent outperformers
    """
    # Select columns to display
    display_columns = []
    
    # Core columns that should always be included if available
    core_columns = ['property_type_en', 'rooms_en', 'reg_type_en', 'area_name_en', 
                   'median_price_sqft', 'transaction_count']
    
    # Performance metrics in order of preference
    perf_metrics = ['outperformance_ratio', 'outperformance_count', 'available_periods']
    
    # Add core columns if available
    for col in core_columns:
        if col in outperformers_df.columns:
            display_columns.append(col)
    
    # Add performance metrics if available
    for metric in perf_metrics:
        if metric in outperformers_df.columns:
            display_columns.append(metric)
    
    # Prepare display DataFrame
    if len(display_columns) > 0:
        display_df = outperformers_df[display_columns].copy()
        
        # Sort by outperformance ratio if available
        if 'outperformance_ratio' in display_df.columns:
            display_df = display_df.sort_values('outperformance_ratio', ascending=False)
        
        # Rename columns for display
        column_map = {
            'property_type_en': 'Property Type',
            'rooms_en': 'Room Config',
            'reg_type_en': 'Registration Type',
            'area_name_en': 'Area',
            'median_price_sqft': 'Price (AED/sqft)',
            'transaction_count': 'Transactions',
            'outperformance_ratio': 'Consistency Score',
            'outperformance_count': 'Periods Outperforming',
            'available_periods': 'Total Periods'
        }
        
        display_df = display_df.rename(columns={col: column_map.get(col, col) for col in display_df.columns})
        
        # Format numeric columns
        for col in display_df.columns:
            if col == 'Price (AED/sqft)' and col in display_df.columns:
                display_df[col] = display_df[col].round(0).astype(int)
            elif col == 'Consistency Score' and col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(0).astype(int)
        
        # Create the table
        table = dash_table.DataTable(
            id='outperformers-table',
            columns=[
                {"name": i, "id": i} for i in display_df.columns
            ],
            data=display_df.to_dict('records'),
            sort_action="native",
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '8px'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Consistency Score', 'filter_query': '{Consistency Score} >= 90'},
                    'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'Consistency Score', 'filter_query': '{Consistency Score} >= 75 && {Consistency Score} < 90'},
                    'backgroundColor': 'rgba(255, 255, 0, 0.2)'
                }
            ]
        )
        
        return html.Div([
            html.H6("Consistent Market Outperformers"),
            html.P("Segments that consistently outperform the market average across multiple time periods."),
            table
        ])
    else:
        return html.Div("No consistent outperformers data available")

def create_premium_discount_chart(df, premium_columns):
    """
    Create a visualization for segment premium/discount analysis
    
    Args:
        df (pd.DataFrame): DataFrame with premium/discount data
        premium_columns (list): List of premium column names
        
    Returns:
        go.Figure: Plotly figure with premium/discount chart
    """
    if len(df) == 0 or not premium_columns:
        return create_default_figure("Segment Premium/Discount", "No premium/discount data available")
    
    # Select the first premium column if multiple are available
    premium_col = premium_columns[0]
    
    # Group by property type and calculate average premium
    if 'property_type_en' in df.columns:
        property_premium = df.groupby('property_type_en')[premium_col].mean().reset_index()
        property_premium = property_premium.sort_values(premium_col, ascending=False)
        
        # Create bar chart for property type premium
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=property_premium['property_type_en'],
            y=property_premium[premium_col],
            name='Premium/Discount (%)',
            marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in property_premium[premium_col]]
        ))
        
        # Add horizontal line at 0
        fig.add_shape(
            type='line',
            y0=0,
            y1=0,
            x0=-0.5,
            x1=len(property_premium) - 0.5,
            line=dict(color='black', width=1, dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title="Property Type Premium/Discount Analysis",
            xaxis_title="Property Type",
            yaxis_title="Premium/Discount (%)",
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            )
        )
        
        return fig
    
    # If no property_type_en, try area_name_en
    elif 'area_name_en' in df.columns:
        area_premium = df.groupby('area_name_en')[premium_col].mean().reset_index()
        area_premium = area_premium.sort_values(premium_col, ascending=False).head(15)  # Top 15 areas
        
        # Create bar chart for area premium
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=area_premium['area_name_en'],
            y=area_premium[premium_col],
            name='Premium/Discount (%)',
            marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in area_premium[premium_col]]
        ))
        
        # Add horizontal line at 0
        fig.add_shape(
            type='line',
            y0=0,
            y1=0,
            x0=-0.5,
            x1=len(area_premium) - 0.5,
            line=dict(color='black', width=1, dash='dash')
        )
        
        # Update layout
        fig.update_layout(
            title="Area Premium/Discount Analysis",
            xaxis_title="Area",
            yaxis_title="Premium/Discount (%)",
            xaxis=dict(
                tickangle=45
            ),
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            )
        )
        
        return fig
    
    # Default case
    else:
        return create_default_figure("Segment Premium/Discount", "Premium/discount data is not properly structured")

def create_relative_performance_chart(df, rel_column):
    """
    Create a visualization for relative performance analysis
    
    Args:
        df (pd.DataFrame): DataFrame with relative performance data
        rel_column (str): Name of the relative performance column
        
    Returns:
        go.Figure: Plotly figure with relative performance chart
    """
    if len(df) == 0 or rel_column not in df.columns:
        return create_default_figure("Relative Performance", "No relative performance data available")
    
    # Create scatter plot of relative performance vs price
    if 'median_price_sqft' in df.columns and 'area_name_en' in df.columns:
        # Sort by relative performance
        df_sorted = df.sort_values(rel_column, ascending=False)
        
        # Create scatter plot
        fig = px.scatter(
            df_sorted,
            x='median_price_sqft',
            y=rel_column,
            color=rel_column,
            size='transaction_count' if 'transaction_count' in df.columns else None,
            hover_name='area_name_en',
            color_continuous_scale='RdYlGn',
            labels={
                'median_price_sqft': 'Median Price (AED/sqft)',
                rel_column: 'Performance vs Market Avg (%)',
                'transaction_count': 'Transaction Volume'
            }
        )
        
        # Add horizontal line at 0 (market average)
        fig.add_shape(
            type='line',
            y0=0,
            y1=0,
            x0=df_sorted['median_price_sqft'].min(),
            x1=df_sorted['median_price_sqft'].max(),
            line=dict(color='black', width=1, dash='dash')
        )
        
        # Add quadrant labels
        median_price = df_sorted['median_price_sqft'].median()
        max_rel = df_sorted[rel_column].max()
        min_rel = df_sorted[rel_column].min()
        mid_rel = (max_rel + min_rel) / 2
        
        # Add quadrant annotations
        fig.add_annotation(
            x=median_price * 0.7,
            y=mid_rel * 0.7 if mid_rel > 0 else mid_rel * 0.3,
            text="Value Outperformers<br>(Low Price, High Growth)",
            showarrow=False,
            bgcolor="rgba(0,255,0,0.1)"
        )
        
        fig.add_annotation(
            x=median_price * 1.3,
            y=mid_rel * 0.7 if mid_rel > 0 else mid_rel * 0.3,
            text="Premium Outperformers<br>(High Price, High Growth)",
            showarrow=False,
            bgcolor="rgba(255,255,0,0.1)"
        )
        
        # Update layout
        fig.update_layout(
            title="Relative Performance Analysis",
            xaxis_title="Median Price (AED/sqft)",
            yaxis_title="Performance vs Market Average (%)",
            yaxis=dict(
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black'
            )
        )
        
        return fig
    
    # If no median_price_sqft, create bar chart of top performers
    else:
        # Group by area or property type
        group_by_col = next((col for col in ['area_name_en', 'property_type_en', 'rooms_en'] if col in df.columns), None)
        
        if group_by_col:
            grouped_df = df.groupby(group_by_col)[rel_column].mean().reset_index()
            grouped_df = grouped_df.sort_values(rel_column, ascending=False).head(15)  # Top 15
            
            # Create bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=grouped_df[group_by_col],
                y=grouped_df[rel_column],
                marker_color=['#2ca02c' if x >= 0 else '#d62728' for x in grouped_df[rel_column]]
            ))
            
            # Add horizontal line at 0 (market average)
            fig.add_shape(
                type='line',
                y0=0,
                y1=0,
                x0=-0.5,
                x1=len(grouped_df) - 0.5,
                line=dict(color='black', width=1, dash='dash')
            )
            
            # Update layout
            fig.update_layout(
                title="Relative Performance vs Market Average",
                xaxis_title=group_by_col.replace('_en', '').title(),
                yaxis_title="Performance vs Market Average (%)",
                xaxis=dict(
                    tickangle=45
                ),
                yaxis=dict(
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='black'
                )
            )
            
            return fig
        else:
            return create_default_figure("Relative Performance", "Relative performance data is not properly structured")

def create_statistical_significance_chart(df, time_horizon):
    """
    Create a visualization for statistical significance analysis
    
    Args:
        df (pd.DataFrame): DataFrame with statistical significance data
        time_horizon (str): Time horizon to analyze
        
    Returns:
        go.Figure: Plotly figure with statistical significance chart
    """
    # Check for required columns
    growth_col = time_horizon
    ci_lower = f'{growth_col}_ci_lower'
    ci_upper = f'{growth_col}_ci_upper'
    significant = f'{growth_col}_significant'
    
    if not all(col in df.columns for col in [growth_col, ci_lower, ci_upper, significant]):
        return create_default_figure("Statistical Significance", "Statistical significance data not available")
    
    # Focus on top areas by transaction count
    if 'area_name_en' in df.columns and 'transaction_count' in df.columns:
        # Group by area
        area_df = df.groupby('area_name_en').agg({
            growth_col: 'mean',
            ci_lower: 'mean',
            ci_upper: 'mean',
            significant: 'mean',
            'transaction_count': 'sum'
        }).reset_index()
        
        # Sort by growth and take top 15
        top_areas = area_df.sort_values(growth_col, ascending=False).head(15)
        
        # Create the figure
        fig = go.Figure()
        
        # Add market average reference line
        market_avg = df[growth_col].median()
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=top_areas['area_name_en'],
            y=top_areas[growth_col],
            error_y=dict(
                type='data',
                symmetric=False,
                array=top_areas[ci_upper] - top_areas[growth_col],
                arrayminus=top_areas[growth_col] - top_areas[ci_lower]
            ),
            mode='markers',
            marker=dict(
                color=['#2ca02c' if x else '#d62728' for x in top_areas[significant]],
                size=10
            ),
            name='Growth Rate'
        ))
        
        # Add market average line
        fig.add_shape(
            type='line',
            y0=market_avg,
            y1=market_avg,
            x0=-0.5,
            x1=len(top_areas) - 0.5,
            line=dict(color='black', width=1, dash='dash')
        )
        
        # Add annotation for market average
        fig.add_annotation(
            x=len(top_areas) - 1,
            y=market_avg,
            text=f"Market Average: {market_avg:.1f}%",
            showarrow=False,
            yshift=10
        )
        
        # Update layout
        fig.update_layout(
            title="Statistical Significance of Growth Rates",
            xaxis_title="Area",
            yaxis_title="Growth Rate (%)",
            xaxis=dict(
                tickangle=45
            ),
            showlegend=False
        )
        
        return fig
    else:
        return create_default_figure("Statistical Significance", "Data not properly structured for significance analysis")

def calculate_simple_premium(df):
    """
    Calculate a simplified version of premium/discount for each segment
    This is a workaround for the error in the original function
    
    Args:
        df (pd.DataFrame): DataFrame to calculate premiums for
        
    Returns:
        pd.DataFrame: DataFrame with added premium columns
    """
    result_df = df.copy()
    
    # Calculate property type premium only (simplified)
    if 'property_type_en' in df.columns and 'median_price_sqft' in df.columns:
        # Get the overall median price
        overall_median = df['median_price_sqft'].median()
        
        # Calculate property type averages
        property_type_avg = df.groupby('property_type_en')['median_price_sqft'].median()
        
        # Add premium column
        result_df['property_type_premium'] = 0
        
        # Calculate premium for each property type
        for prop_type, avg_price in property_type_avg.items():
            if avg_price > 0:
                premium = ((avg_price / overall_median) - 1) * 100
                result_df.loc[result_df['property_type_en'] == prop_type, 'property_type_premium'] = premium
    
    return result_df

# Helper function for creating default figures
def create_default_figure(title="No Data", message="No data available for the selected filters"):
    """Create a default figure with a message when data is not available"""
    fig = go.Figure()
    
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 18
                }
            }
        ]
    )
    
    return fig

def safe_import(module_name, function_name, fallback_value=None):
    """Safely import a function from a module with fallback"""
    try:
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {function_name} from {module_name}: {e}")
        return lambda *args, **kwargs: fallback_value