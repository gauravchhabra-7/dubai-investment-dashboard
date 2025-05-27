import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from dash import html, dash_table
import warnings
warnings.filterwarnings('ignore')

def prepare_project_data(df):
    """
    Clean and enhance the project dataset for comprehensive analysis
    
    Args:
        df (pd.DataFrame): Raw project dataset
        
    Returns:
        pd.DataFrame: Cleaned and enhanced dataset with additional metrics
    """
    print("Preparing project data for analysis...")
    
    # Create a copy to avoid modifying original
    enhanced_df = df.copy()
    
    # 1. Handle column name standardization
    # Use area_name_en_x as primary and rename to area_name_en
    if 'area_name_en_x' in enhanced_df.columns:
        enhanced_df['area_name_en'] = enhanced_df['area_name_en_x']
        # Drop the duplicate area columns
        enhanced_df = enhanced_df.drop(['area_name_en_x'], axis=1, errors='ignore')
        if 'area_name_en_y' in enhanced_df.columns:
            enhanced_df = enhanced_df.drop(['area_name_en_y'], axis=1, errors='ignore')
    
    # 2. Data type conversions
    # Convert date columns from string to datetime
    date_columns = ['first_transaction_date', 'last_transaction_date', 'completion_date', 'project_start_date']
    for col in date_columns:
        if col in enhanced_df.columns:
            enhanced_df[col] = pd.to_datetime(enhanced_df[col], errors='coerce')
    
    # Convert string boolean to actual boolean
    if 'single_transaction' in enhanced_df.columns:
        enhanced_df['single_transaction'] = enhanced_df['single_transaction'].map({
            'True': True, 'False': False, True: True, False: False
        })
    
    # 3. Handle missing data
    # Fill missing developer names
    if 'developer_name' in enhanced_df.columns:
        enhanced_df['developer_name'] = enhanced_df['developer_name'].fillna('Unknown Developer')
    
    # Fill missing area names
    if 'area_name_en' in enhanced_df.columns:
        enhanced_df['area_name_en'] = enhanced_df['area_name_en'].fillna('Unknown Area')
    
    # 4. Calculate additional metrics
    enhanced_df = calculate_peer_volatility(enhanced_df)
    enhanced_df = calculate_market_outperformance(enhanced_df)
    
    # 5. Add data quality indicators (objective, not prescriptive)
    enhanced_df['has_sufficient_transactions'] = enhanced_df['transaction_count'] >= 10
    enhanced_df['is_long_term_data'] = enhanced_df['duration_years'] >= 1.0
    enhanced_df['data_points_available'] = enhanced_df['transaction_count']
    
    print(f"Data preparation complete. Enhanced dataset has {len(enhanced_df)} project segments.")
    print(f"Unique areas: {enhanced_df['area_name_en'].nunique()}")
    print(f"Unique developers: {enhanced_df['developer_name'].nunique()}")
    print(f"Property type combinations: {enhanced_df.groupby(['property_type_en', 'rooms_en']).size().count()}")
    
    return enhanced_df

def calculate_peer_volatility(df):
    """
    Calculate volatility based on peer group performance (same property type + room type)
    
    Args:
        df (pd.DataFrame): Project dataset
        
    Returns:
        pd.DataFrame: Dataset with peer_volatility column added
    """
    df = df.copy()
    df['peer_volatility'] = np.nan
    
    # Group by property type and room type to find peers
    peer_groups = df.groupby(['property_type_en', 'rooms_en'])
    
    for (prop_type, room_type), group in peer_groups:
        # Calculate standard deviation of CAGR within peer group
        valid_cagrs = group['price_sqft_cagr'].dropna()
        
        if len(valid_cagrs) >= 3:  # Need at least 3 projects for meaningful volatility
            volatility = valid_cagrs.std()
            # Assign volatility to all projects in this peer group
            mask = (df['property_type_en'] == prop_type) & (df['rooms_en'] == room_type)
            df.loc[mask, 'peer_volatility'] = volatility
        else:
            # For peer groups with too few projects, use market-wide volatility
            market_volatility = df['price_sqft_cagr'].std()
            mask = (df['property_type_en'] == prop_type) & (df['rooms_en'] == room_type)
            df.loc[mask, 'peer_volatility'] = market_volatility
    
    return df

def calculate_market_outperformance(df):
    """
    Calculate how each project performs vs transaction-weighted average of peers
    
    Args:
        df (pd.DataFrame): Project dataset
        
    Returns:
        pd.DataFrame: Dataset with market_outperformance column added
    """
    df = df.copy()
    df['market_outperformance'] = np.nan
    df['peer_weighted_average'] = np.nan
    
    # Group by property type and room type for peer comparison
    peer_groups = df.groupby(['property_type_en', 'rooms_en'])
    
    for (prop_type, room_type), group in peer_groups:
        # Calculate transaction-weighted average CAGR for peer group
        valid_data = group.dropna(subset=['price_sqft_cagr', 'transaction_count'])
        
        if len(valid_data) >= 2:  # Need at least 2 projects for comparison
            # Transaction-weighted average
            total_weighted_cagr = (valid_data['price_sqft_cagr'] * valid_data['transaction_count']).sum()
            total_weights = valid_data['transaction_count'].sum()
            
            if total_weights > 0:
                weighted_avg_cagr = total_weighted_cagr / total_weights
                
                # Calculate outperformance for each project in this peer group
                mask = (df['property_type_en'] == prop_type) & (df['rooms_en'] == room_type)
                df.loc[mask, 'peer_weighted_average'] = weighted_avg_cagr
                df.loc[mask, 'market_outperformance'] = df.loc[mask, 'price_sqft_cagr'] - weighted_avg_cagr
    
    return df

def filter_project_data(df, property_type=None, room_type=None, area=None, developer=None, 
                       min_duration=None, max_duration=None, min_transactions=None):
    """
    Apply filters to the project dataset
    
    Args:
        df (pd.DataFrame): Project dataset
        property_type (str): Filter by property type
        room_type (str): Filter by room configuration  
        area (str): Filter by area
        developer (str): Filter by developer
        min_duration (float): Minimum duration in years
        max_duration (float): Maximum duration in years
        min_transactions (int): Minimum transaction count
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filtered_df = df.copy()
    
    # Apply filters only if values are provided and not 'All'
    if property_type and property_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type_en'] == property_type]
    
    if room_type and room_type != 'All':
        filtered_df = filtered_df[filtered_df['rooms_en'] == room_type]
    
    if area and area != 'All':
        filtered_df = filtered_df[filtered_df['area_name_en'] == area]
    
    if developer and developer != 'All':
        filtered_df = filtered_df[filtered_df['developer_name'] == developer]
    
    if min_duration is not None:
        filtered_df = filtered_df[filtered_df['duration_years'] >= min_duration]
    
    if max_duration is not None:
        filtered_df = filtered_df[filtered_df['duration_years'] <= max_duration]
    
    if min_transactions is not None:
        filtered_df = filtered_df[filtered_df['transaction_count'] >= min_transactions]
    
    return filtered_df

def validate_and_clean_data(df):
    """
    Validate and clean data to remove extreme outliers and errors
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None or len(df) == 0:
        return df
    
    # Create a copy
    clean_df = df.copy()
    
    # Remove rows with missing essential data
    essential_columns = ['price_sqft_cagr', 'project_name_en']
    clean_df = clean_df.dropna(subset=[col for col in essential_columns if col in clean_df.columns])
    
    # Filter CAGR outliers (keep reasonable range: -50% to +50%)
    if 'price_sqft_cagr' in clean_df.columns:
        clean_df = clean_df[
            (clean_df['price_sqft_cagr'] >= -50) & 
            (clean_df['price_sqft_cagr'] <= 50)
        ]
    
    # Ensure transaction count is reasonable (1 to 10000)
    if 'transaction_count' in clean_df.columns:
        clean_df = clean_df[
            (clean_df['transaction_count'] >= 1) & 
            (clean_df['transaction_count'] <= 10000)
        ]
    
    # Clean text fields - remove any pandas Series artifacts
    text_columns = ['project_name_en', 'area_name_en', 'developer_name']
    for col in text_columns:
        if col in clean_df.columns:
            # Convert to string and clean
            clean_df[col] = clean_df[col].astype(str).str.strip()
            # Remove any entries that look like pandas Series representations
            clean_df = clean_df[~clean_df[col].str.contains('Name:|dtype:', na=False)]
    
    return clean_df

def create_individual_project_analysis(df, title="Individual Project Performance Analysis"):
    """
    Create horizontal bar chart for top individual project performance
    
    Args:
        df (pd.DataFrame): Filtered project dataset
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure with top project performance
    """
    if df is None or len(df) == 0:
        return create_empty_figure("No projects available for the selected filters")
    
    # Clean and validate data first
    clean_df = validate_and_clean_data(df)
    
    if len(clean_df) == 0:
        return create_empty_figure("No valid data after cleaning outliers")
    
    # Filter for quality data (minimum 5 transactions for reliability)
    quality_df = clean_df[clean_df['transaction_count'] >= 5].copy()
    
    if len(quality_df) < 5:
        # If not enough quality data, use all cleaned data but with warning
        quality_df = clean_df.copy()
        data_quality_note = " (Limited transaction data)"
    else:
        data_quality_note = ""
    
    # Sort by CAGR and take top 15
    top_projects = quality_df.nlargest(15, 'price_sqft_cagr').iloc[::-1].copy()
    
    # Create clean project labels
    def create_project_label(row):
        """Create clean project label"""
        try:
            project_name = str(row['project_name_en']).strip()
            area_name = str(row['area_name_en']).strip()
            
            # Handle long names
            if len(project_name) > 30:
                project_name = project_name[:27] + "..."
            if len(area_name) > 20:
                area_name = area_name[:17] + "..."
                
            return f"{project_name} ({area_name})"
        except:
            return "Unknown Project"
    
    top_projects['project_label'] = top_projects.apply(create_project_label, axis=1)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Create color array based on transaction count
    transaction_counts = top_projects['transaction_count'].fillna(0)
    
    fig.add_trace(go.Bar(
        y=top_projects['project_label'],
        x=top_projects['price_sqft_cagr'],
        orientation='h',
        marker=dict(
            color=transaction_counts,
            colorscale='Viridis',
            colorbar=dict(
                title="Transaction<br>Count",
                titleside="right"
            ),
            line=dict(color='rgba(50,50,50,0.8)', width=0.5)
        ),
        text=[f"{val:.1f}%" for val in top_projects['price_sqft_cagr']],
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate=(
            '<b>%{y}</b><br>' +
            'CAGR: %{x:.1f}%<br>' +
            'Transactions: %{marker.color}<br>' +
            'Duration: %{customdata[0]:.1f} years<br>' +
            'Market Outperformance: %{customdata[1]:.1f}%<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack((
            top_projects['duration_years'].fillna(0),
            top_projects['market_outperformance'].fillna(0)
        ))
    ))
    
    # Add vertical line at 0%
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="0% Return")
    
    # Set reasonable x-axis range
    max_cagr = top_projects['price_sqft_cagr'].max()
    min_cagr = top_projects['price_sqft_cagr'].min()
    x_range = [min(min_cagr - 2, -5), max(max_cagr + 2, 25)]
    
    # Update layout
    fig.update_layout(
        title=f"{title}{data_quality_note}",
        xaxis_title="Annualized Return (CAGR %)",
        yaxis_title="Project",
        height=max(500, len(top_projects) * 35 + 150),
        showlegend=False,
        autosize=True,     
        width=None,  
        yaxis=dict(
            tickmode='linear',
            automargin=True
        ),
        xaxis=dict(
            tickformat='.1f',
            ticksuffix='%',
            range=x_range
        ),
        margin=dict(l=250, r=120, t=80, b=120)
    )
    
    # Add subtitle with data info
    total_projects = len(clean_df)
    fig.add_annotation(
        text=f"Showing top 15 of {total_projects} projects (ranked by CAGR)",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor="center"
    )
    
    return fig

def create_area_performance_comparison(df, title="Area Performance Comparison"):
    """
    Create horizontal bar chart for area performance comparison
    
    Args:
        df (pd.DataFrame): Filtered project dataset
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure with area comparison
    """
    if df is None or len(df) == 0:
        return create_empty_figure("No data available for area comparison")
    
    # Clean and validate data first
    clean_df = validate_and_clean_data(df)
    
    if len(clean_df) == 0:
        return create_empty_figure("No valid data after cleaning")
    
    # Group by area and calculate key metrics
    try:
        area_stats = clean_df.groupby('area_name_en').agg({
            'price_sqft_cagr': ['mean', 'count', 'std'],
            'market_outperformance': 'mean',
            'transaction_count': 'sum',
            'duration_years': 'mean'
        }).round(2)
        
        # Flatten column names properly
        area_stats.columns = ['avg_cagr', 'project_count', 'cagr_volatility', 
                             'avg_outperformance', 'total_transactions', 'avg_duration']
        area_stats = area_stats.reset_index()
        
        # Filter areas with at least 3 projects for meaningful comparison
        area_stats = area_stats[area_stats['project_count'] >= 3].copy()
        
        if len(area_stats) == 0:
            return create_empty_figure("Insufficient data for area comparison (need ≥3 projects per area)")
        
        # Sort by average CAGR and take top 15
        top_areas = area_stats.nlargest(15, 'avg_cagr').iloc[::-1].copy()
        
        # Create clean area labels
        def create_area_label(row):
            """Create clean area label"""
            try:
                area_name = str(row['area_name_en']).strip()
                project_count = int(row['project_count'])
                
                # Handle long area names
                if len(area_name) > 25:
                    area_name = area_name[:22] + "..."
                    
                return f"{area_name} ({project_count} projects)"
            except:
                return "Unknown Area"
        
        top_areas['area_label'] = top_areas.apply(create_area_label, axis=1)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_areas['area_label'],
            x=top_areas['avg_cagr'],
            orientation='h',
            marker=dict(
                color=top_areas['total_transactions'],
                colorscale='Blues',
                colorbar=dict(
                    title="Total<br>Transactions",
                    titleside="right"
                ),
                line=dict(color='rgba(50,50,50,0.8)', width=0.5)
            ),
            text=[f"{val:.1f}%" for val in top_areas['avg_cagr']],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Average CAGR: %{x:.1f}%<br>' +
                'Total Transactions: %{marker.color}<br>' +
                'Project Count: %{customdata[0]}<br>' +
                'Volatility: %{customdata[1]:.1f}%<br>' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((
                top_areas['project_count'],
                top_areas['cagr_volatility'].fillna(0)
            ))
        ))
        
        # Add vertical line at 0%
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="0% Return")
        
        # Set reasonable x-axis range
        max_cagr = top_areas['avg_cagr'].max()
        min_cagr = top_areas['avg_cagr'].min()
        x_range = [min(min_cagr - 1, -5), max(max_cagr + 1, 20)]
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Average CAGR (%)",
            yaxis_title="Area",
            height=max(500, len(top_areas) * 35 + 150),
            showlegend=False,
            autosize=True,    
            width=None,  
            yaxis=dict(
                tickmode='linear',
                automargin=True
            ),
            xaxis=dict(
                tickformat='.1f',
                ticksuffix='%',
                range=x_range
            ),
            margin=dict(l=300, r=120, t=80, b=120)
        )
        
        # Add subtitle with data info
        total_areas = len(area_stats)
        fig.add_annotation(
            text=f"Showing top 15 of {total_areas} areas with ≥3 projects each",
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in area comparison: {e}")
        return create_empty_figure(f"Error creating area comparison: {str(e)}")

def create_developer_track_record_analysis(df, title="Developer Track Record Analysis"):
    """
    Create horizontal bar chart for developer track record comparison
    
    Args:
        df (pd.DataFrame): Filtered project dataset
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure with developer comparison
    """
    if df is None or len(df) == 0:
        return create_empty_figure("No data available for developer comparison")
    
    # Clean and validate data first
    clean_df = validate_and_clean_data(df)
    
    if len(clean_df) == 0:
        return create_empty_figure("No valid data after cleaning")
    
    try:
        # Group by developer and calculate portfolio metrics
        dev_stats = clean_df.groupby('developer_name').agg({
            'price_sqft_cagr': ['mean', 'count', 'std', 'min', 'max'],
            'market_outperformance': 'mean',
            'transaction_count': 'sum',
            'duration_years': 'mean'
        }).round(2)
        
        # Flatten column names properly
        dev_stats.columns = ['avg_cagr', 'project_count', 'cagr_volatility', 'min_cagr', 'max_cagr',
                            'avg_outperformance', 'total_transactions', 'avg_duration']
        dev_stats = dev_stats.reset_index()
        
        # Filter developers with at least 2 projects
        dev_stats = dev_stats[dev_stats['project_count'] >= 2].copy()
        
        if len(dev_stats) == 0:
            return create_empty_figure("Insufficient data for developer comparison (need ≥2 projects per developer)")
        
        # Calculate consistency score (lower volatility = higher consistency)
        max_volatility = dev_stats['cagr_volatility'].max()
        if max_volatility > 0:
            dev_stats['consistency_score'] = 100 * (1 - dev_stats['cagr_volatility'] / max_volatility)
        else:
            dev_stats['consistency_score'] = 100
        dev_stats['consistency_score'] = dev_stats['consistency_score'].fillna(100)
        
        # Sort by average CAGR and take top 15
        top_developers = dev_stats.nlargest(15, 'avg_cagr').iloc[::-1].copy()
        
        # Create clean developer labels
        def create_developer_label(row):
            """Create clean developer label"""
            try:
                dev_name = str(row['developer_name']).strip()
                project_count = int(row['project_count'])
                
                # Handle long developer names
                if len(dev_name) > 25:
                    dev_name = dev_name[:22] + "..."
                    
                return f"{dev_name} ({project_count} projects)"
            except:
                return "Unknown Developer"
        
        top_developers['dev_label'] = top_developers.apply(create_developer_label, axis=1)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_developers['dev_label'],
            x=top_developers['avg_cagr'],
            orientation='h',
            marker=dict(
                color=top_developers['consistency_score'],
                colorscale='RdYlGn',
                colorbar=dict(
                    title="Consistency<br>Score",
                    titleside="right"
                ),
                line=dict(color='rgba(50,50,50,0.8)', width=0.5)
            ),
            text=[f"{val:.1f}%" for val in top_developers['avg_cagr']],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Average CAGR: %{x:.1f}%<br>' +
                'Consistency Score: %{marker.color:.0f}<br>' +
                'Best Project: %{customdata[0]:.1f}%<br>' +
                'Worst Project: %{customdata[1]:.1f}%<br>' +
                'Total Transactions: %{customdata[2]}<br>' +
                '<extra></extra>'
            ),
            customdata=np.column_stack((
                top_developers['max_cagr'],
                top_developers['min_cagr'],
                top_developers['total_transactions']
            ))
        ))
        
        # Add vertical line at 0%
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="0% Return")
        
        # Set reasonable x-axis range
        max_cagr = top_developers['avg_cagr'].max()
        min_cagr = top_developers['avg_cagr'].min()
        x_range = [min(min_cagr - 1, -5), max(max_cagr + 1, 20)]
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Average Portfolio CAGR (%)",
            yaxis_title="Developer",
            height=max(500, len(top_developers) * 35 + 150),
            showlegend=False,
            yaxis=dict(
                tickmode='linear',
                automargin=True
            ),
            xaxis=dict(
                tickformat='.1f',
                ticksuffix='%',
                range=x_range
            ),
            margin=dict(l=350, r=120, t=80, b=120)
        )
        
        # Add subtitle with data info
        total_developers = len(dev_stats)
        fig.add_annotation(
            text=f"Showing top 15 of {total_developers} developers with ≥2 projects each",
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=12, color="gray"),
            xanchor="center"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in developer comparison: {e}")
        return create_empty_figure(f"Error creating developer comparison: {str(e)}")

def create_project_performance_table(df, max_rows=20):
    """
    Create a detailed data table showing project performance metrics
    
    Args:
        df (pd.DataFrame): Filtered project dataset
        max_rows (int): Maximum number of rows to display
        
    Returns:
        dash_table.DataTable: Interactive data table
    """
    if df is None or len(df) == 0:
        return html.Div("No data available for table display")
    
    # Clean data first
    clean_df = validate_and_clean_data(df)
    
    if len(clean_df) == 0:
        return html.Div("No valid data after cleaning")
    
    # Select key columns for display
    display_columns = [
        'project_name_en', 'area_name_en', 'developer_name', 'property_type_en', 'rooms_en',
        'price_sqft_cagr', 'price_sqft_percentage_growth', 'market_outperformance',
        'transaction_count', 'duration_years'
    ]
    
    # Filter for available columns
    available_columns = [col for col in display_columns if col in clean_df.columns]
    table_df = clean_df[available_columns].copy()
    
    # Sort by CAGR descending
    if 'price_sqft_cagr' in table_df.columns:
        table_df = table_df.sort_values('price_sqft_cagr', ascending=False)
    
    # Limit rows
    table_df = table_df.head(max_rows)
    
    # Round numeric columns
    numeric_columns = ['price_sqft_cagr', 'price_sqft_percentage_growth', 'market_outperformance', 'duration_years']
    for col in numeric_columns:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(2)
    
    # Create column definitions
    columns = []
    for col in table_df.columns:
        column_name = col.replace('_', ' ').replace('en', '').title()
        column_name = column_name.replace('Sqft', 'sqft').replace('Cagr', 'CAGR')
        
        columns.append({
            "name": column_name,
            "id": col,
            "type": "numeric" if col in numeric_columns + ['transaction_count'] else "text"
        })
    
    # Create the data table
    table = dash_table.DataTable(
        columns=columns,
        data=table_df.to_dict('records'),
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{price_sqft_cagr} > 10',
                    'column_id': 'price_sqft_cagr'
                },
                'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{price_sqft_cagr} < 0',
                    'column_id': 'price_sqft_cagr'
                },
                'backgroundColor': 'rgba(255, 0, 0, 0.2)'
            }
        ]
    )
    
    return table

def create_empty_figure(message="No data available"):
    """
    Create an empty figure with a message when no data is available
    
    Args:
        message (str): Message to display
        
    Returns:
        go.Figure: Empty plotly figure with message
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=400,
        plot_bgcolor='white'
    )
    return fig

def calculate_data_quality(df):
    """
    Calculate data quality indicator based on transaction counts and duration
    
    Args:
        df (pd.DataFrame): Project dataset
        
    Returns:
        str: Data quality level ('high', 'medium', 'low')
    """
    if df is None or len(df) == 0:
        return 'none'
    
    # Calculate percentage of projects with sufficient transactions and duration
    sufficient_tx = (df['transaction_count'] >= 20).mean() * 100
    sufficient_duration = (df['duration_years'] >= 1).mean() * 100
    
    # Combined quality score
    quality_score = (sufficient_tx + sufficient_duration) / 2
    
    if quality_score >= 70:
        return 'high'
    elif quality_score >= 40:
        return 'medium'
    else:
        return 'low'

def calculate_coverage_percentage(df):
    """
    Calculate data coverage percentage based on available metrics
    
    Args:
        df (pd.DataFrame): Project dataset
        
    Returns:
        float: Coverage percentage
    """
    if df is None or len(df) == 0:
        return 0.0
    
    # Key metrics that should be available
    key_metrics = ['price_sqft_cagr', 'market_outperformance', 'peer_volatility']
    
    coverage_scores = []
    for metric in key_metrics:
        if metric in df.columns:
            coverage = df[metric].notna().mean() * 100
            coverage_scores.append(coverage)
    
    return np.mean(coverage_scores) if coverage_scores else 0.0

def prepare_insights_metadata(df, filters=None, peer_group_info=None):
    """
    Prepare metadata dictionary for the insights system
    
    Args:
        df (pd.DataFrame): Filtered project dataset
        filters (dict): Applied filters
        peer_group_info (dict): Information about peer groups
        
    Returns:
        dict: Metadata for insights system
    """
    if df is None or len(df) == 0:
        return {
            'data_quality': 'none',
            'coverage_pct': 0.0,
            'estimation_method': None,
            'total_projects': 0
        }
    
    metadata = {
        'data_quality': calculate_data_quality(df),
        'coverage_pct': calculate_coverage_percentage(df),
        'estimation_method': None,  # We use actual transaction data, no estimation
        'total_projects': len(df),
        'total_transactions': df['transaction_count'].sum() if 'transaction_count' in df.columns else 0,
        'median_duration': df['duration_years'].median() if 'duration_years' in df.columns else 0
    }
    
    # Add peer group information if available
    if peer_group_info:
        metadata.update(peer_group_info)
    
    return metadata

def get_peer_group_info(df, property_type=None, room_type=None):
    """
    Get information about peer groups for a specific property+room combination
    
    Args:
        df (pd.DataFrame): Full project dataset
        property_type (str): Property type filter
        room_type (str): Room type filter
        
    Returns:
        dict: Peer group information
    """
    if df is None or len(df) == 0:
        return {'peer_group_size': 0}
    
    # If specific filters provided, get peer group size
    if property_type and property_type != 'All' and room_type and room_type != 'All':
        peer_group = df[
            (df['property_type_en'] == property_type) & 
            (df['rooms_en'] == room_type)
        ]
        return {
            'peer_group_size': len(peer_group),
            'peer_property_type': property_type,
            'peer_room_type': room_type
        }
    
    # Otherwise, get average peer group size across all combinations
    peer_groups = df.groupby(['property_type_en', 'rooms_en']).size()
    return {
        'avg_peer_group_size': peer_groups.mean(),
        'total_property_combinations': len(peer_groups)
    }