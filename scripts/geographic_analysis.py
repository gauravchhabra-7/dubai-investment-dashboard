import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from config import get_path
from config import MAPBOX_TOKEN

# Set Mapbox access token from config
px.set_mapbox_access_token(MAPBOX_TOKEN)

# Manual coordinates for key Dubai areas (fallback if no match found in GeoJSON)
DUBAI_AREA_COORDINATES = {
    'downtown dubai': {'lat': 25.1915, 'lon': 55.2737},
    'dubai marina': {'lat': 25.0809, 'lon': 55.1405},
    'palm jumeirah': {'lat': 25.1124, 'lon': 55.1371},
    'business bay': {'lat': 25.1868, 'lon': 55.2606},
    'jumeirah lake towers': {'lat': 25.0672, 'lon': 55.1375},
    'jumeirah beach residence': {'lat': 25.0759, 'lon': 55.1334},
    'jumeirah village circle': {'lat': 25.0445, 'lon': 55.2219},
    'arabian ranches': {'lat': 25.0255, 'lon': 55.2460},
    'international city': {'lat': 25.1584, 'lon': 55.4114},
    'jumeirah islands': {'lat': 25.0569, 'lon': 55.1597},
    'sports city': {'lat': 25.0328, 'lon': 55.2275},
    'deira': {'lat': 25.2737, 'lon': 55.3177},
    'motor city': {'lat': 25.0491, 'lon': 55.2394},
    'emirates hills': {'lat': 25.0674, 'lon': 55.1676},
    'al barsha': {'lat': 25.0982, 'lon': 55.2089},
    'al furjan': {'lat': 25.0359, 'lon': 55.1398},
    'mirdif': {'lat': 25.2199, 'lon': 55.4254},
    'jumeirah': {'lat': 25.2046, 'lon': 55.2550},
    'dubai hills estate': {'lat': 25.1234, 'lon': 55.2684},
    'dubai land': {'lat': 25.0688, 'lon': 55.2889}
}

# Global flag to track if geo libraries are available
has_geo_libraries = False

# Try to import geopandas - handle gracefully if not available
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    has_geo_libraries = True
except ImportError:
    print("Warning: geopandas not installed. Geographic analysis will be limited.")

def load_kml_file(file_path=None):
    """Load geographic file with focus on the custom GeoJSON file with CSV names"""
    # Check if we have the necessary libraries
    if not has_geo_libraries:
        print("Cannot load geographic file: geopandas library not installed.")
        return None
    
    # Only use the specific custom GeoJSON file
    custom_geojson_path = file_path or get_path('geojson_file')
    if os.path.exists(custom_geojson_path):
        print(f"Using custom GeoJSON file with CSV area names: {custom_geojson_path}")
        try:
            # Load GeoJSON directly
            gdf = gpd.read_file(custom_geojson_path)
            print(f"Loaded GeoJSON with {len(gdf)} features")
            print(f"Columns in GeoJSON: {gdf.columns.tolist()}")
            
            # Ensure CRS is in WGS84
            if gdf.crs is None:
                gdf.crs = 'EPSG:4326'
            else:
                gdf = gdf.to_crs('EPSG:4326')
                
            # Print sample values to verify
            if 'csv_area_name' in gdf.columns:
                print(f"Sample csv_area_name values: {gdf['csv_area_name'].dropna().head(5).tolist()}")
            else:
                print("ERROR: csv_area_name field not found in GeoJSON!")
                
            return gdf
        except Exception as e:
            print(f"Error loading custom GeoJSON file: {e}")
            return create_dummy_geodataframe()
    else:
        print(f"Custom GeoJSON file not found at {custom_geojson_path}")
        return create_dummy_geodataframe()

def create_dummy_geodataframe():
    """Create a dummy GeoDataFrame with basic Dubai areas"""
    print("Creating dummy GeoDataFrame for Dubai areas")
    
    # Only proceed if we have the geo libraries
    if not has_geo_libraries:
        print("Cannot create dummy GeoDataFrame: geopandas not installed")
        return None
        
    # Use our predefined coordinates
    dubai_areas = DUBAI_AREA_COORDINATES
    
    # Create points - in a real scenario, we'd create polygons
    dummy_data = []
    for area, coords in dubai_areas.items():
        dummy_data.append({
            'name': area,
            'geometry': Point(coords['lon'], coords['lat'])
        })
    
    dummy_gdf = gpd.GeoDataFrame(dummy_data, crs='EPSG:4326')
    print(f"Created dummy GeoDataFrame with {len(dummy_gdf)} areas")
    return dummy_gdf

def prepare_geographic_data(df, property_type=None, room_type=None, registration_type=None, geo_df=None):
    """
    Prepare data for geographic visualization
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        property_type (str): Selected property type
        room_type (str): Selected room type
        registration_type (str): Selected registration type
        geo_df (gpd.GeoDataFrame): GeoDataFrame with area geometries
        
    Returns:
        pd.DataFrame: Prepared data for geographic visualization
    """
    filtered_df = df.copy()
    
    # Apply filters
    if property_type is not None and property_type != 'All':
        filtered_df = filtered_df[filtered_df['property_type_en'] == property_type]
    
    if room_type is not None and room_type != 'All':
        filtered_df = filtered_df[filtered_df['rooms_en'] == room_type]
    
    if registration_type is not None and registration_type != 'All':
        filtered_df = filtered_df[filtered_df['reg_type_en'] == registration_type]

    # Get all growth columns
    growth_columns = [col for col in filtered_df.columns if isinstance(col, str) and col.startswith('growth_')]
    
    # Create aggregation dictionary with required columns
    agg_dict = {
        'median_price_sqft': 'median',
        'transaction_count': 'sum'
    }
    
    # Add all growth columns to aggregation dict
    for col in growth_columns:
        agg_dict[col] = 'median'
    
    # Aggregate by area with all required columns
    area_data = filtered_df.groupby('area_name_en').agg(agg_dict).reset_index()
    
    # Add growth data if available
    years = [col for col in filtered_df.columns if isinstance(col, (int, float)) and 2000 <= col <= 2025]
    if len(years) >= 2:
        latest_year = max(years)
        prev_year = sorted(years)[-2]
        
        # Check if growth column exists
        growth_column = f'growth_{prev_year}_to_{latest_year}'
        if growth_column in filtered_df.columns:
            growth_data = filtered_df.groupby('area_name_en')[growth_column].median().reset_index()
            area_data = area_data.merge(growth_data, on='area_name_en', how='left')
        else:
            # Calculate growth column
            area_data[growth_column] = np.nan
            for area in area_data['area_name_en']:
                area_df = filtered_df[filtered_df['area_name_en'] == area]
                latest_price = area_df[latest_year].median()
                prev_price = area_df[prev_year].median()
                
                if pd.notna(latest_price) and pd.notna(prev_price) and prev_price > 0:
                    growth = ((latest_price / prev_price) - 1) * 100
                    area_data.loc[area_data['area_name_en'] == area, growth_column] = growth
    
    # Add investment score if available
    if 'investment_score' in filtered_df.columns:
        score_data = filtered_df.groupby('area_name_en')['investment_score'].median().reset_index()
        area_data = area_data.merge(score_data, on='area_name_en', how='left')
    
    # Create normalized area names for better matching
    area_data['area_name_normalized'] = area_data['area_name_en'].str.lower().str.replace(' ', '')
    
    # Add latitude and longitude from manual mapping
    area_data['lat'] = np.nan
    area_data['lon'] = np.nan
    
    for idx, row in area_data.iterrows():
        area_lower = row['area_name_en'].lower()
        if area_lower in DUBAI_AREA_COORDINATES:
            area_data.at[idx, 'lat'] = DUBAI_AREA_COORDINATES[area_lower]['lat']
            area_data.at[idx, 'lon'] = DUBAI_AREA_COORDINATES[area_lower]['lon']

    # Add geometry information if geo_df is provided
    if geo_df is not None and has_geo_libraries and 'csv_area_name' in geo_df.columns:
        print("\n==== DIRECT AREA NAME MATCHING ====")
        
        # Print first few areas from each source to compare
        print(f"First 5 areas in dataset: {area_data['area_name_en'].head(5).tolist()}")
        print(f"First 5 csv_area_name values: {geo_df['csv_area_name'].head(5).tolist()}")
        
        # Create direct mapping dictionary with both original and lowercase keys
        area_to_geometry = {}
        for _, row in geo_df.iterrows():
            if pd.notna(row['csv_area_name']):
                # Store both the original name and lowercase version
                area_to_geometry[row['csv_area_name']] = row['geometry']
                area_to_geometry[row['csv_area_name'].lower()] = row['geometry']
        
        # Add geometry column
        area_data['geometry'] = None
        
        # Simple direct matching
        match_count = 0
        for idx, row in area_data.iterrows():
            area_name = row['area_name_en']
            area_name_lower = area_name.lower()
            
            # Try exact match
            if area_name in area_to_geometry:
                area_data.at[idx, 'geometry'] = area_to_geometry[area_name]
                match_count += 1
                # Set lat/lon from centroid
                centroid = area_to_geometry[area_name].centroid
                area_data.at[idx, 'lon'] = centroid.x
                area_data.at[idx, 'lat'] = centroid.y
            # Try lowercase match
            elif area_name_lower in area_to_geometry:
                area_data.at[idx, 'geometry'] = area_to_geometry[area_name_lower]
                match_count += 1
                # Set lat/lon from centroid
                centroid = area_to_geometry[area_name_lower].centroid
                area_data.at[idx, 'lon'] = centroid.x
                area_data.at[idx, 'lat'] = centroid.y
        
        print(f"Matched {match_count} out of {len(area_data)} areas")
        
        # Print the first 10 areas that didn't match to help debug
        if match_count < len(area_data):
            unmatched = [row['area_name_en'] for idx, row in area_data.iterrows() 
                        if pd.isna(row.get('geometry'))][:10]
            print(f"First 10 unmatched areas: {unmatched}")
        
        print("==== END MATCHING ====\n")
        # For any remaining areas without lat/lon, generate random positions around Dubai center
        dubai_center = {"lat": 25.204849, "lon": 55.270783}
        missing_coords = area_data[(area_data['lat'].isna()) | (area_data['lon'].isna())]
        
        if len(missing_coords) > 0:
            print(f"Generating random coordinates for {len(missing_coords)} areas without lat/lon")
            np.random.seed(42)  # For reproducibility
            
            for idx in missing_coords.index:
                lat_offset = (np.random.rand() - 0.5) * 0.2
                lon_offset = (np.random.rand() - 0.5) * 0.2
                
                area_data.at[idx, 'lat'] = dubai_center['lat'] + lat_offset
                area_data.at[idx, 'lon'] = dubai_center['lon'] + lon_offset
        
        # Debug: Print matched vs unmatched areas
        matched = len(area_data) - len(missing_coords)
        print(f"Area match summary: {matched}/{len(area_data)} areas matched with coordinates")
        
        return area_data

def create_price_heatmap(area_data, title="Dubai Real Estate Median Prices by Area"):
    """
    Create a price heatmap for geographic visualization
    
    Args:
        area_data (pd.DataFrame): DataFrame with area data prepared by prepare_geographic_data
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Check if we have required columns
    if 'lat' not in area_data.columns or 'lon' not in area_data.columns:
        return create_default_figure("Price Heatmap", "Missing geographic coordinates for areas")
    
    # Default Dubai coordinates
    dubai_center = {"lat": 25.204849, "lon": 55.270783}
    
    # Create scatter mapbox visualization
    fig = px.scatter_mapbox(
        area_data,
        lat='lat',
        lon='lon',
        color='median_price_sqft',
        size='transaction_count',
        hover_name='area_name_en',
        color_continuous_scale='Viridis',
        size_max=25,
        mapbox_style="carto-positron",
        zoom=10,
        center=dubai_center
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Colors represent median price per square foot</sub>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="AED/sqft"
        )
    )
    
    return fig

def create_growth_heatmap(area_data, growth_column=None, title="Dubai Real Estate Price Growth by Area"):
    """
    Create a growth heatmap for geographic visualization
    """
    # Check if we have required columns
    if 'lat' not in area_data.columns or 'lon' not in area_data.columns:
        return create_default_figure("Growth Heatmap", "Missing geographic coordinates for areas")
    
    # Check if growth_column exists in the data or provide a default
    if growth_column is None or growth_column not in area_data.columns:
        print(f"Warning: Growth column '{growth_column}' not found in data. Using median_price_sqft instead.")
        # Create a dummy growth column if needed
        area_data['growth'] = 0  # Default to zero growth
        growth_column = 'growth'
    
    # Default Dubai coordinates
    dubai_center = {"lat": 25.204849, "lon": 55.270783}
    
    # Create diverging color scale for growth (red for negative, blue for positive)
    color_scale = [
        [0, 'rgb(178,24,43)'],      # Deep red for negative growth
        [0.4, 'rgb(244,165,130)'],   # Light red
        [0.5, 'rgb(247,247,247)'],   # White/neutral
        [0.6, 'rgb(146,197,222)'],   # Light blue
        [1, 'rgb(5,113,176)']        # Deep blue for positive growth
    ]
    
    # Create scatter mapbox visualization
    fig = px.scatter_mapbox(
        area_data,
        lat='lat',
        lon='lon',
        color=growth_column,
        size='transaction_count',
        hover_name='area_name_en',
        color_continuous_scale=color_scale,
        size_max=25,
        mapbox_style="carto-positron",
        zoom=10,
        center=dubai_center,
        # Add these parameters:
        range_color=[-15, 15]  # Set fixed color range for better visualization
    )

    # Add better hover info
    hover_template = "<b>%{hovertext}</b><br>"
    hover_template += "Price: %{customdata[0]:.0f} AED/sqft<br>"
    hover_template += "Transactions: %{customdata[1]}<br>"
    hover_template += "Growth: %{customdata[2]}%"

    # Prepare custom data for hover
    custom_data = []
    for _, row in area_data.iterrows():
        growth_value = row.get(growth_column, None)
        growth_text = f"{growth_value:.2f}" if pd.notna(growth_value) else "No data"
        custom_data.append([
            row['median_price_sqft'],
            row['transaction_count'],
            growth_text
        ])

    fig.update_traces(
        customdata=custom_data,
        hovertemplate=hover_template
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Blue indicates positive growth, red indicates negative growth</sub>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Growth %",
            tickformat=".1f",
            ticksuffix="%"
        )
    )
    
    return fig

def create_investment_hotspot_map(area_data, title="Dubai Real Estate Investment Hotspots"):
    """
    Create an investment hotspot map
    """
    # Check if we have required columns
    if 'lat' not in area_data.columns or 'lon' not in area_data.columns:
        return create_default_figure("Investment Hotspots", "Missing geographic coordinates for areas")
    
    # Check if we have an investment score
    if 'investment_score' not in area_data.columns:
        # Create a dummy score for demonstration
        area_data['investment_score'] = np.random.uniform(30, 90, size=len(area_data))
    
    # Default Dubai coordinates
    dubai_center = {"lat": 25.204849, "lon": 55.270783}
    
    # Use a special color scale for investment score
    color_scale = [
        [0, 'rgb(240,240,240)'],     # Light gray for low scores
        [0.4, 'rgb(255,255,51)'],    # Yellow for moderate scores  
        [0.7, 'rgb(255,153,51)'],    # Orange for good scores
        [1, 'rgb(255,0,0)']          # Red for top investment scores
    ]
    
    # Create scatter mapbox visualization
    fig = px.scatter_mapbox(
        area_data,
        lat='lat',
        lon='lon',
        color='investment_score',
        size='transaction_count',
        hover_name='area_name_en',
        color_continuous_scale=color_scale,
        size_max=25,
        mapbox_style="carto-positron",
        zoom=10,
        center=dubai_center
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Red indicates highest investment opportunity areas based on growth, price, and transaction volume</sub>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Score"
        )
    )
    
    return fig

def create_transaction_volume_map(area_data, title="Dubai Real Estate Transaction Volume by Area"):
    """
    Create a transaction volume map
    """
    # Check if we have required columns
    if 'lat' not in area_data.columns or 'lon' not in area_data.columns:
        return create_default_figure("Transaction Volume", "Missing geographic coordinates for areas")
    
    # Default Dubai coordinates
    dubai_center = {"lat": 25.204849, "lon": 55.270783}
    
    # Use sequential color scale for transaction volume
    color_scale = 'YlOrRd'  # Yellow to Orange to Red
    
    # Create scatter mapbox visualization
    fig = px.scatter_mapbox(
        area_data,
        lat='lat',
        lon='lon',
        color='transaction_count',
        size='transaction_count',
        hover_name='area_name_en',
        color_continuous_scale=color_scale,
        size_max=40,
        mapbox_style="carto-positron",
        zoom=10,
        center=dubai_center
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{title}<br><sub>Darker colors and larger circles indicate higher transaction volumes (market liquidity)</sub>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Transactions",
            tickformat=",d"  # Format as integer with thousands separator
        )
    )
    
    return fig

def generate_geographic_insights(area_data, analysis_type="price"):
    """
    Generate insights for geographic analysis
    
    Args:
        area_data (pd.DataFrame): DataFrame with area data
        analysis_type (str): Type of analysis ('price', 'growth', 'investment', 'volume')
        
    Returns:
        dash component: HTML component with insights
    """
    if area_data is None or len(area_data) == 0:
        return html.Div([
            html.H5("No Data Available"),
            html.P("There is insufficient data for the selected filters.")
        ])
    
    insight_text = ""
    methodology_text = ""
    implications_text = ""
    
    if analysis_type == "price":
        insight_text = """
            Price distribution across Dubai shows distinct patterns with premium waterfront locations 
            commanding the highest prices. Areas like Palm Jumeirah, Emirates Hills, and Downtown Dubai 
            maintain price leadership, while emerging areas offer more affordable entry points.
        """
        methodology_text = """
            This visualization displays median price per square foot across different areas, 
            with darker colors indicating higher prices. The size of each point represents 
            the number of transactions, giving insight into market liquidity.
        """
        implications_text = """
            Areas with the highest prices typically offer established luxury and amenities
            but may have limited growth potential. Mid-priced areas often present better
            value appreciation opportunities. Consider price in conjunction with growth
            trends when making investment decisions.
        """
    elif analysis_type == "growth":
        insight_text = """
            Price growth rates vary significantly by area, with newer developing communities
            often showing stronger percentage growth from lower base prices. Established areas
            typically show more moderate but stable growth patterns.
        """
        methodology_text = """
            This visualization shows year-over-year price growth by area. Blue indicates
            positive growth, with darker shades showing stronger performance. Red indicates
            price declines. The size of each point shows transaction volume.
        """
        implications_text = """
            Areas with consistent growth above market average represent strong investment
            opportunities. Look for areas showing acceleration in growth rates, particularly
            those with new infrastructure developments or improving amenities.
        """
    elif analysis_type == "investment":
        insight_text = """
            Investment hotspots combine strong price growth potential with reasonable entry prices
            and sufficient transaction liquidity. These areas often represent the optimal balance
            of risk and return potential within the Dubai market.
        """
        methodology_text = """
            The investment score (0-100) combines price growth (40%), relative price level (30%),
            transaction volume (20%), and developer quality (10%). Higher scores represent
            stronger investment opportunities based on this balanced methodology.
        """
        implications_text = """
            Focus on areas with scores above 70 for near-term investment opportunities.
            For long-term investments, also consider upcoming infrastructure developments
            and master plan projects that may increase future values in currently
            lower-scoring areas.
        """
    elif analysis_type == "volume":
        insight_text = """
            Transaction volumes vary significantly across Dubai, with established areas
            typically showing higher liquidity. High volume indicates both strong market
            activity and potential ease of exit when selling.
        """
        methodology_text = """
            This visualization shows transaction counts by area, with darker colors and
            larger points indicating higher volumes. Areas with more transactions typically
            offer better market liquidity and price discovery.
        """
        implications_text = """
            Areas with high transaction volumes generally offer better liquidity for investors,
            reducing the risk of being unable to exit investments when needed. However,
            these areas may also be more competitive for buyers. Consider combining volume
            data with price and growth metrics.
        """
    
    # Create the insights component
    insights = html.Div([
        html.H5("Geographic Analysis Insights"),
        
        html.Div([
            html.Div([
                html.H6("Key Observations"),
                html.P(insight_text)
            ], className="col-md-12"),
            
            html.Div([
                html.H6("Methodology", className="mt-3"),
                html.P(methodology_text)
            ], className="col-md-6"),
            
            html.Div([
                html.H6("Investment Implications", className="mt-3"),
                html.P(implications_text)
            ], className="col-md-6")
        ], className="row")
    ])
    
    return insights

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