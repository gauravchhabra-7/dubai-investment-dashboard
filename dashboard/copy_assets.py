# copy_assets.py
import os
import shutil
import glob

def copy_visualizations_to_assets():
    """Copy all visualization files to the dashboard assets folder"""
    # Get the absolute path of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up if we're in the dashboard directory
    if os.path.basename(base_dir) == 'dashboard':
        project_dir = os.path.dirname(base_dir)
    else:
        project_dir = base_dir
    
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(project_dir, 'dashboard', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Output directories to check
    source_dirs = [
        os.path.join(project_dir, 'output', 'visualizations'),
        os.path.join(project_dir, 'output', 'visualizations', 'geographic'),
        os.path.join(project_dir, 'output', 'visualizations', 'time_series'),
        os.path.join(project_dir, 'output', 'visualizations', 'supply_demand')
    ]
    
    # Copy from each directory
    files_copied = 0
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            print(f"Checking {source_dir} for visualization files...")
            for ext in ['*.png', '*.html']:
                for file_path in glob.glob(os.path.join(source_dir, ext)):
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(assets_dir, file_name)
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {file_name} to assets folder")
                    files_copied += 1
        else:
            print(f"Directory not found: {source_dir}")
    
    # Create investment score explanation
    create_investment_score_explanation(os.path.join(assets_dir, 'investment_score_explanation.png'))
    
    print(f"\nTotal files copied: {files_copied}")
    print(f"All visualization files copied to {assets_dir}")

def create_investment_score_explanation(output_path):
    """Create a chart explaining the investment score calculation."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define the factors and weights used in the investment score
    factors = [
        'Short-term growth (1 year)',
        'Long-term CAGR (5 years)',
        'Developer premium/reputation',
        'Consistent outperformance',
        'Market liquidity (transaction volume)',
        'Emerging area potential',
        'Statistical significance'
    ]
    
    weights = [0.20, 0.15, 0.10, 0.15, 0.10, 0.15, 0.15]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(factors, weights, color='skyblue')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=10)
    
    # Add explanatory text
    explanation = """
    Investment Score Calculation Methodology:
    
    The investment score is a composite metric designed to identify optimal real estate investment opportunities.
    It combines short and long-term price appreciation, developer quality, consistency of performance,
    market liquidity, growth potential, and statistical robustness.
    
    Each property or area receives a score from 0-100, with higher scores indicating better investment prospects.
    Scores are then categorized as:
    - Exceptional Opportunity (90th percentile+)
    - Strong Opportunity (75th-90th percentile)
    - Good Opportunity (50th-75th percentile)
    - Average Opportunity (25th-50th percentile)
    - Below Average Opportunity (below 25th percentile)
    """
    
    plt.figtext(0.1, 0.02, explanation, wrap=True, fontsize=12)
    
    plt.title('Investment Score Components and Weights', fontsize=16)
    plt.xlabel('Weight in Final Score', fontsize=12)
    plt.xlim(0, 0.30)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Make room for text at bottom
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Created investment score explanation at {output_path}")

if __name__ == "__main__":
    copy_visualizations_to_assets()