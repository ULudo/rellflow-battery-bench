#!/usr/bin/env python3
"""
Plot Monthly Loads for Battery Benchmark Data (15-minute resolution)

This script converts 1-minute resolution data to 15-minute resolution by summing loads,
then creates detailed plots showing the load for each month in each dataset.
Each file gets its own plot with subplots for each month, showing load changes over time.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data(file_path):
    """Load data with optimized settings for large CSV files"""
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df

def convert_to_15min_resolution(df):
    """Convert 1-minute resolution data to 15-minute resolution by summing loads"""
    print("  Converting to 15-minute resolution...")
    
    # Create a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Round timestamps to 15-minute intervals
    df_copy['timestamp_15min'] = df_copy['timestamp'].dt.floor('15min')
    
    # Group by 15-minute intervals and sum the loads
    # Note: We sum the loads (kW) over 15 minutes to get kWh equivalent
    df_15min = df_copy.groupby('timestamp_15min').agg({
        'load_kw': 'sum',  # Sum of kW over 15 minutes = kWh equivalent
        'unixtime': 'first'  # Keep first unixtime for reference
    }).reset_index()
    
    # Rename timestamp column back to original name
    df_15min = df_15min.rename(columns={'timestamp_15min': 'timestamp'})
    
    # Calculate kWh/15min (this is already the sum of kW over 15 minutes)
    # Since we have 1-minute data, summing 15 minutes of kW gives us kWh equivalent
    df_15min['load_kwh_15min'] = df_15min['load_kw'] / 60  # Convert to kWh (divide by 60 since we summed 15 minutes of kW)
    
    print(f"  Original records: {len(df):,}")
    print(f"  15-min records: {len(df_15min):,}")
    print(f"  Reduction factor: {len(df) / len(df_15min):.1f}x")
    
    return df_15min

def plot_monthly_loads_15min(df, filename, save_path):
    """Plot load for each month in the dataset at 15-minute resolution"""
    
    # Add month and year columns
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['timestamp'].dt.to_period('M')
    df_copy['month'] = df_copy['timestamp'].dt.month
    df_copy['year'] = df_copy['timestamp'].dt.year
    
    # Get unique months
    months = df_copy['year_month'].unique()
    months = sorted(months)
    
    if len(months) == 0:
        print(f"No data found for {filename}")
        return
    
    print(f"  Creating plot for {len(months)} months...")
    
    # Calculate subplot layout
    n_months = len(months)
    n_cols = 2  # 2 columns
    n_rows = (n_months + 1) // 2  # Ceiling division
    
    # Create figure with appropriate size
    fig_width = 20
    fig_height = 6 * n_rows  # 6 inches per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot each month
    for i, month in enumerate(months):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter data for this month
        month_data = df_copy[df_copy['year_month'] == month].copy()
        month_data = month_data.sort_values('timestamp')
        
        if len(month_data) == 0:
            continue
        
        # Plot load over time (using kWh/15min)
        ax.plot(month_data['timestamp'], month_data['load_kwh_15min'], 
               linewidth=1.2, alpha=0.8, color='red', marker='o', markersize=2)
        
        # Set title with month info and statistics
        month_str = str(month)
        mean_load = month_data['load_kwh_15min'].mean()
        max_load = month_data['load_kwh_15min'].max()
        min_load = month_data['load_kwh_15min'].min()
        
        title = f"{month_str}\nMean: {mean_load:.3f} kWh/15min | Max: {max_load:.3f} kWh/15min | Min: {min_load:.3f} kWh/15min"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_ylabel('Load (kWh/15min)', fontsize=10)
        ax.set_xlabel('Time', fontsize=10)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits with some padding
        y_min = max(0, min_load - 0.01)
        y_max = max_load + 0.01
        ax.set_ylim(y_min, y_max)
        
        # Add statistics text box
        stats_text = f"Records: {len(month_data):,}\nStd: {month_data['load_kwh_15min'].std():.3f} kWh/15min"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor='lightgray', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(months), len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Monthly Load Profiles (15-min resolution) - {filename}', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Monthly loads plot saved to {save_path}")
    
    plt.close()

def main():
    """Main function to create monthly load plots at 15-minute resolution"""
    print("Creating Monthly Load Plots (15-minute resolution)")
    print("="*60)
    
    # Define file paths
    files = {
        'file1': 'data/0080E1FA00236634_consumption.csv',
        'file2': 'data/0080E1FA00237198_consumption.csv',
        'file3': 'data/0080E1FA00236638_consumption.csv'
    }
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Process each file
    for name, file_path in files.items():
        if os.path.exists(file_path):
            print(f"\nProcessing {name}...")
            
            # Load data
            df = load_data(file_path)
            
            # Convert to 15-minute resolution
            df_15min = convert_to_15min_resolution(df)
            
            # Create monthly loads plot
            plot_monthly_loads_15min(df_15min, name, f"plots/{name}_monthly_loads_15min.png")
            
        else:
            print(f"Warning: {file_path} not found!")
    
    print("\n" + "="*60)
    print("MONTHLY LOADS PLOTS SUMMARY (15-min resolution)")
    print("="*60)
    print("\nGenerated plots:")
    print("- plots/file1_monthly_loads_15min.png - Monthly loads for Dataset 1")
    print("- plots/file2_monthly_loads_15min.png - Monthly loads for Dataset 2")
    print("- plots/file3_monthly_loads_15min.png - Monthly loads for Dataset 3")
    print("\nEach plot shows:")
    print("- Subplots for each month in the dataset")
    print("- Load (kWh/15min) on y-axis, time on x-axis")
    print("- 15-minute resolution data (converted from 1-minute)")
    print("- Monthly statistics (mean, max, min, std)")
    print("- High-resolution output for detailed analysis")
    print("\nData conversion:")
    print("- Original: 1-minute resolution (kW)")
    print("- Converted: 15-minute resolution (kWh/15min)")
    print("- Method: Sum of kW over 15-minute intervals")

if __name__ == "__main__":
    main()
