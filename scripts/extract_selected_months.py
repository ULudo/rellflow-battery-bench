#!/usr/bin/env python3
"""
Extract Selected Months for Battery Benchmark

This script extracts the selected months from each dataset:
- File1: July 2024
- File2: April 2025  
- File3: November 2025

Each extraction includes 24 hours before and after the selected month for warm-up and prediction.
Data is converted to 15-minute resolution with a PV column (all zeros).
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

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
    df_15min = df_copy.groupby('timestamp_15min').agg({
        'load_kw': 'sum',  # Sum of kW over 15 minutes
        'unixtime': 'first'  # Keep first unixtime for reference
    }).reset_index()
    
    # Rename timestamp column back to original name
    df_15min = df_15min.rename(columns={'timestamp_15min': 'timestamp'})
    
    # Calculate kWh/15min (sum of kW over 15 minutes)
    df_15min['load'] = df_15min['load_kw'] / 60  # Convert to kWh
    
    print(f"  Original records: {len(df):,}")
    print(f"  15-min records: {len(df_15min):,}")
    print(f"  Reduction factor: {len(df) / len(df_15min):.1f}x")
    
    return df_15min

def extract_month_with_buffer(df, year, month, buffer_hours=24):
    """Extract a specific month with buffer periods before and after"""
    
    # Define the target month with UTC timezone
    start_of_month = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_of_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end_of_month = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    
    # Add buffer periods
    start_with_buffer = start_of_month - timedelta(hours=buffer_hours)
    end_with_buffer = end_of_month + timedelta(hours=buffer_hours)
    
    print(f"  Extracting {year}-{month:02d} with {buffer_hours}h buffer")
    print(f"  Period: {start_with_buffer} to {end_with_buffer}")
    
    # Filter data
    mask = (df['timestamp'] >= start_with_buffer) & (df['timestamp'] <= end_with_buffer)
    filtered_df = df[mask].copy()
    
    print(f"  Extracted records: {len(filtered_df):,}")
    
    return filtered_df

def create_final_dataset(df_15min, file_id):
    """Create final dataset with unixtime, load, and pv columns"""
    
    # Create final dataset
    final_df = pd.DataFrame({
        'unixtime': df_15min['unixtime'],
        'load': df_15min['load'],
        'pv': 0.0  # All zeros as requested
    })
    
    print(f"  Final dataset shape: {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")
    
    return final_df

def plot_selected_month(df_15min, file_id, year, month, save_path):
    """Create plot for the selected month"""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter to just the target month (without buffer)
    start_of_month = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_of_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    else:
        end_of_month = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
    
    month_data = df_15min[(df_15min['timestamp'] >= start_of_month) & 
                          (df_15min['timestamp'] <= end_of_month)].copy()
    
    if len(month_data) == 0:
        print(f"  No data found for {year}-{month:02d}")
        return
    
    # Calculate statistics
    mean_load = month_data['load'].mean()
    max_load = month_data['load'].max()
    min_load = month_data['load'].min()
    total_consumption = month_data['load'].sum()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot load over time
    ax.plot(month_data['timestamp'], month_data['load'], 
           linewidth=1.2, alpha=0.8, color='blue', marker='o', markersize=2)
    
    # Set title with statistics
    month_name = datetime(year, month, 1).strftime('%B %Y')
    title = f'Load Profile - {file_id} - {month_name}\n'
    title += f'Mean: {mean_load:.4f} kWh/15min | Max: {max_load:.4f} kWh/15min | '
    title += f'Min: {min_load:.4f} kWh/15min | Total: {total_consumption:.2f} kWh'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Load (kWh/15min)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"Records: {len(month_data):,}\nStd: {month_data['load'].std():.4f} kWh/15min"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                             facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {save_path}")
    plt.close()

def main():
    """Main function to extract selected months"""
    print("Extracting Selected Months for Battery Benchmark")
    print("="*60)
    
    # Define file mappings
    files = {
        'file1': {
            'path': 'data/0080E1FA00236634_consumption.csv',
            'id': '0080E1FA00236634',
            'year': 2024,
            'month': 7  # July
        },
        'file2': {
            'path': 'data/0080E1FA00237198_consumption.csv', 
            'id': '0080E1FA00237198',
            'year': 2025,
            'month': 4  # April
        },
        'file3': {
            'path': 'data/0080E1FA00236638_consumption.csv',
            'id': '0080E1FA00236638', 
            'year': 2024,
            'month': 11  # November 2024 (not 2025)
        }
    }
    
    # Create output directories
    os.makedirs('selected_months', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Process each file
    for file_key, file_info in files.items():
        print(f"\nProcessing {file_key} ({file_info['id']})...")
        
        if os.path.exists(file_info['path']):
            # Load data
            df = load_data(file_info['path'])
            
            # Extract month with buffer
            df_extracted = extract_month_with_buffer(df, file_info['year'], file_info['month'])
            
            if len(df_extracted) > 0:
                # Convert to 15-minute resolution
                df_15min = convert_to_15min_resolution(df_extracted)
                
                # Create final dataset
                final_df = create_final_dataset(df_15min, file_info['id'])
                
                # Save CSV file
                output_path = f"selected_months/{file_info['id']}.csv"
                final_df.to_csv(output_path, index=False)
                print(f"  CSV saved to: {output_path}")
                
                # Create plot
                plot_path = f"plots/{file_info['id']}_load_profile.png"
                plot_selected_month(df_15min, file_info['id'], file_info['year'], file_info['month'], plot_path)
                
                # Print summary statistics
                month_start = datetime(file_info['year'], file_info['month'], 1, tzinfo=timezone.utc)
                if file_info['month'] < 12:
                    month_end = datetime(file_info['year'], file_info['month'] + 1, 1, tzinfo=timezone.utc)
                else:
                    month_end = datetime(file_info['year'] + 1, 1, 1, tzinfo=timezone.utc)
                
                month_data = df_15min[(df_15min['timestamp'] >= month_start) & 
                                     (df_15min['timestamp'] < month_end)].copy()
                
                print(f"  Month statistics:")
                print(f"    Mean load: {month_data['load'].mean():.4f} kWh/15min")
                print(f"    Max load: {month_data['load'].max():.4f} kWh/15min")
                print(f"    Min load: {month_data['load'].min():.4f} kWh/15min")
                print(f"    Total consumption: {month_data['load'].sum():.2f} kWh")
                print(f"    Records: {len(month_data):,}")
                
            else:
                print(f"  No data found for {file_info['year']}-{file_info['month']:02d}")
                
        else:
            print(f"Warning: {file_info['path']} not found!")
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print("\nGenerated files:")
    print("- selected_months/0080E1FA00236634.csv - July 2024 (File1)")
    print("- selected_months/0080E1FA00237198.csv - April 2025 (File2)")
    print("- selected_months/0080E1FA00236638.csv - November 2024 (File3)")
    print("\nGenerated plots:")
    print("- plots/0080E1FA00236634_load_profile.png")
    print("- plots/0080E1FA00237198_load_profile.png")
    print("- plots/0080E1FA00236638_load_profile.png")
    print("\nData format:")
    print("- unixtime: Unix timestamp")
    print("- load: Energy consumption in kWh per 15-minute interval")
    print("- pv: Solar PV generation (all zeros)")
    print("\nEach file includes:")
    print("- Selected month data")
    print("- 24 hours before the month (warm-up period)")
    print("- 24 hours after the month (prediction period)")

if __name__ == "__main__":
    main()
