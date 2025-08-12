#!/usr/bin/env python3
"""
Plot Time Ranges for Battery Benchmark Data

This script creates a visualization showing the available time ranges for each dataset
as horizontal bars with details displayed inside the bars.
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

def plot_time_ranges(datasets, save_path='plots/time_ranges.png'):
    """Plot time ranges for each dataset as horizontal bars"""
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define colors for each dataset
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Get global time range
    all_dates = []
    for df in datasets.values():
        all_dates.extend([df['timestamp'].min(), df['timestamp'].max()])
    
    global_start = min(all_dates)
    global_end = max(all_dates)
    
    # Plot each dataset as a horizontal bar
    for i, (name, df) in enumerate(datasets.items()):
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        duration = end_date - start_date
        
        # Convert dates to numeric values for plotting
        start_num = (start_date - global_start).total_seconds() / (24 * 3600)  # days
        end_num = (end_date - global_start).total_seconds() / (24 * 3600)  # days
        
        # Plot horizontal bar
        y_pos = len(datasets) - i - 1
        bar = ax.barh(y_pos, end_num - start_num, left=start_num, 
                     height=0.6, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add text inside the bar
        center_x = start_num + (end_num - start_num) / 2
        center_y = y_pos
        
        # Format duration text
        days = duration.days
        hours = duration.seconds // 3600
        if days > 0:
            duration_text = f"{days}d {hours}h"
        else:
            duration_text = f"{hours}h"
        
        # Add dataset name and details
        text_lines = [
            f"{name.upper()}",
            f"Records: {len(df):,}",
            f"Duration: {duration_text}",
            f"Mean Load: {df['load_kw'].mean():.3f} kW"
        ]
        
        # Position text inside bar
        for j, line in enumerate(text_lines):
            y_offset = 0.15 - j * 0.08
            ax.text(center_x, center_y + y_offset, line, 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color='white', bbox=dict(boxstyle="round,pad=0.2", 
                                          facecolor='black', alpha=0.7))
    
    # Set up the plot
    ax.set_ylim(-0.5, len(datasets) - 0.5)
    ax.set_xlim(0, (global_end - global_start).total_seconds() / (24 * 3600))
    
    # Set x-axis labels (months)
    months = []
    month_labels = []
    current_date = global_start.replace(day=1)
    while current_date <= global_end:
        months.append((current_date - global_start).total_seconds() / (24 * 3600))
        month_labels.append(current_date.strftime('%Y-%m'))
        current_date = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Set y-axis labels
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([f"Dataset {i+1}" for i in range(len(datasets))])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Set title and labels
    ax.set_title('Available Time Ranges for Battery Benchmark Datasets', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    
    # Add legend
    legend_elements = []
    for i, name in enumerate(datasets.keys()):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                           edgecolor='black', linewidth=1, 
                                           label=f"{name.upper()}"))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Add summary statistics
    summary_text = f"Total Time Coverage: {global_start.strftime('%Y-%m-%d')} to {global_end.strftime('%Y-%m-%d')}\n"
    summary_text += f"Overall Duration: {(global_end - global_start).days} days\n"
    summary_text += f"Total Records: {sum(len(df) for df in datasets.values()):,}"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                                             facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time ranges plot saved to {save_path}")
    
    plt.close()

def main():
    """Main function to create time range visualization"""
    print("Creating Time Range Visualization")
    print("="*50)
    
    # Define file paths
    files = {
        'file1': 'data/0080E1FA00236634_consumption.csv',
        'file2': 'data/0080E1FA00237198_consumption.csv',
        'file3': 'data/0080E1FA00236638_consumption.csv'
    }
    
    # Load all datasets
    datasets = {}
    for name, file_path in files.items():
        if os.path.exists(file_path):
            datasets[name] = load_data(file_path)
        else:
            print(f"Warning: {file_path} not found!")
    
    if not datasets:
        print("No data files found!")
        return
    
    # Create time range visualization
    plot_time_ranges(datasets)
    
    # Print summary
    print("\n" + "="*50)
    print("TIME RANGE SUMMARY")
    print("="*50)
    
    for name, df in datasets.items():
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        duration = end_date - start_date
        
        print(f"\n{name.upper()}:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        print(f"  Duration: {duration}")
        print(f"  Records: {len(df):,}")
        print(f"  Mean Load: {df['load_kw'].mean():.4f} kW")

if __name__ == "__main__":
    main()
