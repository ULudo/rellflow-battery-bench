#!/usr/bin/env python3
"""
Extract Benchmark Months

This script extracts the recommended months from the data files for the benchmark environment.
Based on the analysis results, it will extract:
- Winter: December 2024 from file1
- Spring: March 2025 from file1  
- Summer: July 2024 from file1
"""

import pandas as pd
import json
import os
from pathlib import Path

def load_recommendations():
    """Load the benchmark recommendations"""
    with open('benchmark_recommendations.json', 'r') as f:
        return json.load(f)

def extract_month_data(file_path, year_month_str, output_path):
    """Extract data for a specific month and save to CSV"""
    print(f"Extracting {year_month_str} from {file_path}...")
    
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Parse the year-month string (e.g., "2024-12")
    year, month = map(int, year_month_str.split('-'))
    
    # Filter for the specific month
    month_data = df[
        (df['timestamp'].dt.year == year) & 
        (df['timestamp'].dt.month == month)
    ].copy()
    
    # Sort by timestamp
    month_data = month_data.sort_values('timestamp')
    
    # Save to CSV
    month_data.to_csv(output_path, index=False)
    
    print(f"  Extracted {len(month_data):,} records")
    print(f"  Date range: {month_data['timestamp'].min()} to {month_data['timestamp'].max()}")
    print(f"  Mean load: {month_data['load_kw'].mean():.4f} kW")
    print(f"  Saved to: {output_path}")
    
    return month_data

def main():
    """Main function to extract benchmark months"""
    print("Extracting Benchmark Months")
    print("="*50)
    
    # Load recommendations
    recommendations = load_recommendations()
    
    # Create output directory
    os.makedirs('benchmark_data', exist_ok=True)
    
    # File mapping
    file_mapping = {
        'file1': 'data/0080E1FA00236634_consumption.csv',
        'file2': 'data/0080E1FA00237198_consumption.csv',
        'file3': 'data/0080E1FA00236638_consumption.csv'
    }
    
    extracted_data = {}
    
    # Extract each recommended month
    for season, rec in recommendations.items():
        dataset = rec['dataset']
        month = rec['month']
        file_path = file_mapping[dataset]
        
        output_filename = f"benchmark_data/{season.lower()}_{month}.csv"
        
        if os.path.exists(file_path):
            month_data = extract_month_data(file_path, month, output_filename)
            extracted_data[season] = {
                'file': output_filename,
                'records': len(month_data),
                'mean_load': month_data['load_kw'].mean(),
                'date_range': f"{month_data['timestamp'].min().date()} to {month_data['timestamp'].max().date()}"
            }
        else:
            print(f"Warning: {file_path} not found!")
    
    # Create summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    
    for season, data in extracted_data.items():
        print(f"\n{season} Season:")
        print(f"  File: {data['file']}")
        print(f"  Records: {data['records']:,}")
        print(f"  Mean Load: {data['mean_load']:.4f} kW")
        print(f"  Date Range: {data['date_range']}")
    
    # Save extraction summary
    with open('benchmark_data/extraction_summary.json', 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    print(f"\nExtraction summary saved to: benchmark_data/extraction_summary.json")
    print("\nBenchmark data extraction complete!")

if __name__ == "__main__":
    main()
