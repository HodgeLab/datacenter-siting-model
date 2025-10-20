"""
Script to run optimization sweeps with different configurations.
Usage: python run_optimization_sweep.py --sweep economic_sweep --output results/economic
"""

import argparse
from pathlib import Path
from sweeps import (SWEEP, OptimizationConfig, OptimizationExperimentRunner)

from config import config
from cost_dict import *
'''
def main():
    parser = argparse.ArgumentParser(description='Run optimization parameter sweeps')
    parser.add_argument('--sweep', type=str, default='test_sweep', 
                        choices=list(SWEEP.keys()),
                        help='Which sweep configuration to run')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--base-name', type=str, default='datacenter_opt',
                        help='Base name for experiments')
    parser.add_argument('--time-limit', type=float, default=3600,
                        help='Solver time limit in seconds')
    
    args = parser.parse_args()
    
    print(f"Running sweep: {args.sweep}")
    print(f"Output directory: {args.output}")
    
    # Define your file paths here
    file_paths = {
        'state_shapefile': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2022_us_state_20m/cb_2022_us_state_20m.shp',
        'county_csv': 'CountyMaps/county_data.csv',
        'supply_data': '/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water_clim.csv',
        'merged_cf': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/merged_hourly_solar_wind_cf.csv',
        'demand_data': 'fake_demand.csv',
        'county2zone': 'CountyMaps/county2zone.csv',
        'hierarchy': 'CountyMaps/hierarchy.csv',
        'electric_prices': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/electric_prices.csv',
        'water_risk': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_risk.gpkg'
    }
    
    # Create experiment runner
    runner = OptimizationExperimentRunner(file_paths)
    
    # Define base configuration
    base_config = OptimizationConfig(
        exp_name=args.base_name,
        solver_time_limit=args.time_limit
    )
    
    # Get sweep configuration
    sweep_dict = SWEEP[args.sweep]
    print(f"Sweep parameters: {sweep_dict}")
    
    # Run sweep
    results = runner.run_sweep(base_config, sweep_dict)
    
    # Save results
    runner.save_results(results, output_dir=args.output)
    
    print(f"\n{'='*50}")
    print(f"SWEEP COMPLETE: {args.sweep}")
    print(f"Total experiments: {len(results)}")
    print(f"Results saved to: {args.output}/")
    print(f"{'='*50}")
    
    # Print top 3 results
    successful_results = [(cfg, metrics) for cfg, metrics in results if metrics.get('status') == 'success']
    
    if successful_results:
        # Sort by objective value
        successful_results.sort(key=lambda x: x[1].get('objective_value', float('inf')))
        
        print(f"\nTOP 3 RESULTS:")
        print("-" * 30)
        for i, (cfg, metrics) in enumerate(successful_results[:3], 1):
            print(f"{i}. Location: {metrics.get('selected_location')}")
            print(f"   Objective: ${metrics.get('objective_value', 0):,.0f}")
            print(f"   Renewable %: {metrics.get('renewable_utilization', 0)*100:.1f}%")
            print(f"   Config: {cfg.exp_name}")
            print()


def run_custom_sweep():
    """Example of running a custom sweep with specific parameters."""
    
    # Define custom sweep for testing specific scenarios
    custom_sweep = {
        'datacenter_capacity': [200, 300, 400],      # Different DC sizes
        'curtail_penalty': [5.0, 20.0],             # Low vs high curtailment penalty
        'state_filter': ['TX', 'CA'],               # Texas vs California
        'max_water_risk': [3.0, 4.0]               # Different water risk tolerances
    }
    
    file_paths = {
        'state_shapefile': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2022_us_state_20m/cb_2022_us_state_20m.shp',
        'county_csv': 'CountyMaps/county_data.csv',
        'supply_data': '/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water_clim.csv',
        'merged_cf': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/merged_hourly_solar_wind_cf.csv',
        'demand_data': 'fake_demand.csv',
        'county2zone': 'CountyMaps/county2zone.csv',
        'hierarchy': 'CountyMaps/hierarchy.csv',
        'electric_prices': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/electric_prices.csv',
        'water_risk': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_risk.gpkg',
        'county_shapefile': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'
    }
    
    runner = OptimizationExperimentRunner(file_paths)
    base_config = OptimizationConfig(exp_name="custom_sweep_test")
    
    results = runner.run_sweep(base_config, custom_sweep)
    runner.save_results(results, output_dir="results/custom")
    
    print(f"Custom sweep complete: {len(results)} experiments")


def analyze_sweep_results(results_file: str):
    """Analyze results from a completed sweep."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(results_file)
    successful = df[df['status'] == 'success']
    
    if len(successful) == 0:
        print("No successful experiments to analyze")
        return
    
    print(f"Analysis of {len(successful)} successful experiments:")
    print(f"Best objective: ${successful['objective_value'].min():,.0f}")
    print(f"Worst objective: ${successful['objective_value'].max():,.0f}")
    print(f"Average renewable utilization: {successful['renewable_utilization'].mean()*100:.1f}%")
    
    # Plot results if matplotlib available
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Objective value distribution
        ax1.hist(successful['objective_value'], bins=20)
        ax1.set_title('Objective Value Distribution')
        ax1.set_xlabel('Objective Value ($)')
        
        # Renewable utilization
        ax2.scatter(successful['total_renewable_capacity'], 
                   successful['renewable_utilization'])
        ax2.set_title('Renewable Utilization vs Capacity')
        ax2.set_xlabel('Total Renewable Capacity (MW)')
        ax2.set_ylabel('Renewable Utilization %')
        
        # Location frequency
        location_counts = successful['selected_location'].value_counts().head(10)
        ax3.bar(range(len(location_counts)), location_counts.values)
        ax3.set_title('Top 10 Selected Locations')
        ax3.set_ylabel('Selection Frequency')
        
        # Cost vs renewable utilization
        ax4.scatter(successful['renewable_utilization'], 
                   successful['objective_value'])
        ax4.set_title('Cost vs Renewable Utilization')
        ax4.set_xlabel('Renewable Utilization %')
        ax4.set_ylabel('Total Cost ($)')
        
        plt.tight_layout()
        plt.savefig('sweep_analysis.png')
        print("Analysis plots saved to sweep_analysis.png")
        
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Run the main sweep based on command line arguments
    main()
    
    # Uncomment to run custom sweep example
    run_custom_sweep()
    
    #Uncomment to analyze existing results
    analyze_sweep_results('results/all_results.csv')

'''


import pandas as pd
import glob
import os

# Folder where results are stored
results_dir = "results/"   # change if needed

# Match all your results files
files = glob.glob(os.path.join(results_dir, "datacenter_sweep_v1*results.csv"))

rows = []

for file in files:
    df = pd.read_csv(file)
    
    # If each file has only one row, just take it
    row = df.iloc[0].copy()
    
    # Add filename (without extension) as identifier
    row["exp_file"] = os.path.splitext(os.path.basename(file))[0]
    
    rows.append(row)

# Combine all into one DataFrame
combined = pd.DataFrame(rows)

# Save to CSV
combined.to_csv(os.path.join(results_dir, "all_results_combined.csv"), index=False)

