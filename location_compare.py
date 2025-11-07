"""
Methods for comparing specific locations of interest in data center optimization.
Shows different approaches: single optimization vs individual comparisons.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from data_loader import process_data_pipeline
from siting_model import run_datacenter_optimization
from sweeps import OptimizationConfig
from cost_dict import *


class LocationComparisonAnalyzer:
    """
    Analyzer for comparing specific locations of interest.
    """
    
    def __init__(self, file_paths: Dict[str, str]):
        self.file_paths = file_paths
        self.processor = None
        self.model_dictionaries = None
        
        
        from data_loader import process_data_pipeline
        
        self.processor, self.model_dictionaries = process_data_pipeline(
            file_paths=self.file_paths,
            pue_climate_dict=pue_climate_region_same,
            wue_climate_dict=wue_climate_region_same,
            trans_mult_dict=trans_mult_dict,
            telecom_cost_dict=telecom_cost,
        )


# APPROACH 1: SINGLE OPTIMIZATION 
def compare_locations_single_optimization(analyzer: LocationComparisonAnalyzer, 
                                         target_locations: List[int],
                                         config: OptimizationConfig) -> Dict[str, Any]:
    """
    Method 1: Run single optimization with only target locations.
    The model will automatically choose the best one.
    
    This is the RECOMMENDED approach for finding the optimal location.
    """
    print(f"\n{'='*60}")
    print("METHOD 1: SINGLE OPTIMIZATION WITH TARGET LOCATIONS")
    print(f"{'='*60}")
    
    # Filter data to only include target locations
    filtered_dictionaries = {}
    
    for key, data_dict in analyzer.model_dictionaries.items():
        if isinstance(data_dict, dict):
            if key in ['energy_load', 'water_load', 'solar_generation', 'wind_generation', 'geo_generation']:
                # Handle (hour, location) keys
                filtered_dictionaries[key] = {
                    k: v for k, v in data_dict.items() 
                    if isinstance(k, tuple) and k[1] in target_locations
                }
            elif key == 'base_load':
                # Handle hour-only keys
                filtered_dictionaries[key] = data_dict
            elif key in ['fips_to_region', 'ba_to_region']:
                # Handle regional mappings
                filtered_dictionaries[key] = data_dict
            else:
                # Handle location-only keys
                filtered_dictionaries[key] = {
                    k: v for k, v in data_dict.items() 
                    if k in target_locations
                }
        else:
            filtered_dictionaries[key] = data_dict
    
    print(f"Filtered to {len([k for k in filtered_dictionaries['solar_capacity'].keys()])} target locations")
    
    # Run optimization
    cost_params = config.to_cost_params()
    model_config = config.to_model_config()
    
    opt_model, solution = run_datacenter_optimization(
        model_dictionaries=filtered_dictionaries,
        config=model_config,
        cost_params=cost_params,
        solver_name=config.solver_name
    )
    
    return {
        'method': 'single_optimization',
        'target_locations': target_locations,
        'optimal_location': solution['selected_locations'][0] if solution['selected_locations'] else None,
        'optimal_cost': solution['objective_value'],
        'solution': solution,
        'model': opt_model
    }


# APPROACH 2: INDIVIDUAL COMPARISONS
def compare_locations_individual_runs(analyzer: LocationComparisonAnalyzer,
                                     target_locations: List[int],
                                     config: OptimizationConfig) -> Dict[str, Any]:
    """
    Method 2: Run separate optimization for each location.
    Forces the model to use each location and compares costs.
    
    Good for detailed analysis and understanding trade-offs.
    """
    print(f"\n{'='*60}")
    print("METHOD 2: INDIVIDUAL LOCATION OPTIMIZATIONS")
    print(f"{'='*60}")
    
    results = []
    
    for location in target_locations:
        print(f"\nOptimizing for location {location}...")
        
        # Filter to single location
        single_location_dict = {}
        
        for key, data_dict in analyzer.model_dictionaries.items():
            if isinstance(data_dict, dict):
                if key in ['energy_load', 'water_load', 'solar_generation', 'wind_generation', 'geo_generation']:
                    # Handle (hour, location) keys
                    single_location_dict[key] = {
                        k: v for k, v in data_dict.items() 
                        if isinstance(k, tuple) and k[1] == location
                    }
                elif key == 'base_load':
                    # Handle hour-only keys
                    single_location_dict[key] = data_dict
                elif key in ['fips_to_region', 'ba_to_region']:
                    # Handle regional mappings
                    single_location_dict[key] = data_dict
                else:
                    # Handle location-only keys - only include target location
                    if location in data_dict:
                        single_location_dict[key] = {location: data_dict[location]}
                    else:
                        single_location_dict[key] = {}
            else:
                single_location_dict[key] = data_dict
        
        # Run optimization for this location
        try:
            cost_params = config.to_cost_params()
            model_config = config.to_model_config()
            
            opt_model, solution = run_datacenter_optimization(
                model_dictionaries=single_location_dict,
                config=model_config,
                cost_params=cost_params,
                solver_name=config.solver_name
            )
            
            results.append({
                'location': location,
                'objective_value': solution['objective_value'],
                'status': solution['status'],
                'solution': solution,
                'renewable_capacity': {
                    'solar': single_location_dict['solar_capacity'].get(location, 0),
                    'wind': single_location_dict['wind_capacity'].get(location, 0),
                    'geo': single_location_dict.get('geo_capacity', {}).get(location, 0)
                }
            })
            
        except Exception as e:
            results.append({
                'location': location,
                'objective_value': float('inf'),
                'status': 'failed',
                'error': str(e),
                'renewable_capacity': {'solar': 0, 'wind': 0, 'geo': 0}
            })
    
    # Sort by objective value
    results.sort(key=lambda x: x['objective_value'])
    
    return {
        'method': 'individual_runs',
        'target_locations': target_locations,
        'results': results,
        'best_location': results[0]['location'] if results else None,
        'best_cost': results[0]['objective_value'] if results else float('inf')
    }


# APPROACH 3: COMPARISON TABLE 
def create_location_comparison_table(analyzer: LocationComparisonAnalyzer,
                                    target_locations: List[int]) -> pd.DataFrame:
    """
    Method 3: Create detailed comparison table without optimization.
    Shows characteristics of each location for manual comparison.
    """
    print(f"\n{'='*60}")
    print("METHOD 3: LOCATION CHARACTERISTICS COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for location in target_locations:
        location_data = {'location': location}
        
        # Get basic characteristics
        for key in ['solar_capacity', 'wind_capacity', 'geo_capacity', 'trans_dist', 
                   'telecom_dist', 'water_price', 'electric_price']:
            if key in analyzer.model_dictionaries:
                location_data[key] = analyzer.model_dictionaries[key].get(location, 0)
        
        # Calculate total renewable capacity
        location_data['total_renewable_capacity'] = (
            location_data.get('solar_capacity', 0) + 
            location_data.get('wind_capacity', 0) + 
            location_data.get('geo_capacity', 0)
        )
        
        # Get coordinates
        if 'location_coordinates' in analyzer.model_dictionaries:
            coords = analyzer.model_dictionaries['location_coordinates'].get(location, (None, None))
            location_data['latitude'] = coords[0]
            location_data['longitude'] = coords[1]
        
        # Get climate efficiency
        for key in ['pue', 'wue']:
            if key in analyzer.model_dictionaries:
                location_data[key] = analyzer.model_dictionaries[key].get(location, 0)
        
        # Get risk factors
        if 'water_risk' in analyzer.model_dictionaries:
            location_data['water_risk'] = analyzer.model_dictionaries['water_risk'].get(location, 0)
        
        comparison_data.append(location_data)
    
    df = pd.DataFrame(comparison_data)
    return df


# COMBINED ANALYSIS 
def comprehensive_location_analysis(file_paths: Dict[str, str],
                                   target_locations: List[int],
                                   config: OptimizationConfig = None) -> Dict[str, Any]:
    """
    Run all three comparison methods and provide comprehensive analysis.
    """
    if config is None:
        config = OptimizationConfig(exp_name="location_comparison")
    
    print(f"COMPREHENSIVE LOCATION ANALYSIS")
    print(f"Analyzing {len(target_locations)} locations: {target_locations}")
    
    # Setup analyzer
    analyzer = LocationComparisonAnalyzer(file_paths)
    analyzer.setup_data_processor(min_capacity=0)  # Don't filter by capacity
    
    results = {}
    
    # Method 1: Single optimization (recommended for finding optimal)
    try:
        results['single_opt'] = compare_locations_single_optimization(
            analyzer, target_locations, config
        )
    except Exception as e:
        results['single_opt'] = {'error': str(e)}
    
    # Method 2: Individual runs (good for detailed comparison)
    try:
        results['individual_runs'] = compare_locations_individual_runs(
            analyzer, target_locations, config
        )
    except Exception as e:
        results['individual_runs'] = {'error': str(e)}
    
    # Method 3: Comparison table (good for manual analysis)
    try:
        results['comparison_table'] = create_location_comparison_table(
            analyzer, target_locations
        )
    except Exception as e:
        results['comparison_table'] = {'error': str(e)}
    
    return results


def print_comprehensive_results(results: Dict[str, Any]):
    """Print formatted results from comprehensive analysis."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    # Single optimization results
    if 'single_opt' in results and 'error' not in results['single_opt']:
        single = results['single_opt']
        print(f"\n🎯 SINGLE OPTIMIZATION RESULT:")
        print(f"   Optimal Location: {single['optimal_location']}")
        print(f"   Optimal Cost: ${single['optimal_cost']:,.2f}")
    
    # Individual run results
    if 'individual_runs' in results and 'error' not in results['individual_runs']:
        individual = results['individual_runs']
        print(f"\n📊 INDIVIDUAL LOCATION RESULTS:")
        for i, result in enumerate(individual['results'][:5], 1):  # Top 5
            print(f"   {i}. Location {result['location']}: ${result['objective_value']:,.2f}")
    
    # Comparison table
    if 'comparison_table' in results and 'error' not in results['comparison_table']:
        df = results['comparison_table']
        print(f"\n📋 LOCATION CHARACTERISTICS:")
        print(df[['location', 'total_renewable_capacity', 'electric_price', 
                 'trans_dist', 'water_risk']].to_string(index=False))


# EXAMPLE USAGE 
if __name__ == "__main__":
    
    # Example file paths
    file_paths = {
        'supply_data': '/path/to/supply_data.csv',
        'merged_cf': '/path/to/merged_cf.csv',
        # ... other file paths
    }
    
    # Your 7 locations of interest
    target_locations = [12345, 23456, 34567, 45678, 56789, 67890, 78901]
    
    # Configuration
    config = OptimizationConfig(
        exp_name="7_location_comparison",
        datacenter_capacity=250,
        curtail_penalty=10.0
    )
    
    # Run comprehensive analysis
    results = comprehensive_location_analysis(file_paths, target_locations, config)
    
    # Print results
    print_comprehensive_results(results)
    
    # Save detailed results
    if 'comparison_table' in results:
        results['comparison_table'].to_csv('location_comparison.csv', index=False)
        print("\nDetailed comparison saved to 'location_comparison.csv'")


# RECOMMENDATION
"""
RECOMMENDATIONS:

1. **Use Method 1 (Single Optimization) if:**
   - You want the mathematically optimal choice
   - You trust the model to make the decision
   - You want the fastest solution

2. **Use Method 2 (Individual Runs) if:**
   - You want to understand trade-offs between locations
   - You need detailed analysis for each location
   - You want to validate the single optimization result

3. **Use Method 3 (Comparison Table) if:**
   - You want to do manual analysis
   - You need to present characteristics to stakeholders
   - The optimization is too complex or failing

4. **Use Comprehensive Analysis if:**
   - You want all perspectives
   - You're doing this analysis once and want complete information
   - You need to justify your decision with multiple approaches

The single optimization (Method 1) is usually the best approach because:
- It's mathematically rigorous
- It considers all interactions between variables
- It's computationally efficient
- It gives you the truly optimal solution

But individual runs (Method 2) are valuable for understanding WHY one location
is better than another.
"""