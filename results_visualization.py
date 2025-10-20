"""
Visualization module for data center optimization results.
Provides comprehensive plotting and analysis functionality.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from cost_dict import *
from config import config
from data_loader import process_data_pipeline
from siting_model import run_datacenter_optimization
from components.storage import *
from components.plant import *

import matplotlib as mpl

# --- Set Times New Roman globally ---
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelweight'] = 'bold'
#mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

class OptimizationVisualizer:
    """
    Class for visualizing optimization results and generating analysis plots.
    """
    
    def __init__(self, output_dir: str = "output_files"):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save plots and analysis files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _has_plants(self, model) -> bool:
        """Check if model uses plant systems (multi-plant) or legacy SMR."""
        return hasattr(model, 'PLANTS')
        
    def calculate_location_breakdown(self, model, cost_params: Dict[str, Dict], 
                                   storage_cost_per_mwh: float = 0) -> pd.DataFrame:
        """
        Calculate detailed cost breakdown and generation totals for each location.
        
        Args:
            model: Solved Pyomo model
            cost_params: Cost parameter dictionaries
            storage_cost_per_mwh: Storage cost per MWh
            
        Returns:
            DataFrame with per-location breakdowns
        """
        print("Calculating location breakdowns...")
        
        variable_gen_cost = cost_params.get('variable_gen_cost', {})
        capital_gen_cost = cost_params.get('capital_gen_cost', {})
        fixed_gen_cost = cost_params.get('fixed_gen_cost', {})

        use_plants = self._has_plants(model)
        
        results = []
        
        for loc in model.LOCATIONS:
            total_cost = 0
            variable_cost = 0
            grid_cost = 0
            curtail_cost = 0
            export_credit = 0
            water_cost = 0
            capital_cost = 0
            fixed_cost = 0
            transmission_cost = 0
            telecom_cost = 0
            
            # Generation totals
            solar_gen = 0
            wind_gen = 0
            geo_gen = 0
            storage_discharge = 0
            grid_import = 0
            solar_export = 0
            wind_export = 0
            
            # Plant generation (either legacy SMR or multi-plant)
            plant_gen = {}
            if use_plants:
                for p in model.PLANTS:
                    plant_gen[f'{p}_gen'] = 0
            
            for h in model.HOURS:
                # Renewable generation values
                solar_gen += pyo.value(model.solar_to_load[h, loc])
                wind_gen += pyo.value(model.wind_to_load[h, loc])
                geo_gen += pyo.value(model.geo_45_gen[h, loc])
                storage_discharge += pyo.value(model.storage_discharge[h, loc])
                grid_import += pyo.value(model.P_g[h, loc])
                solar_export += pyo.value(model.solar_to_grid[h, loc])
                wind_export += pyo.value(model.wind_to_grid[h, loc])
                
                # Plant generation
                if use_plants:
                    for p in model.PLANTS:
                        plant_gen[f'{p} Gen'] += pyo.value(model.plant_output[h, loc, p])
                
                # Costs
                variable_cost += (
                    pyo.value(model.solar_to_load[h, loc]) * variable_gen_cost.get('solar', 0)
                    + pyo.value(model.wind_to_load[h, loc]) * variable_gen_cost.get('wind', 0)
                    + pyo.value(model.geo_45_gen[h, loc]) * variable_gen_cost.get('geo_45', 0)
                )

                # Plant variable costs
                if use_plants:
                    # For multi-plant, costs should be calculated from Plant object
                    # This is a simplified version - you may need to pass plant objects
                    for p in model.PLANTS:
                        variable_cost += pyo.value(model.plant_output[h, loc, p]) * variable_gen_cost.get(p, 0)
            
                grid_cost += pyo.value(model.P_g[h, loc]) * pyo.value(model.grid_price[loc])
                
                curtail_cost += (
                    (pyo.value(model.wind_curtailed[h, loc]) + pyo.value(model.solar_curtailed[h, loc]))
                    * pyo.value(model.curtail_penalty)
                )
                export_credit += (
                    (pyo.value(model.wind_to_grid[h, loc]) + pyo.value(model.solar_to_grid[h, loc]))
                    * pyo.value(model.ren_export_price)
                )
                
                if hasattr(model, 'water_load'):
                    water_cost += (
                        pyo.value(model.water_load[h, loc]) * pyo.value(model.water_price_hourly[h, loc])
                    )
            
            # Capacity-based costs
            capital_cost = (
                pyo.value(model.solar_cap[loc]) * capital_gen_cost.get('solar', 0)
                + pyo.value(model.wind_cap[loc]) * capital_gen_cost.get('wind', 0)
                + pyo.value(model.geo_45_cap[loc]) * capital_gen_cost.get('geo_45', 0)
            )
            fixed_cost = (
                pyo.value(model.solar_cap[loc]) * fixed_gen_cost.get('solar', 0)
                + pyo.value(model.wind_cap[loc]) * fixed_gen_cost.get('wind', 0)
                + pyo.value(model.geo_45_cap[loc]) * fixed_gen_cost.get('geo_45', 0)
            )
            # Plant capital and fixed costs
            if use_plants:
                for p in model.PLANTS:
                    capital_cost += pyo.value(model.plant_capacity[p]) * capital_gen_cost.get(p, 0)
                    fixed_cost += pyo.value(model.plant_capacity[p]) * fixed_gen_cost.get(p, 0)

            if hasattr(model, 'trans_dist'):
                transmission_cost = (
                    pyo.value(model.trans_dist[loc]) * pyo.value(model.trans_cap_cost[loc]) 
                    * pyo.value(model.trans_multiplier[loc])
                )
            
            if hasattr(model, 'telecom_cost'):
                telecom_cost = pyo.value(model.telecom_cost[loc])
            
            # Net total cost
            total_cost = (
                variable_cost + grid_cost + curtail_cost - export_credit
                + water_cost + capital_cost + fixed_cost
                + transmission_cost + telecom_cost
            )
            
            result_dict = {
                "location": loc,
                "objective": total_cost,
                "Variable Cost": variable_cost,
                "Grid Cost": grid_cost,
                "Curtail Cost": curtail_cost,
                "Export Credit": export_credit,
                "Water Cost": water_cost,
                "Capital Cost": capital_cost,
                "Fixed Cost": fixed_cost,
                "Transmission Cost": transmission_cost,
                "Telecom Cost": telecom_cost,
                # Generation totals
                "Solar Gen": solar_gen,
                "Wind Gen": wind_gen,
                "Geo Gen": geo_gen,
                "Storage Discharge": storage_discharge,
                "Grid Import": grid_import,
                "Solar Export": solar_export,
                "Wind Export": wind_export,
            }

            # Add plant generation columns
            result_dict.update(plant_gen)
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def create_correlation_analysis(self, location_df: pd.DataFrame, 
                                   supply_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create correlation analysis between objective and location characteristics.
        
        Args:
            location_df: Location breakdown DataFrame
            supply_data: Original supply data with location characteristics
            
        Returns:
            Merged DataFrame with correlations
        """
        print("Creating correlation analysis...")
        
        # Merge with original data characteristics
        og_data = supply_data[['location', 'trans_dist', 'telecom_dist', 'clim_zone']].copy()    #'latitude', 'longitude', 
        
        merged_df = location_df.merge(og_data, on='location', how='left')

        merged_df = merged_df.rename(
            columns = {
            'trans_dist': 'Trans Dist',
            'telecom_dist': 'Telecom Dist',
            'clim_zone': 'Climate Zone'
            }
        )
        
        # Calculate correlations
        correlations = merged_df.corr(numeric_only=True)['objective'].sort_values()
        
        print("Correlations with objective:")
        print(correlations)
        
        return merged_df, correlations
    
    def plot_correlation_analysis(self, correlations: pd.Series, 
                                 figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot correlation analysis between objective and features.
        
        Args:
            correlations: Correlation series
            figsize: Figure size
        """
        correlations = correlations.drop('location')

        plt.figure(figsize=figsize)
        correlations_plot = correlations.drop('objective') if 'objective' in correlations else correlations
        correlations_plot.plot(kind='barh')
        #plt.title('Correlation with Total Cost (Objective)')
        plt.xlabel('Pearson Correlation', fontweight='bold', size = 18)
        plt.grid(True, alpha=0.3)
        #plt.rcParams.update({'font.size': 14})
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_county_choropleth(self, location_df: pd.DataFrame, 
                                county_shapefile_path: str,
                                exclude_states: List[str] = ['02', '15', '60', '66', '69', '72', '78'],
                                figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Create choropleth map of optimization results by county.
        
        Args:
            location_df: Location breakdown DataFrame
            county_shapefile_path: Path to county shapefile
            exclude_states: State FIPS codes to exclude (AK, HI, territories)
            figsize: Figure size
        """
        print("Creating county choropleth map...")
        
        # Load county shapefile
        county_shape_data = gpd.read_file(county_shapefile_path)
        
        # Filter out excluded states
        county_shape_data = county_shape_data[~county_shape_data['STATEFP'].isin(exclude_states)]
        
        # Create location ID
        county_shape_data['location'] = (
            county_shape_data['STATEFP'].astype(str).str.zfill(2) + 
            county_shape_data['COUNTYFP'].astype(str).str.zfill(3)
        )
        county_shape_data['location'] = county_shape_data['location'].astype('Int64')
        
        # Merge with optimization results
        merged = location_df[['location', "objective"]].merge(
            county_shape_data, on='location', how="right"
        )
        merged = gpd.GeoDataFrame(merged, geometry='geometry')
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all counties (base layer)
        county_shape_data.plot(
            ax=ax,
            color='white',
            edgecolor='lightgray',
            linewidth=0.3
        )
        
        # Plot counties with data
        merged.dropna(subset=['objective']).plot(
            column='objective',
            ax=ax,
            cmap='coolwarm',
            edgecolor='gray',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': 'Total Cost ($)',
                         'orientation': 'horizontal',
                         'shrink': 0.8,              # make colorbar height match map better
                         'aspect': 60,
                         'pad': 0.001}
        )
        cbar = ax.get_figure().axes[-1]  # colorbar axis
        cbar.set_xlabel('Total Cost ($)', fontsize=16)  
        cbar.tick_params(labelsize=14)
        #ax.set_title('Data Center Optimization Results by County', fontsize=16)
        ax.axis('off')
        plt.tight_layout(pad=0.25)
        #plt.rcParams.update({'font.size': 14})
        
        # Save plot
        plt.savefig(self.output_dir / 'county_choropleth.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_generation_dispatch_plot(self, model, location: int = None,
                                      figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Create stacked generation dispatch plot for a specific location.
        
        Args:
            model: Solved Pyomo model
            location: Location to plot (if None, uses first selected location)
            figsize: Figure size
        """
        print("Creating generation dispatch plot...")

        use_plants = self._has_plants(model)
        
        # Find selected location
        if location is None:
            selected_locs = [loc for loc in model.LOCATIONS if pyo.value(model.x[loc]) > 0.5]
            if not selected_locs:
                print("No selected locations found for dispatch plot")
                return
            location = selected_locs[1]
        
        # Prepare hourly generation DataFrame
        hours = list(model.HOURS)
        gen_df = pd.DataFrame({'hour': hours})
        
        # Load data
        load = [pyo.value(model.load_datacenter[h, location]) for h in hours]
        
        # Extract generation from optimized variables
        gen_df['solar'] = [pyo.value(model.solar_to_load[h, location]) for h in hours]
        gen_df['wind'] = [pyo.value(model.wind_to_load[h, location]) for h in hours]
        gen_df['geo_45'] = [pyo.value(model.geo_45_gen[h, location]) for h in hours]
        gen_df['grid_import'] = [pyo.value(model.P_g[h, location]) for h in hours]
        gen_df['storage_discharge'] = [pyo.value(model.storage_discharge[h, location]) for h in hours]

        # Plant generation
        plant_columns = []
        if use_plants:
            for p in model.PLANTS:
                col_name = f'{p}_gen'  #f'{p}_gen'
                gen_df[col_name] = [pyo.value(model.plant_output[h, location, p]) for h in hours]
                plant_columns.append(col_name)
        
        # Optional: curtailed and exports
        gen_df['solar_curtailed'] = [pyo.value(model.solar_curtailed[h, location]) for h in hours]
        gen_df['wind_curtailed'] = [pyo.value(model.wind_curtailed[h, location]) for h in hours]
        gen_df['solar_export'] = [pyo.value(model.solar_to_grid[h, location]) for h in hours]
        gen_df['wind_export'] = [pyo.value(model.wind_to_grid[h, location]) for h in hours]
        
        # Create stacked plot
        plt.figure(figsize=figsize)

        stack_cols = ['solar', 'wind', 'geo_45'] + plant_columns + [
            'grid_import', 'storage_discharge', 'solar_curtailed', 
            'wind_curtailed', 'solar_export', 'wind_export'
        ]
        
        labels = ['Solar', 'Wind', 'Geo 45'] + [p.split('_')[0].upper() + ' Gen' for p in plant_columns] + [
            'Grid Import', 'Storage Discharge', 'Solar Curtailed',
            'Wind Curtailed', 'Solar to Grid', 'Wind to Grid'
        ]
        
        colors = ['gold', 'skyblue', 'green'] + list(plt.cm.Set1(range(len(plant_columns)))) + [
            'purple', 'pink', 'gray', 'orange', 'teal', 'lime'
        ]
        
        # Stack plot
        plt.stackplot(
            gen_df['hour'],
            *[gen_df[col] for col in stack_cols],
            labels=labels,
            colors=colors[:len(stack_cols)],
            alpha=0.8
        )
        
        # Plot load line
        plt.plot(hours, load, 'k--', linewidth=2, label='Load')
        
        plt.xlabel('Hour', fontweight='bold', fontsize=16)
        plt.xlim(min(hours), max(hours))
        plt.ylabel('Generation (MW)', fontweight='bold', fontsize=16)
        #plt.title(f'Optimized Hourly Generation Dispatch for Location {location}')
        plt.legend(bbox_to_anchor=(0.5, -0.15),  # center below the axes
                    loc='upper center', 
                    ncol=5,
                    borderaxespad=1.1,          # reduces space between axis and legend
                    frameon=True,               # optional: makes legend readable
                    fontsize=14, 
                    handlelength=1.5,
                    handleheight=1.5,
                    title_fontsize=16)
        plt.grid(alpha=0.3)
        #plt.rcParams.update({'font.size': 12})
        plt.tight_layout()
        print(f'Optimized Hourly Generation Dispatch for Location {location}')
        
        # Save plot
        plt.savefig(self.output_dir / f'generation_dispatch_loc_{location}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save generation data
        gen_df['load'] = load
        gen_df.to_csv(self.output_dir / f'generation_dispatch_data_loc_{location}.csv', index=False)
        
        return gen_df
    
    def create_all_generation_dispatch_plots(self, model, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create and save generation dispatch plots for all selected locations.

        Args:
            model: Solved Pyomo model
            figsize: Figure size
        """
        print("Creating generation dispatch plots for all selected locations...")

        # Identify selected locations (where x[loc] = 1)
        selected_locs = [loc for loc in model.LOCATIONS if pyo.value(model.x[loc]) > 0.5]

        if not selected_locs:
            print("No selected locations found for dispatch plots.")
            return

        # Loop through each selected location
        for loc in selected_locs:
            print(f"→ Plotting generation dispatch for location {loc}...")
            try:
                self.create_generation_dispatch_plot(model, location=loc, figsize=figsize)
                plt.savefig(self.output_dir / f'generation_dispatch_location_final_{loc}.png', dpi=300, bbox_inches='tight')

            except Exception as e:
                print(f"⚠️  Skipping location {loc} due to error: {e}")

    
    def create_cost_breakdown_chart(self, location_df: pd.DataFrame, 
                                   top_n: int = 10,
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create cost breakdown chart for top N locations.
        
        Args:
            location_df: Location breakdown DataFrame
            top_n: Number of top locations to show
            figsize: Figure size
        """
        print(f"Creating cost breakdown chart for top {top_n} locations...")
        
        # Get top N locations by lowest cost
        top_locations = location_df.nsmallest(top_n, 'objective')
        
        # Cost categories for stacked bar
        cost_categories = ['Variable Cost', 'Grid Cost', 'Capital Cost', 'Fixed Cost', 
                          'Transmission Cost', 'Telecom Cost', 'Water Cost', 'Curtail Cost']
        cost_categories = [col for col in cost_categories if col in top_locations.columns]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=figsize)
        
        bottom = np.zeros(len(top_locations))
        colors = plt.cm.Set3(np.linspace(0, 1, len(cost_categories)))
        
        for i, category in enumerate(cost_categories):
            ax.bar(range(len(top_locations)), top_locations[category], 
                   bottom=bottom, label=category.replace('_', ' ').title(), 
                   color=colors[i])
            bottom += top_locations[category]
        
        # Subtract export credit
        if 'export_credit' in top_locations.columns:
            ax.bar(range(len(top_locations)), -top_locations['export_credit'],
                   bottom=bottom, label='Export Credit (Revenue)', 
                   color='lightgreen', alpha=0.7)
        
        ax.set_xlabel('Location Rank', fontweight='bold')
        ax.set_ylabel('Cost ($)', fontweight='bold')
        ax.set_title(f'Cost Breakdown for Top {top_n} Locations')
        ax.set_xticks(range(len(top_locations)))
        ax.set_xticklabels([f"{i+1}\n({int(loc)})" for i, loc in enumerate(top_locations['location'])])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        #plt.rcParams.update({'font.size': 14})
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_breakdown_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_top_n_locations(self, model, location_df: pd.DataFrame, 
                           supply_data: pd.DataFrame,
                           county_shapefile_path: str,
                           top_n: int = 10,
                           exclude_states: List[str] = ['02', '15', '60', '66', '69', '72', '78'],
                           figsize: Tuple[int, int] = (12, 8)) -> pd.DataFrame:
        """
        Plot and print information about top N locations by lowest cost.
        
        Args:
            model: Solved Pyomo model
            location_df: Location breakdown DataFrame with costs
            supply_data: Supply data with location coordinates
            state_shapefile_path: Path to state shapefile for base map
            top_n: Number of top locations to display
            figsize: Figure size for plot
            
        Returns:
            DataFrame with top N locations
        """
        print(f"\nTOP {top_n} LOCATIONS BY LOWEST COST")
        print("=" * 60)
        
        use_plants = self._has_plants(model)

        # Get top N locations by lowest objective value
        top_locations = location_df.nsmallest(top_n, 'objective').copy()
        
        print('columns in top_locations:')
        #print(top_locations.columns)   

        #Columns of top_locations
        '''
        Index(['location', 'objective', 'Variable Cost', 'Grid Cost', 'Curtail Cost',
       'Export Credit', 'Water Cost', 'Capital Cost', 'Fixed Cost',
       'Transmission Cost', 'Telecom Cost', 'Solar Gen', 'Wind Gen', 'Geo Gen',
       'Storage Discharge', 'Grid Import', 'Solar Export', 'Wind Export'],
        dtype='object')
        '''
        
        # Merge with supply data to get coordinates
        top_locations_data = top_locations.merge(
            supply_data[['location', 'latitude', 'longitude', 'capacity_solar', 
                        'capacity_wind', 'capacity_geo', 'trans_dist', 'telecom_dist']],
            on='location',
            how='left'
        )
        
        # Print detailed information
        for i, row in top_locations_data.iterrows():
            rank = top_locations_data.index.get_loc(i) + 1
            print(f"\nRank {rank}: Location {row['location']}")
            print(f"  Total Cost: ${row['objective']:,.2f}")
            print(f"  Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})")
            print(f"  Renewable Capacity: Solar={row['capacity_solar']:.1f} MW, "
                  f"Wind={row['capacity_wind']:.1f} MW, Geo={row['capacity_geo']:.1f} MW")
            
            # Print generation by type
            gen_str = f"  Generation: Solar={row['Solar Gen']:.1f} MWh, Wind={row['Wind Gen']:.1f} MWh"
            if use_plants:
                for p in model.PLANTS:
                    gen_str += f", {p.upper()}={row[f'{p}_gen']:.1f} MWh"
            gen_str += f", Grid={row['Grid Import']:.1f} MWh"
            print(gen_str)
            
            print(f"  Infrastructure: Trans Dist={row['trans_dist']:.1f} mi, "
                  f"Telecom Dist={row['telecom_dist']:.1f} mi")
        
        # Check if selected location is in top N
        selected_locs = [loc for loc in model.LOCATIONS if pyo.value(model.x[loc]) > 0.5]
        if selected_locs:
            selected_loc = selected_locs[0]
            if selected_loc in top_locations_data['location'].values:
                print(f"\n*** SELECTED LOCATION {selected_loc} IS IN TOP {top_n} ***")
            else:
                print(f"\n*** SELECTED LOCATION {selected_loc} IS NOT IN TOP {top_n} ***")
        
        # Create geographic plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Load and plot state boundaries as base map
        try:
            county_shape_data = gpd.read_file(county_shapefile_path)
            #county_shape_data['STATEFP'] = county_shape_data['STATEFP'].astype(str)
            county_shape_data = county_shape_data[~county_shape_data['STATEFP'].isin(exclude_states)]
            county_shape_data.plot(ax=ax, edgecolor='black', color='white', linewidth=0.5)
        except Exception as e:
            print(f"Could not load state shapefile: {e}")
        
        # Plot top N locations
        scatter = ax.scatter(
            top_locations_data['longitude'], 
            top_locations_data['latitude'],
            c=top_locations_data['objective'],
            s=200,
            cmap='RdYlGn_r',  # Red (expensive) to Green (cheap)
            edgecolors='black',
            linewidth=1.5,
            alpha=0.8,
            zorder=5
        )
        
        # Highlight the selected location if it exists
        if selected_locs:
            selected_data = top_locations_data[top_locations_data['location'] == selected_locs[0]]
            if not selected_data.empty:
                ax.scatter(
                    selected_data['longitude'],
                    selected_data['latitude'],
                    s=400,
                    marker='*',
                    c='gold',
                    edgecolors='black',
                    linewidth=2,
                    zorder=10,
                    label=f'Selected: {selected_locs[0]}'
                )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total Cost ($)', rotation=270, labelpad=20)
        
        # Add labels for top 3 locations
        for idx, row in top_locations_data.head(3).iterrows():
            rank = top_locations_data.index.get_loc(idx) + 1
            ax.annotate(
                f"#{rank}: {int(row['location'])}",
                xy=(row['longitude'], row['latitude']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=16,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_title(f'Top {top_n} Locations by Lowest Cost', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if selected_locs:
            ax.legend(loc='best')
        
        #plt.rcParams.update({'font.size': 14})
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'top_{top_n}_locations_map.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save top locations data
        output_path = self.output_dir / f'top_{top_n}_locations.csv'
        top_locations_data.to_csv(output_path, index=False)
        print(f"\nSaved top {top_n} locations data to {output_path}")
        
        return top_locations_data

    
    def generate_comprehensive_report(self, model, cost_params: Dict, 
                                    supply_data: pd.DataFrame,
                                    county_shapefile_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive visualization report.
        
        Args:
            model: Solved Pyomo model
            cost_params: Cost parameters
            supply_data: Original supply data
            county_shapefile_path: Path to county shapefile for maps
            
        Returns:
            Dictionary with all analysis results
        """
        print("Generating comprehensive visualization report...")
        
        # 1. Calculate location breakdown
        location_df = self.calculate_location_breakdown(model, cost_params)
        
        # Save location breakdown
        output_path = self.output_dir / "location_objectives.csv"
        location_df.to_csv(output_path, index=False)
        print(f"Saved location breakdown to {output_path}")
        
        # 2. Correlation analysis
        merged_df, correlations = self.create_correlation_analysis(location_df, supply_data)
        
        # 3. Generate plots
        self.plot_correlation_analysis(correlations)
        
        if county_shapefile_path and Path(county_shapefile_path).exists():
            self.create_county_choropleth(location_df, county_shapefile_path)
        
        self.create_generation_dispatch_plot(model) #, location = 12077)
        self.create_cost_breakdown_chart(location_df)
        top_n_data = self.plot_top_n_locations(model, location_df, supply_data, county_shapefile_path)
        #self.create_all_generation_dispatch_plots(model)

        
        # 4. Summary statistics
        summary_stats = {
            'total_locations_analyzed': len(location_df),
            'best_location': location_df.loc[location_df['objective'].idxmin()],
            'cost_range': {
                'min': location_df['objective'].min(),
                'max': location_df['objective'].max(),
                'std': location_df['objective'].std()
            },
            'generation_totals': {
                'total_solar': location_df['Solar Gen'].sum(),
                'total_wind': location_df['Wind Gen'].sum(),
                'total_grid': location_df['Grid Import'].sum()
            }
        }
        
        return {
            'location_breakdown': location_df,
            'merged_data': merged_df,
            'correlations': correlations,
            'summary_stats': summary_stats,
            'top_n_locations': top_n_data
        }


# Standalone function for easy integration
def visualize_optimization_results(model, cost_params: Dict, supply_data: pd.DataFrame,
                                 output_dir: str = "output_files",
                                 county_shapefile_path: str = None) -> Dict[str, Any]:
    """
    Standalone function to visualize optimization results.
    
    Args:
        model: Solved Pyomo model
        cost_params: Cost parameters used in optimization
        supply_data: Original supply data DataFrame
        output_dir: Output directory for plots and files
        county_shapefile_path: Path to county shapefile for choropleth maps
        
    Returns:
        Dictionary with analysis results
    """
    visualizer = OptimizationVisualizer(output_dir)
    return visualizer.generate_comprehensive_report(
        model, cost_params, supply_data, county_shapefile_path
    )

# Example usage
if __name__ == "__main__":
    # Call after solving your optimization model
    
    # Assuming you have:
    # - model: solved Pyomo model
    # - cost_params: your cost parameter dictionaries  
    # - supply_data: your original supply data DataFrame

    file_paths = {
        'state_shapefile': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2022_us_state_20m/cb_2022_us_state_20m.shp',
        'supply_data': '/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water_clim.csv',
        'merged_cf': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/merged_hourly_solar_wind_cf.csv',
        'demand_data': 'fake_demand.csv',
        'county2zone': 'CountyMaps/county2zone.csv',
        'hierarchy': 'CountyMaps/hierarchy.csv',
        'electric_prices': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/electric_prices.csv',
        'water_risk': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/water_risk.gpkg',
        'county_shapefile': '/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'
    }
 
    processor, model_dictionaries = process_data_pipeline(
        file_paths=file_paths,
        pue_climate_dict=pue_climate_region_5,
        wue_climate_dict=wue_climate_region_5,
        trans_mult_dict=trans_mult_dict,
        telecom_cost_dict=telecom_cost,
        min_capacity=200,
        state_filter=None,
        max_water_risk=5.0,
        #county_filter = [22075, 22087, 22089, 22095, 36047, 36061, 44005, 50013, 51650, 51710]
    )

    # 5. Run the optimization
    opt_model, solution = run_datacenter_optimization(
        model_dictionaries=model_dictionaries,
        config=config,
        cost_params=cost_params,
        trans_rating = trans_rating,
        trans_cost = trans_cost,
        solver_name='gurobi',
        processor=processor,
        storage_system = StorageTemplates.create_lithium_ion("my_battery"),
        plant_systems = {
        'smr': PlantTemplates.create_smr_plant("my_smr", 200000)
        }
    )
    
    results = visualize_optimization_results(
        model=opt_model.model,  # Your solved model
        cost_params=cost_params,
        supply_data=processor.processed_data['supply_data'], 
        output_dir="optimization_output",
        county_shapefile_path=file_paths['county_shapefile']
    )
    
    print("Visualization complete!")
    print(f"Best location: {results['summary_stats']['best_location']['location']}")
    print(f"Best cost: ${results['summary_stats']['best_location']['objective']:,.2f}")

    # Print detailed summary
    #print_detailed_summary(viz_results, solution)
