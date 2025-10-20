import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
import ast
import string
import matplotlib as mpl

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.family'] = 'Times New Roman'

# Load results
#df = pd.read_csv('results/all_results_ren_pen_v1.csv')
df = pd.read_csv('results/all_results.csv')

# Filter successful runs only
df_success = df[df['status'] == 'success'].copy()

print(f"Total experiments: {len(df)}")
print(f"Successful: {len(df_success)}")

# These are always constant (not swept)
constant_cols = ['exp_name', 'status', 'solver_status', 'all_locations', 'state_filter',
                 'planning_horizon', 'single_location', 'solver_name', 'solver_time_limit', 
                 'solver_mip_gap']

# These are output metrics (not inputs) # Automatically detect sweep parameters
output_cols = ['objective_value', 'num_locations', 'total_solar_capacity', 
               'total_wind_capacity', 'total_geo_capacity', 'total_renewable_capacity',
               'total_renewable_generation', 'total_grid_purchases', 'total_load',
               'renewable_utilization', 'grid_dependence', 'avg_solar_per_location',
               'avg_wind_per_location']

# Find sweep parameters (columns with more than 1 unique value)
sweep_params = []
for col in df_success.columns:
    if col not in constant_cols + output_cols:
        unique_vals = df_success[col].dropna().unique()
        if len(unique_vals) > 1:
            sweep_params.append(col)

print(f"\nDetected sweep parameters: {sweep_params}")
print("\nSweep parameter values:")
for param in sweep_params:
    print(f"  {param}: {sorted(df_success[param].unique())}")

df_success[sweep_params] = df_success[sweep_params].apply(pd.to_numeric, errors='ignore')


# 1. Create line plots for each sweep parameter
numeric_params = [p for p in sweep_params if pd.api.types.is_numeric_dtype(df_success[p])]

n_params = len(sweep_params)
if not numeric_params:
    print("No numeric sweep parameters detected.")
else:
    # Create subplots (one per parameter)
    fig, axes = plt.subplots(len(numeric_params), 1, figsize=(10, 6 * len(numeric_params)))

    if len(numeric_params) == 1:
        axes = [axes]  # make iterable if only one

    for idx, param in enumerate(numeric_params):
        ax1 = axes[idx]
        ax2 = ax1.twinx()  # second y-axis

        df_sorted = df_success.sort_values(param)

        # Plot cost (left axis)
        line1, = ax1.plot(df_sorted[param],
                          df_sorted['objective_value'] / 1e9,
                          color='steelblue',
                          marker='o',
                          linewidth=2,
                          label='Total Cost (Billions $)')

        # Plot renewable utilization (right axis)
        line2, = ax2.plot(df_sorted[param],
                          df_sorted['renewable_utilization'] * 100,
                          color='green',
                          marker='s',
                          linewidth=2,
                          label='Renewable Utilization (%)')

        # Labeling and axes styling
        ax1.set_xlabel(param.replace('_', ' ').title(), fontsize=15)
        ax1.set_ylabel('Total Cost (Billion $)',  fontsize=14)
        ax2.set_ylabel('VG Utilization (%)', fontsize=14)
        ax1.tick_params(axis='y')
        ax2.tick_params(axis='y')

        # Title & grid
        #ax1.set_title(f'{param.replace("_", " ").title()} Sweep Results', fontsize=16, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax2.set_ylim([0, 100])

        # Combine legends
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=2, fontsize=13, frameon=True)
        

    plt.tight_layout()
    plt.savefig('results/sweep_combined_dual_axis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nSaved: results/sweep_combined_dual_axis.png")


if n_params == 0:
    print("No sweep parameters detected - all experiments have same configuration")
else:
    # Create subplots - 2 metrics per sweep parameter
    fig, axes = plt.subplots(n_params, 2, figsize=(15, 5 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param in enumerate(numeric_params):
        # Plot 1: Cost vs sweep parameter
        ax = axes[idx, 0]
        
        # Group by other parameters and plot separately
        other_params = [p for p in numeric_params if p != param]
        
        if other_params:
            # Group by combinations of other parameters
            grouped = df_success.groupby(other_params)
            for name, group in grouped:
                label = ', '.join([f"{p}={v}" for p, v in zip(other_params, name if len(other_params) > 1 else [name])])
                group_sorted = group.sort_values(param)
                ax.plot(group_sorted[param], group_sorted['objective_value'] / 1e9,
                       marker='o', label=label, linewidth=2)
        else:
            # Only one sweep parameter
            df_sorted = df_success.sort_values(param)
            ax.plot(df_sorted[param], df_sorted['objective_value'] / 1e9,
                   marker='o', linewidth=2, color='steelblue')
        
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize = 16)
        ax.set_ylabel('Total Cost (Billion $)', fontsize = 16)
        #ax.set_title(f'Cost vs {param.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Renewable utilization vs sweep parameter
        ax = axes[idx, 1]
        
        if other_params:
            grouped = df_success.groupby(other_params)
            for name, group in grouped:
                label = ', '.join([f"{p}={v}" for p, v in zip(other_params, name if len(other_params) > 1 else [name])])
                group_sorted = group.sort_values(param)
                ax.plot(group_sorted[param], group_sorted['renewable_utilization'] * 100,
                       marker='o', label=label, linewidth=2)
        else:
            df_sorted = df_success.sort_values(param)
            ax.plot(df_sorted[param], df_sorted['renewable_utilization'] * 100,
                   marker='o', linewidth=2, color='green')
        
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize = 16)
        ax.set_ylabel('Renewable Utilization (%)', fontsize = 16)
        #ax.set_title(f'Renewable Utilization vs {param.replace("_", " ").title()}', 
        #           fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/sweep_parameters.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/sweep_parameters.png")
    plt.show()


# 2. Heatmap for 2-parameter sweeps
if len(sweep_params) == 2:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = [
        ('objective_value', 'Total Cost (B$)', 1e9),
        ('renewable_utilization', 'Renewable Util. (%)', 0.01),
        ('num_locations', 'Num Locations', 1)
    ]
    
    for idx, (metric, label, scale) in enumerate(metrics_to_plot):
        ax = axes[idx]
        pivot = df_success.pivot_table(
            values=metric, 
            index=sweep_params[0], 
            columns=sweep_params[1]
        )
        
        sns.heatmap(pivot / scale, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': label})
        ax.set_title(f'{label} Heatmap', fontweight='bold')
        ax.set_xlabel(sweep_params[1].replace('_', ' ').title(), fontsize = 16)
        ax.set_ylabel(sweep_params[0].replace('_', ' ').title(), fontsize = 16)
    
    plt.tight_layout()
    plt.savefig('results/sweep_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: results/sweep_heatmap.png")
    plt.show()

# 3. Capacity Mix Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Create configuration labels
config_labels = []
for _, row in df_success.iterrows():
    label_parts = [f"{p[:4]}={row[p]:.0f}" for p in sweep_params[:2]]  # Use first 2 params
    config_labels.append('\n'.join(label_parts) if len(label_parts) <= 2 else 
                        ', '.join(label_parts))

# Plot 1: Capacity mix by configuration
ax = axes[0]
x = range(len(df_success))
width = 0.6

ax.bar(x, df_success['total_solar_capacity'], width, label='Solar', alpha=0.8)
ax.bar(x, df_success['total_wind_capacity'], width, 
       bottom=df_success['total_solar_capacity'], label='Wind', alpha=0.8)
ax.bar(x, df_success['total_geo_capacity'], width,
       bottom=df_success['total_solar_capacity'] + df_success['total_wind_capacity'],
       label='Geothermal', alpha=0.8)

ax.set_xlabel('Configuration', fontsize=16)
ax.set_ylabel('Total Capacity (MW)', fontsize=16)
#ax.set_title('Renewable Capacity Mix by Configuration', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(config_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Generation vs Grid purchases
ax = axes[1]
ax.bar(x, df_success['total_renewable_generation'], width, 
       label='Renewable Generation', alpha=0.8, color='green')
ax.bar(x, df_success['total_grid_purchases'], width,
       bottom=df_success['total_renewable_generation'],
       label='Grid Purchases', alpha=0.8, color='red')

ax.set_xlabel('Configuration', fontsize = 16)
ax.set_ylabel('Energy (MWh)', fontsize = 16)
#ax.set_title('Energy Supply Mix: Renewable vs Grid', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(config_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/capacity_mix.png', dpi=300, bbox_inches='tight')
print("Saved: results/capacity_mix.png")
plt.show()

# ============================================
# 4. Correlation Heatmap
# ============================================
fig, ax = plt.subplots(figsize=(12, 10))

# Select all numeric columns
numeric_cols = df_success.select_dtypes(include=[np.number]).columns
# Remove ID-like columns
exclude = ['min_location', 'planning_horizon', 'solver_time_limit', 'solver_mip_gap']
metrics = [col for col in numeric_cols if col not in exclude]

corr_matrix = df_success[metrics].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix of All Metrics', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: results/correlation_heatmap.png")
plt.show()

# ============================================
# 5. Summary Statistics
# ============================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if sweep_params:
    summary_stats = df_success.groupby(sweep_params).agg({
        'objective_value': 'first',
        'num_locations': 'first',
        'renewable_utilization': 'first',
        'grid_dependence': 'first',
        'total_renewable_capacity': 'first',
        'total_solar_capacity': 'first',
        'total_wind_capacity': 'first'
    }).round(2)
    
    print(summary_stats)
    summary_stats.to_csv('results/summary_statistics.csv')
    print("\nSaved: results/summary_statistics.csv")
else:
    print("Only one configuration - no grouping performed")
    print(df_success[['objective_value', 'num_locations', 'renewable_utilization', 
                      'grid_dependence', 'total_renewable_capacity']])

# ============================================
# 6. Pareto Frontier (if relevant)
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(df_success['objective_value'] / 1e9, 
                     df_success['renewable_utilization'] * 100,
                     c=df_success['num_locations'], 
                     s=200, alpha=0.6, cmap='viridis', 
                     edgecolors='black', linewidth=1.5)

# Add labels for each point
for idx, row in df_success.iterrows():
    label_parts = [f"{p[:4]}={row[p]:.0f}" for p in sweep_params[:2]]
    ax.annotate(', '.join(label_parts), 
               (row['objective_value'] / 1e9, row['renewable_utilization'] * 100),
               fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Total Cost (Billion $)', fontsize = 16)
ax.set_ylabel('Renewable Utilization (%)', fontsize = 16)
#ax.set_title('Cost vs Renewable Utilization Trade-off', fontweight='bold')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Locations', fontsize=16)

plt.tight_layout()
plt.savefig('results/tradeoff_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: results/tradeoff_analysis.png")
plt.show()

print("\n" + "="*80)
print("Visualization complete! Check the 'results/' directory for all plots.")
print("="*80)


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import ast

def plot_experiment_locations_on_map(results_csv: str,
                                     supply_data: str,
                                     shapefile_path: str,
                                     exclude_states=['AK', 'HI', 'PR', 'GU', 'VI', 'MP', 'AS'],
                                     figsize=(12, 8)):

    # Load results and supply data
    results = pd.read_csv(results_csv)
    supply_data = pd.read_csv(supply_data)
    states = gpd.read_file(shapefile_path)
    states = states[~states['STUSPS'].isin(exclude_states)]
    print(supply_data.columns)

    # Convert results['all_locations'] from string to list
    def safe_parse(x):
        try:
            return ast.literal_eval(x)
        except:
            return []
    results['all_locations'] = results['all_locations'].apply(safe_parse)
    print(results.columns)
    
    # Filter for successful runs only
    results = results[results['status'] == 'success']
    print(f"Plotting {len(results)} experiment cases...")

    # Create letter labels (A, B, C, ...)
    case_labels = list(string.ascii_uppercase)[:len(results)]
    results = results.reset_index(drop=True)

    # Print mapping of labels to experiment names
    print("\nCase Label Mapping:")
    for label, row in zip(case_labels, results.itertuples()):
        print(f"  {label}: {row.exp_name}")

    # Set up figure and base map
    fig, ax = plt.subplots(figsize=figsize)
    states.plot(ax=ax, edgecolor='black', color='white', linewidth=0.6)

    # Assign unique colors and labels
    n_cases = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cases))
    case_labels = [chr(65 + i) for i in range(n_cases)]  # ['A', 'B', 'C', ...]

    supply_data = supply_data.dropna(subset=['FIPS']).copy() 
    supply_data['FIPS'] = supply_data['FIPS'].astype(int)
    supply_data_unique = supply_data.drop_duplicates(subset=['FIPS'])

    # Plot each case
    for i, (row, label) in enumerate(zip(results.itertuples(), case_labels)):
        locs = list(set(row.all_locations)) 
        #locs = row.all_locations
        case_name = row.exp_name
        
        loc_df = supply_data_unique[supply_data_unique['FIPS'].isin(locs)]

        ax.scatter(
            loc_df['longitude'],
            loc_df['latitude'],
            color=colors[i],
            s=60,
            label=f"{label}",
            alpha=0.9,
            edgecolors='black',
            linewidth=0.3
        )

        print(f"Case {label}: {case_name}")
        print(f"Case {label}: {case_name}, expected {len(locs)} FIPS, plotted {len(loc_df)} points")

    
    '''
    # Choose colors for each experiment
    cmap = plt.cm.get_cmap('tab10', len(results))  # distinct colors
    scatter_handles = []

    # Define distinct colors for each case
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    #case_labels = [chr(65+i) for i in range(len(results))]
    
    # Plot each experiment’s selected locations
    for i, row in enumerate(results.itertuples()):
        locs = row.all_locations
        #case_name = row.exp_name
        label = case_labels[i]

        # Merge with supply data to get coordinates
        loc_df = supply_data[supply_data['FIPS'].isin(locs)]
        
        sc = ax.scatter(
            loc_df['longitude'],
            loc_df['latitude'],
            color = colors[i],
            s=60,
            label=f"{label}",
            alpha=0.8,
            edgecolors='black',
            linewidth=0.3
        )
        scatter_handles.append(sc)
    '''

    # Styling
    ax.axis('off')
    ax.set_xlim(supply_data['longitude'].min() - 1, supply_data['longitude'].max() + 1)
    ax.set_ylim(supply_data['latitude'].min() - 1, supply_data['latitude'].max() + 1)
    #ax.set_title("Selected Datacenter Locations Across Experiments", fontsize=14, fontweight='bold')
    #ax.set_xlabel("Longitude", fontweight='bold', fontsize = 16)
    #ax.set_ylabel("Latitude", fontweight='bold', fontsize = 16)

    # Legend
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1, 1),
        ncol=1,
        title="Case",
        fontsize=14,
        title_fontsize=16,
        markerscale=1.2
    )
    for text in legend.get_texts():
        text.set_fontweight('bold')
    legend.get_frame().set_linewidth(0.6)
    
    plt.tight_layout()
    plt.show()

    plt.savefig('results/dot_locations.png', dpi=300, bbox_inches='tight')
    print("Saved: results/dot_locations.png")

plot_experiment_locations_on_map(
    results_csv="results/all_results.csv",
    supply_data="/Users/maria/Documents/Research/deloitte-proj/telecom-data/supply_data_lat_lon_water_clim.csv",
    shapefile_path="/Users/maria/Documents/Research/deloitte-proj/deloitte-data/cb_2022_us_state_20m/cb_2022_us_state_20m.shp"
)