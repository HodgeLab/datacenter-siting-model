# How to use

## Components

### 1. Plant Module (`plant.py`)
Generic power plant modeling for dispatchable generation resources.

### 2. Storage Module (`storage.py`)
Energy storage system modeling for various storage technologies.

### 3. Constraints Module (`constraints.py`)
Pre-built Pyomo constraint rules for common energy system constraints.

## Requirements

Needs pyomo and numpy which you probably already have.

## Quick Start

### Example 1: Using Plant Components

```python
from plant import Plant, PlantTemplates
import pyomo.environ as pyo

# Create plants using templates
gas_plant = PlantTemplates.create_gas_turbine(
    name="GT1", # refers to gas turbine 1
    capacity_kw=5000
)

diesel_gen = PlantTemplates.create_diesel_generator(
    name="DG1", # refers to diesel generator 1
    capacity_kw=1000
)

# Whatever plant you want
custom_plant = Plant(
    name="CustomPlant",
    plant_type="combined_cycle", # This is only for indexing, not used to fill any parameters.
    capacity_kw=10000,
    capex_per_kw=1200,
    min_output_fraction=0.4,
    fuel_type="natural_gas",
    fuel_cost_per_unit=3.5,
    fuel_efficiency=0.55
)

# Use in Pyomo model
model = pyo.ConcreteModel()
model.T = pyo.RangeSet(0, 23)  # 24 hours
model.P = pyo.Set(initialize=['GT1', 'DG1'])

# Decision variables
model.plant_output = pyo.Var(model.T, model.P, domain=pyo.NonNegativeReals)
model.plant_online = pyo.Var(model.T, model.P, domain=pyo.Binary)

# Get plant parameters
plant_params = {
    'GT1': gas_plant.get_parameters_dict(),
    'DG1': diesel_gen.get_parameters_dict()
}
```

### Example 2: Using Storage Components

```python
from storage import Storage, StorageTemplates
import pyomo.environ as pyo

# Create storage systems using templates
battery = StorageTemplates.create_lithium_ion(name="Battery1")
pumped_hydro = StorageTemplates.create_pumped_hydro(name="PHS1")

# Or create custom storage
custom_storage = Storage(
    name="CustomESS",
    storage_type="flow_battery",
    capex_per_kwh=350,
    round_trip_efficiency=0.85,
    max_c_rate=0.5,
    max_d_rate=0.5,
    min_soc=0.05,
    max_soc=0.95
)

# Use in Pyomo model
model = pyo.ConcreteModel()
model.T = pyo.RangeSet(0, 167)  # One week
model.S = pyo.Set(initialize=['Battery1', 'PHS1'])

# Decision variables
model.storage_capacity = pyo.Var(model.S, domain=pyo.NonNegativeReals)
model.storage_charge = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
model.storage_discharge = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
model.storage_energy = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)

# Get storage parameters
storage_params = {
    'Battery1': battery.get_parameters_dict(),
    'PHS1': pumped_hydro.get_parameters_dict()
}
```

### Example 3: Using Modular Constraints

```python
from constraints import PlantConstraints, StorageConstraints, SystemConstraints
import pyomo.environ as pyo
import numpy as np

# Create model
model = pyo.ConcreteModel()
model.T = pyo.RangeSet(0, 23)
model.P = pyo.Set(initialize=['Plant1'])
model.S = pyo.Set(initialize=['Storage1'])

# Add variables (as shown above)
# ... (add all necessary variables)

# Load profiles
load_profile = np.array([100, 120, 150, ...])  # kW for each hour

# Add plant constraints
plant_params = {'Plant1': {...}}  # Plant parameters dictionary

model.plant_min_output = pyo.Constraint(
    model.T, model.P,
    rule=PlantConstraints.min_output_rule(plant_params)
)

model.plant_max_output = pyo.Constraint(
    model.T, model.P,
    rule=PlantConstraints.max_output_rule(plant_params)
)

model.plant_ramp_up = pyo.Constraint(
    model.T, model.P,
    rule=PlantConstraints.ramp_up_rule(plant_params)
)

# Add storage constraints
storage_params = {'Storage1': {...}}  # Storage parameters dictionary

model.storage_energy_balance = pyo.Constraint(
    model.T, model.S,
    rule=StorageConstraints.energy_balance_rule(storage_params)
)

model.storage_soc_min = pyo.Constraint(
    model.T, model.S,
    rule=StorageConstraints.min_soc_rule(storage_params)
)

# Add system energy balance
model.energy_balance = pyo.Constraint(
    model.T,
    rule=SystemConstraints.energy_balance_rule(
        load_profile=load_profile,
        has_storage=True,
        has_plants=True
    )
)
```

### Example 4: Complete Microgrid Optimization

```python
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from plant import PlantTemplates
from storage import StorageTemplates
from constraints import (PlantConstraints, StorageConstraints, 
                        RenewableConstraints, SystemConstraints)

def create_microgrid_model():
    """Create a complete microgrid optimization model."""
    
    # Initialize model
    model = pyo.ConcreteModel()
    
    # Time periods (one week)
    hours = 168
    model.T = pyo.RangeSet(0, hours-1)
    
    # Create components
    gas_plant = PlantTemplates.create_gas_turbine("GT1", 5000, hours)
    battery = StorageTemplates.create_lithium_ion("BAT1", hours)
    
    # Sets
    model.P = pyo.Set(initialize=['GT1'])
    model.S = pyo.Set(initialize=['BAT1'])
    
    # Parameters
    load = np.random.uniform(3000, 7000, hours)  # Random load profile
    wind_cf = np.random.uniform(0.1, 0.8, hours)  # Wind capacity factors
    solar_cf = np.maximum(0, np.sin(np.arange(hours) * 2 * np.pi / 24)) * 0.8
    grid_price = np.array([0.10 if h % 24 < 6 or h % 24 > 20 else 0.15 
                          for h in range(hours)])
    
    # Decision Variables
    ## Capacity variables
    model.wind_capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10000))
    model.solar_capacity = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 10000))
    model.storage_capacity = pyo.Var(model.S, domain=pyo.NonNegativeReals, bounds=(0, 20000))
    
    ## Operational variables
    model.wind_power = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.solar_power = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.wind_curtailment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.solar_curtailment = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    
    model.plant_output = pyo.Var(model.T, model.P, domain=pyo.NonNegativeReals)
    model.plant_online = pyo.Var(model.T, model.P, domain=pyo.Binary)
    model.plant_startup = pyo.Var(model.T, model.P, domain=pyo.Binary)
    
    model.storage_charge = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.storage_discharge = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    model.storage_energy = pyo.Var(model.T, model.S, domain=pyo.NonNegativeReals)
    
    # Prepare parameters
    plant_params = {'GT1': gas_plant.get_parameters_dict()}
    storage_params = {'BAT1': battery.get_parameters_dict()}
    
    # Add Constraints
    ## Renewable constraints
    model.wind_gen = pyo.Constraint(
        model.T,
        rule=RenewableConstraints.wind_generation_rule(wind_cf)
    )
    
    model.solar_gen = pyo.Constraint(
        model.T,
        rule=RenewableConstraints.solar_generation_rule(solar_cf)
    )
    
    ## Plant constraints
    model.plant_min = pyo.Constraint(
        model.T, model.P,
        rule=PlantConstraints.min_output_rule(plant_params)
    )
    
    model.plant_max = pyo.Constraint(
        model.T, model.P,
        rule=PlantConstraints.max_output_rule(plant_params)
    )
    
    model.plant_ramp_up = pyo.Constraint(
        model.T, model.P,
        rule=PlantConstraints.ramp_up_rule(plant_params)
    )
    
    model.plant_ramp_down = pyo.Constraint(
        model.T, model.P,
        rule=PlantConstraints.ramp_down_rule(plant_params)
    )
    
    ## Storage constraints
    model.storage_balance_init = pyo.Constraint(
        model.S,
        rule=StorageConstraints.energy_balance_init_rule(storage_params)
    )
    
    model.storage_balance = pyo.Constraint(
        model.T, model.S,
        rule=StorageConstraints.energy_balance_rule(storage_params)
    )
    
    model.storage_min_soc = pyo.Constraint(
        model.T, model.S,
        rule=StorageConstraints.min_soc_rule(storage_params)
    )
    
    model.storage_max_soc = pyo.Constraint(
        model.T, model.S,
        rule=StorageConstraints.max_soc_rule(storage_params)
    )
    
    model.storage_max_charge = pyo.Constraint(
        model.T, model.S,
        rule=StorageConstraints.max_charge_rule(storage_params)
    )
    
    model.storage_max_discharge = pyo.Constraint(
        model.T, model.S,
        rule=StorageConstraints.max_discharge_rule(storage_params)
    )
    
    ## System energy balance
    model.energy_balance = pyo.Constraint(
        model.T,
        rule=SystemConstraints.energy_balance_rule(
            load_profile=load,
            has_storage=True,
            has_plants=True,
            bidirectional_grid=False
        )
    )
    
    # Objective Function - Dummy one to run quickly
    def objective_rule(model):
        capital = (model.wind_capacity * 1500 + 
                  model.solar_capacity * 1000 +
                  sum(model.storage_capacity[s] * 300 for s in model.S))
        
        grid_cost = sum(model.grid_import[t] * grid_price[t] for t in model.T)
        
        fuel_cost = sum(model.plant_output[t, 'GT1'] * 
                       gas_plant.get_fuel_cost_per_kwh() 
                       for t in model.T)
        
        return capital * 0.1 + grid_cost + fuel_cost
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return model

# Solve the model
model = create_microgrid_model()
solver = SolverFactory('ipopt')  # I dont have gurobi so not sure if it works with it
results = solver.solve(model, tee=True)

# Extract results
print(f"Wind Capacity: {pyo.value(model.wind_capacity):.2f} kW")
print(f"Solar Capacity: {pyo.value(model.solar_capacity):.2f} kW")
print(f"Battery Capacity: {pyo.value(model.storage_capacity['BAT1']):.2f} kWh")
print(f"Total Cost: ${pyo.value(model.objective):.2f}")
```

### Example 5: Adding your own constraints

```python
from constraints import HelperFunctions

# Use helper function to add constraints
HelperFunctions.add_constraint_set(
    model=model,
    constraint_class=PlantConstraints,
    method_name='min_uptime_rule',
    rule_args={'min_hours': 4},
    set_name='plant_min_uptime',
    model.T, model.P  # Index sets
)
```