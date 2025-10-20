import pyomo.environ as pyo
from typing import Dict, List, Optional
import numpy as np


class PlantConstraints:
    """Collection of constraint rules for power plant operation."""
    
    @staticmethod
    def min_output_rule(plant_params: Dict):
        """Create minimum output constraint rule for a plant."""
        def rule(model, t, p):
            min_output = plant_params[p]['min_output_kw']
            return model.plant_output[t, p] >= min_output * model.plant_online[t, p]
        return rule
    
    @staticmethod
    def max_output_rule(plant_params: Dict):
        """Create maximum output constraint rule for a plant."""
        def rule(model, t, p):
            max_output = plant_params[p]['capacity_kw']
            availability = plant_params[p]['availability'][t]
            return model.plant_output[t, p] <= max_output * availability * model.plant_online[t, p]
        return rule
    
    @staticmethod
    def ramp_up_rule(plant_params: Dict):
        """Create ramp-up constraint rule for a plant."""
        def rule(model, t, p):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = plant_params[p]['max_ramp_kw']
            return model.plant_output[t, p] - model.plant_output[t-1, p] <= max_ramp
        return rule
    
    @staticmethod
    def ramp_down_rule(plant_params: Dict):
        """Create ramp-down constraint rule for a plant."""
        def rule(model, t, p):
            if t == 0:
                return pyo.Constraint.Skip
            max_ramp = plant_params[p]['max_ramp_kw']
            return model.plant_output[t-1, p] - model.plant_output[t, p] <= max_ramp
        return rule
    
    @staticmethod
    def startup_rule():
        """Create startup tracking constraint rule."""
        def rule(model, t, p):
            if t == 0:
                return model.plant_startup[t, p] >= model.plant_online[t, p]
            return model.plant_startup[t, p] >= model.plant_online[t, p] - model.plant_online[t-1, p]
        return rule
    
    @staticmethod
    def min_uptime_rule(min_hours: int = 1):
        """Create minimum uptime constraint rule."""
        def rule(model, t, p):
            if t < min_hours:
                return pyo.Constraint.Skip
            # If started, must stay online for min_hours
            return sum(model.plant_online[tau, p] for tau in range(t - min_hours + 1, t + 1)) >= \
                   min_hours * model.plant_startup[t - min_hours + 1, p]
        return rule
    
    @staticmethod
    def min_downtime_rule(min_hours: int = 1):
        """Create minimum downtime constraint rule."""
        def rule(model, t, p):
            if t < min_hours:
                return pyo.Constraint.Skip
            # If shut down, must stay offline for min_hours
            startup_sum = sum(model.plant_startup[tau, p] for tau in range(t - min_hours + 1, t + 1))
            return startup_sum <= 1
        return rule


class StorageConstraints:
    """Collection of constraint rules for energy storage operation."""
    
    @staticmethod
    def energy_balance_init_rule(storage_params: Dict):
        """Create initial energy balance constraint rule."""
        def rule(model, s):
            min_soc = storage_params[s]['min_soc']
            efficiency = storage_params[s]['efficiency']
            return model.storage_energy[0, s] == min_soc * model.storage_capacity[s] + \
                   np.sqrt(efficiency) * model.storage_charge[0, s] - \
                   model.storage_discharge[0, s] / np.sqrt(efficiency)
        return rule
    
    @staticmethod
    def energy_balance_rule(storage_params: Dict):
        """Create energy balance constraint rule for storage."""
        def rule(model, t, s):
            if t == 0:
                return pyo.Constraint.Skip
            self_discharge = storage_params[s]['self_discharge']
            efficiency = storage_params[s]['efficiency']
            return model.storage_energy[t, s] == \
                   (1 - self_discharge) * model.storage_energy[t-1, s] + \
                   np.sqrt(efficiency) * model.storage_charge[t, s] - \
                   model.storage_discharge[t, s] / np.sqrt(efficiency)
        return rule
    
    @staticmethod
    def min_soc_rule(storage_params: Dict):
        """Create minimum state of charge constraint rule."""
        def rule(model, t, s):
            min_soc = storage_params[s]['min_soc']
            return model.storage_energy[t, s] >= min_soc * model.storage_capacity[s]
        return rule
    
    @staticmethod
    def max_soc_rule(storage_params: Dict):
        """Create maximum state of charge constraint rule."""
        def rule(model, t, s):
            max_soc = storage_params[s]['max_soc']
            return model.storage_energy[t, s] <= max_soc * model.storage_capacity[s]
        return rule
    
    @staticmethod
    def max_charge_rule(storage_params: Dict):
        """Create maximum charging power constraint rule."""
        def rule(model, t, s):
            max_c_rate = storage_params[s]['max_c_rate']
            return model.storage_charge[t, s] <= max_c_rate * model.storage_capacity[s]
        return rule
    
    @staticmethod
    def max_discharge_rule(storage_params: Dict):
        """Create maximum discharging power constraint rule."""
        def rule(model, t, s):
            max_d_rate = storage_params[s]['max_d_rate']
            return model.storage_discharge[t, s] <= max_d_rate * model.storage_capacity[s]
        return rule
    
    @staticmethod
    def periodicity_rule():
        """Create periodicity constraint (end state equals start state)."""
        def rule(model, s):
            t_final = model.T.last()
            return model.storage_energy[t_final, s] == model.storage_energy[0, s]
        return rule
    
    @staticmethod
    def simultaneous_charge_discharge_rule():
        """Prevent simultaneous charging and discharging (optional)."""
        def rule(model, t, s):
            # Use big-M method with binary variable
            # Requires adding: model.storage_charging[t, s] as Binary variable
            M = 10000  # Big M value
            return model.storage_charge[t, s] <= M * model.storage_charging[t, s]
        return rule


class RenewableConstraints:
    """Collection of constraint rules for renewable generation."""
    
    @staticmethod
    def wind_generation_rule(wind_profile: np.ndarray):
        """Create wind generation constraint rule."""
        def rule(model, t):
            return model.wind_power[t] + model.wind_curtailment[t] == \
                   model.wind_capacity * wind_profile[t]
        return rule
    
    @staticmethod
    def solar_generation_rule(solar_profile: np.ndarray):
        """Create solar generation constraint rule."""
        def rule(model, t):
            return model.solar_power[t] + model.solar_curtailment[t] == \
                   model.solar_capacity * solar_profile[t]
        return rule
    
    @staticmethod
    def max_renewable_generation_rule(load_profile: np.ndarray, max_fraction: float = 1.0):
        """Limit renewable generation to a fraction of load."""
        def rule(model, t):
            return model.wind_power[t] + model.solar_power[t] <= max_fraction * load_profile[t]
        return rule
    
    @staticmethod
    def renewable_capacity_limit_rule(limit: float, resource: str = 'wind'):
        """Create capacity limit constraint for renewable resource."""
        def rule(model):
            if resource == 'wind':
                return model.wind_capacity <= limit
            elif resource == 'solar':
                return model.solar_capacity <= limit
            else:
                return pyo.Constraint.Skip
        return rule


class GridConstraints:
    """Collection of constraint rules for grid interaction."""
    
    @staticmethod
    def grid_import_limit_rule(limit: float):
        """Create grid import limit constraint rule."""
        def rule(model, t):
            return model.grid_import[t] <= limit
        return rule
    
    @staticmethod
    def grid_export_limit_rule(limit: float):
        """Create grid export limit constraint rule."""
        def rule(model, t):
            return model.grid_export[t] <= limit
        return rule
    
    @staticmethod
    def grid_outage_rule(outage_profile: np.ndarray):
        """Create grid outage constraint rule."""
        def rule(model, t):
            if outage_profile[t]:
                return model.grid_import[t] == 0
            else:
                return pyo.Constraint.Skip
        return rule
    
    @staticmethod
    def grid_export_outage_rule(outage_profile: np.ndarray):
        """Create grid export outage constraint rule."""
        def rule(model, t):
            if outage_profile[t]:
                return model.grid_export[t] == 0
            else:
                return pyo.Constraint.Skip
        return rule
    
    @staticmethod
    def peak_demand_tracking_rule(month_mapping: Dict[int, int]):
        """Track monthly peak demand for demand charge calculation."""
        def rule(model, t):
            month = month_mapping[t]
            return model.grid_import[t] <= model.grid_peak_demand[month]
        return rule
    
    @staticmethod
    def net_metering_rule():
        """Net metering constraint (import and export cannot occur simultaneously)."""
        def rule(model, t):
            # Requires binary variable model.grid_importing[t]
            M = 10000  # Big M value
            return model.grid_import[t] <= M * model.grid_importing[t]
        return rule


class SystemConstraints:
    """Collection of system-level constraint rules."""
    
    @staticmethod
    def energy_balance_rule(load_profile: np.ndarray, has_storage: bool = True, 
                           has_plants: bool = True, bidirectional_grid: bool = True):
        """Create energy balance constraint rule."""
        def rule(model, t):
            generation = model.wind_power[t] + model.solar_power[t]
            
            if bidirectional_grid:
                grid_net = model.grid_import[t] - model.grid_export[t]
            else:
                grid_net = model.grid_import[t]
            
            if has_storage:
                storage_net = sum(model.storage_discharge[t, s] - model.storage_charge[t, s] 
                                for s in model.S)
            else:
                storage_net = 0
            
            if has_plants:
                plant_generation = sum(model.plant_output[t, p] for p in model.P)
            else:
                plant_generation = 0
            
            return load_profile[t] == generation + grid_net + storage_net + plant_generation
        return rule
    
    @staticmethod
    def reliability_rule(min_reserve: float):
        """Ensure minimum spinning reserve for reliability."""
        def rule(model, t):
            available_capacity = (model.wind_capacity * model.wind_profile[t] + 
                                model.solar_capacity * model.solar_profile[t])
            if hasattr(model, 'P'):
                available_capacity += sum(model.plant_online[t, p] * model.plant_capacity[p] 
                                        for p in model.P)
            return available_capacity >= model.load[t] * (1 + min_reserve)
        return rule
    
    @staticmethod
    def emissions_limit_rule(emissions_limit: float, time_period: str = 'annual'):
        """Limit total emissions over a time period."""
        def rule(model):
            # Calculate total emissions from plants
            total_emissions = 0
            for t in model.T:
                for p in model.P:
                    # Simplified - assumes emissions proportional to output
                    total_emissions += model.plant_output[t, p] * model.plant_emissions_rate[p]
            
            if time_period == 'annual':
                return total_emissions <= emissions_limit
            elif time_period == 'monthly':
                return total_emissions <= emissions_limit * 12
            else:
                return pyo.Constraint.Skip
        return rule
    
    @staticmethod
    def renewable_portfolio_standard(rps_fraction: float, load_profile: np.ndarray):
        """Enforce renewable portfolio standard."""
        def rule(model):
            total_load = sum(load_profile)
            total_renewable = sum(model.wind_power[t] + model.solar_power[t] for t in model.T)
            return total_renewable >= rps_fraction * total_load
        return rule


class HelperFunctions:
    """Helper functions for constraint creation."""
    
    @staticmethod
    def create_month_mapping(hours_per_year: int = 8760) -> Dict[int, int]:
        """Create mapping from hour to month index."""
        hours_per_month = hours_per_year // 12
        mapping = {}
        for hour in range(hours_per_year):
            mapping[hour] = hour // hours_per_month
        return mapping
    
    @staticmethod
    def add_constraint_set(model, constraint_class, method_name: str, 
                          rule_args: Dict, set_name: str = None, **kwargs):
        """
        Helper to add a set of constraints to a model.
        This is to define any constraint you want.
        
        Args:
            model: Pyomo ConcreteModel
            constraint_class: Class containing constraint methods
            method_name: Name of the constraint method
            rule_args: Arguments to pass to the rule creation method
            set_name: Name for the constraint set in the model
            **kwargs: Additional arguments for Pyomo Constraint
        """
        rule_method = getattr(constraint_class, method_name)
        rule = rule_method(**rule_args)
        
        if set_name is None:
            set_name = f"{constraint_class.__name__}_{method_name}"
        
        model.add_component(set_name, pyo.Constraint(**kwargs, rule=rule))
        return model
    
    @staticmethod
    def validate_model_components(model, required_vars: List[str], 
                                 required_sets: List[str] = None):
        """
        Validate that model has required variables and sets.
        Wrote this because I could not run your code, so I needed to check if the model had the right stuff.
        
        Args:
            model: Pyomo ConcreteModel
            required_vars: List of required variable names
            required_sets: List of required set names
        
        Returns:
            bool: True if all requirements met
        """
        for var in required_vars:
            if not hasattr(model, var):
                raise ValueError(f"Model missing required variable: {var}")
        
        if required_sets:
            for set_name in required_sets:
                if not hasattr(model, set_name):
                    raise ValueError(f"Model missing required set: {set_name}")
        
        return True