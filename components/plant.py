from typing import Dict, Optional, List, Union
import numpy as np

class Plant:
    """
    Generic power plant model with economic and operational parameters.
    Could be used to modeled any type of plant, but assuming linear cost function :D 
    """
    
    def __init__(
        self,
        name: str,
        plant_type: str,
        capacity_kw: float,
        capex_per_kw: float,
        opex_fraction: float = 0.02,
        min_output_fraction: float = 0.3,
        ramp_rate_per_hour: float = 0.2,
        startup_time_hours: float = 1.0,
        startup_cost_factor: float = 0.05,
        fuel_type: Optional[str] = None,
        fuel_cost_per_unit: Optional[float] = None,
        fuel_efficiency: Optional[float] = None,
        emissions_kg_per_mwh: Optional[Dict[str, float]] = None,
        availability_profile: Optional[np.ndarray] = None,
        time_horizon: int = 8760
    ):
        """
        Initialize a generic power plant.
        
        Args:
            name: Unique identifier for the plant
            plant_type: Type of plant (e.g., 'gas', 'coal', 'nuclear', 'biomass') This is just for indexing purposes, does not affect any paramter.
            capacity_kw: Maximum capacity in kW
            capex_per_kw: Capital cost per kW
            opex_fraction: Annual O&M costs as fraction of CAPEX (default: 0.02)
            min_output_fraction: Minimum output as fraction of capacity (default: 0.3)
            ramp_rate_per_hour: Max rate of change per hour as fraction of capacity (default: 0.2)
            startup_time_hours: Time required to start from cold state (default: 1.0)
            startup_cost_factor: Startup cost as fraction of hourly max output cost (default: 0.05)
            fuel_type: Type of fuel used (optional)
            fuel_cost_per_unit: Cost of fuel per unit (optional)
            fuel_efficiency: Efficiency of converting fuel to electricity (optional)
            emissions_kg_per_mwh: Emissions in kg per MWh by type (optional)
            availability_profile: Hourly availability profile [0-1] (optional)
            time_horizon: Length of simulation period in hours (default: 8760)
        """
        self.name = name
        self.plant_type = plant_type
        self.capacity_kw = capacity_kw
        self.capex_per_kw = capex_per_kw
        self.opex_fraction = opex_fraction
        self.min_output_fraction = min_output_fraction
        self.ramp_rate_per_hour = ramp_rate_per_hour
        self.startup_time_hours = startup_time_hours
        self.startup_cost_factor = startup_cost_factor
        self.fuel_type = fuel_type
        self.fuel_cost_per_unit = fuel_cost_per_unit
        self.fuel_efficiency = fuel_efficiency
        self.time_horizon = time_horizon
        
        self.emissions_kg_per_mwh = emissions_kg_per_mwh or {}
        
        if availability_profile is None:
            self.availability_profile = np.ones(time_horizon)
        else:
            assert len(availability_profile) == time_horizon, "Availability profile must match time horizon"
            self.availability_profile = availability_profile
            
    def get_annual_capex(self, crf: float) -> float:
        """Calculate annualized capital expenditure."""
        return self.capex_per_kw * self.capacity_kw * crf
        
    def get_annual_opex(self) -> float:
        """Calculate annual operational expenditure (excluding fuel)."""
        return self.capex_per_kw * self.capacity_kw * self.opex_fraction
        
    def get_min_output_kw(self) -> float:
        """Get minimum output level in kW."""
        return self.capacity_kw * self.min_output_fraction
        
    def get_max_ramp_kw(self) -> float:
        """Get maximum ramp rate in kW/hour."""
        return self.capacity_kw * self.ramp_rate_per_hour
    
    def get_startup_cost(self) -> float:
        """Get startup cost in $/startup."""
        if self.fuel_cost_per_unit is None:
            return 0.0
        return self.capacity_kw * self.get_fuel_cost_per_kwh() * self.startup_cost_factor
        
    def calculate_fuel_consumption(self, power_output_kw: float) -> Optional[float]:
        """Calculate fuel consumption for given power output."""
        if self.fuel_type is None or self.fuel_efficiency is None:
            return None
        power_output_mw = power_output_kw / 1000.0
        return power_output_mw / self.fuel_efficiency
        
    def calculate_emissions(self, power_output_kw: float) -> Dict[str, float]:
        """Calculate emissions for given power output."""
        if not self.emissions_kg_per_mwh:
            return {}
        power_output_mwh = power_output_kw / 1000.0
        return {
            emission_type: emission_rate * power_output_mwh
            for emission_type, emission_rate in self.emissions_kg_per_mwh.items()
        }
        
    def get_fuel_cost_per_kwh(self) -> float:
        """Get fuel cost per kWh generated."""
        if self.fuel_type is None or self.fuel_cost_per_unit is None:
            return 0.0
        fuel_per_kwh = self.calculate_fuel_consumption(1.0)
        if fuel_per_kwh is None:
            return 0.0
        return fuel_per_kwh * self.fuel_cost_per_unit
        
    def get_parameters_dict(self) -> Dict:
        """Get plant parameters as dictionary for easy integration."""
        return {
            'name': self.name,
            'type': self.plant_type,
            'capacity_kw': self.capacity_kw,
            'min_output_kw': self.get_min_output_kw(),
            'max_ramp_kw': self.get_max_ramp_kw(),
            'fuel_cost_per_kwh': self.get_fuel_cost_per_kwh(),
            'startup_cost': self.get_startup_cost(),
            'availability': self.availability_profile
        }


# Predefined plant templates for common use cases
class PlantTemplates:
    """Factory class for creating common plant types with typical parameters."""
    
    @staticmethod
    def create_gas_turbine(name: str, capacity_kw: float, time_horizon: int = 8760) -> Plant:
        """Create a natural gas turbine plant."""
        return Plant(
            name=name,
            plant_type='gas_turbine',
            capacity_kw=capacity_kw,
            capex_per_kw=800,  # $/kW
            opex_fraction=0.025,
            min_output_fraction=0.25,
            ramp_rate_per_hour=0.5,
            startup_time_hours=0.5,
            fuel_type='natural_gas',
            fuel_cost_per_unit=3.0,  # $/MMBtu
            fuel_efficiency=0.40,  # 40% efficiency
            emissions_kg_per_mwh={'CO2': 450, 'NOx': 0.2},
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_diesel_generator(name: str, capacity_kw: float, time_horizon: int = 8760) -> Plant:
        """Create a diesel generator."""
        return Plant(
            name=name,
            plant_type='diesel',
            capacity_kw=capacity_kw,
            capex_per_kw=500,  # $/kW
            opex_fraction=0.03,
            min_output_fraction=0.20,
            ramp_rate_per_hour=0.8,
            startup_time_hours=0.25,
            fuel_type='diesel',
            fuel_cost_per_unit=4.0,  # $/gallon
            fuel_efficiency=0.35,  # 35% efficiency
            emissions_kg_per_mwh={'CO2': 650, 'NOx': 0.5, 'PM': 0.02},
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_biomass_plant(name: str, capacity_kw: float, time_horizon: int = 8760) -> Plant:
        """Create a biomass power plant."""
        return Plant(
            name=name,
            plant_type='biomass',
            capacity_kw=capacity_kw,
            capex_per_kw=3500,  # $/kW
            opex_fraction=0.04,
            min_output_fraction=0.40,
            ramp_rate_per_hour=0.15,
            startup_time_hours=4.0,
            fuel_type='biomass',
            fuel_cost_per_unit=20.0,  # $/ton
            fuel_efficiency=0.25,  # 25% efficiency
            emissions_kg_per_mwh={'CO2': 0, 'NOx': 0.8},  # CO2 neutral
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_smr_plant(name: str, capacity_kw: float, time_horizon: int = 8760) -> Plant:
        """Create a biomass power plant."""
        return Plant(
            name=name,
            plant_type='smr',
            capacity_kw=capacity_kw,
            capex_per_kw=4000,  # $/kW
            opex_fraction=0.04,
            min_output_fraction=0.30,
            ramp_rate_per_hour=0.02,
            startup_time_hours=6.0,
            fuel_type='nuclear',
            fuel_cost_per_unit=5,  # $/Mwh
            fuel_efficiency=0.35,  # 35% efficiency
            emissions_kg_per_mwh={'CO2': 0, 'NOx': 0},  # neutral
            time_horizon=time_horizon
        )