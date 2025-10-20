from typing import Dict, Optional, List
import numpy as np

class Storage:
    """
    Generic energy storage class for various storage technologies.
    Supports batteries, pumped hydro, compressed air, and other storage types.
    """
    
    def __init__(
        self,
        name: str,
        storage_type: str,
        storage_capacity: float = 100, 
        capex_per_kwh: float = 100,
        capex_per_kw: Optional[float] = None,
        opex_fraction: float = 0.02,
        round_trip_efficiency: float = 0.90,
        max_c_rate: float = 0.5,
        max_d_rate: float = 0.5,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        self_discharge_rate: float = 0.001,
        cycle_life: Optional[int] = None,
        degradation_rate: Optional[float] = None,
        time_horizon: int = 8760
    ):
        """
        Initialize an energy storage system.
        
        Args:
            name: Unique identifier for the storage system
            storage_type: Type of storage (e.g., 'lithium_ion', 'flow_battery', 'pumped_hydro')
            capacity: max capacity (e.g. 100 MW)
            capex_per_kwh: Capital cost per kWh of energy capacity
            capex_per_kw: Capital cost per kW of power capacity (optional, for systems with separate power/energy)
            opex_fraction: Annual O&M costs as fraction of CAPEX (default: 0.02)
            round_trip_efficiency: Round-trip efficiency [0-1] (default: 0.90)
            max_c_rate: Maximum charge rate as fraction of capacity (default: 0.5)
            max_d_rate: Maximum discharge rate as fraction of capacity (default: 0.5)
            min_soc: Minimum state of charge [0-1] (default: 0.1)
            max_soc: Maximum state of charge [0-1] (default: 0.9)
            self_discharge_rate: Self-discharge rate per hour [0-1] (default: 0.001)
            cycle_life: Number of cycles before replacement (optional)
            degradation_rate: Annual capacity degradation rate (optional)
            time_horizon: Length of simulation period in hours (default: 8760)
        """
        self.name = name
        self.storage_type = storage_type
        self.storage_capacity = storage_capacity
        self.capex_per_kwh = capex_per_kwh
        self.capex_per_kw = capex_per_kw
        self.opex_fraction = opex_fraction
        self.efficiency = round_trip_efficiency
        self.max_c_rate = max_c_rate
        self.max_d_rate = max_d_rate
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.self_discharge = self_discharge_rate
        self.cycle_life = cycle_life
        self.degradation_rate = degradation_rate
        self.time_horizon = time_horizon
        
        # Calculate derived parameters
        self.charge_efficiency = np.sqrt(self.efficiency)
        self.discharge_efficiency = np.sqrt(self.efficiency)
        self.usable_capacity_fraction = self.max_soc - self.min_soc
        
    def get_annual_capex(self, crf: float, energy_capacity_kwh: float, 
                        power_capacity_kw: Optional[float] = None) -> float:
        """Calculate annualized capital expenditure."""
        energy_cost = self.capex_per_kwh * energy_capacity_kwh * crf
        power_cost = 0
        if self.capex_per_kw is not None and power_capacity_kw is not None:
            power_cost = self.capex_per_kw * power_capacity_kw * crf
        return energy_cost + power_cost
        
    def get_annual_opex(self, energy_capacity_kwh: float, 
                       power_capacity_kw: Optional[float] = None) -> float:
        """Calculate annual operational expenditure."""
        energy_cost = self.capex_per_kwh * energy_capacity_kwh * self.opex_fraction
        power_cost = 0
        if self.capex_per_kw is not None and power_capacity_kw is not None:
            power_cost = self.capex_per_kw * power_capacity_kw * self.opex_fraction
        return energy_cost + power_cost
        
    def get_max_charge_power(self, energy_capacity_kwh: float) -> float:
        """Get maximum charging power for given energy capacity."""
        return self.max_c_rate * energy_capacity_kwh
        
    def get_max_discharge_power(self, energy_capacity_kwh: float) -> float:
        """Get maximum discharging power for given energy capacity."""
        return self.max_d_rate * energy_capacity_kwh
        
    def get_min_energy(self, energy_capacity_kwh: float) -> float:
        """Get minimum stored energy for given capacity."""
        return self.min_soc * energy_capacity_kwh
        
    def get_max_energy(self, energy_capacity_kwh: float) -> float:
        """Get maximum stored energy for given capacity."""
        return self.max_soc * energy_capacity_kwh
        
    def calculate_energy_change(self, charge_kw: float, discharge_kw: float, 
                               current_energy_kwh: float, dt_hours: float = 1.0) -> float:
        """
        Calculate the change in stored energy.
        
        Args:
            charge_kw: Charging power in kW
            discharge_kw: Discharging power in kW
            current_energy_kwh: Current stored energy in kWh
            dt_hours: Time step in hours
            
        Returns:
            New energy level in kWh
        """
        # Apply self-discharge
        energy_after_losses = current_energy_kwh * (1 - self.self_discharge * dt_hours)
        
        # Apply charging and discharging with efficiencies
        energy_change = (self.charge_efficiency * charge_kw - 
                        discharge_kw / self.discharge_efficiency) * dt_hours
        
        return energy_after_losses + energy_change
        
    def get_replacement_schedule(self, annual_cycles: float, project_lifetime: int) -> List[int]:
        """
        Calculate when storage needs replacement based on cycle life.
        
        Args:
            annual_cycles: Expected number of cycles per year
            project_lifetime: Project lifetime in years
            
        Returns:
            List of years when replacement is needed
        """
        if self.cycle_life is None:
            return []
        
        years_per_replacement = self.cycle_life / annual_cycles
        replacements = []
        year = years_per_replacement
        
        while year < project_lifetime:
            replacements.append(int(year))
            year += years_per_replacement
            
        return replacements
        
    def get_degraded_capacity(self, year: int, initial_capacity_kwh: float) -> float:
        """
        Calculate degraded capacity after given years of operation.
        
        Args:
            year: Years of operation
            initial_capacity_kwh: Initial capacity in kWh
            
        Returns:
            Degraded capacity in kWh
        """
        if self.degradation_rate is None:
            return initial_capacity_kwh
        
        return initial_capacity_kwh * (1 - self.degradation_rate) ** year
        
    def get_parameters_dict(self) -> Dict:
        """Get storage parameters as dictionary for easy integration."""
        return {
            'name': self.name,
            'type': self.storage_type,
            'capacity': self.storage_capacity,
            'capex_per_kwh': self.capex_per_kwh,
            'efficiency': self.efficiency,
            'max_c_rate': self.max_c_rate,
            'max_d_rate': self.max_d_rate,
            'min_soc': self.min_soc,
            'max_soc': self.max_soc,
            'self_discharge': self.self_discharge
        }


class StorageTemplates:
    """Factory class for creating common storage types with typical parameters."""
    
    @staticmethod
    def create_lithium_ion(name: str, time_horizon: int = 8760) -> Storage:
        """Create a lithium-ion battery storage system."""
        return Storage(
            name=name,
            storage_type='lithium_ion',
            storage_capacity=200000, # KW --> 100,000 KW = 100 MW
            capex_per_kwh=300,  # $/kWh
            capex_per_kw= 770,     # https://atb.nrel.gov/electricity/2021/utility-scale_battery_storage
            opex_fraction=0.02,
            round_trip_efficiency=0.92,
            max_c_rate=1.0,
            max_d_rate=1.0,
            min_soc=0.1,
            max_soc=0.9,
            self_discharge_rate=0.0001,
            cycle_life=5000,
            degradation_rate=0.02,  # 2% per year
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_flow_battery(name: str, time_horizon: int = 8760) -> Storage:
        """Create a vanadium flow battery storage system."""
        return Storage(
            name=name,
            storage_type='flow_battery',
            storage_capacity=100,
            capex_per_kwh=400,  # $/kWh
            capex_per_kw=800,   # $/kW (separate power module)
            opex_fraction=0.025,
            round_trip_efficiency=0.75,
            max_c_rate=0.25,
            max_d_rate=0.25,
            min_soc=0.0,
            max_soc=1.0,
            self_discharge_rate=0.001,
            cycle_life=15000,
            degradation_rate=0.005,  # 0.5% per year
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_pumped_hydro(name: str, time_horizon: int = 8760) -> Storage:
        """Create a pumped hydro storage system."""
        return Storage(
            name=name,
            storage_type='pumped_hydro',
            storage_capacity=100,
            capex_per_kwh=100,  # $/kWh
            opex_fraction=0.015,
            round_trip_efficiency=0.80,
            max_c_rate=0.1,
            max_d_rate=0.1,
            min_soc=0.1,
            max_soc=0.95,
            self_discharge_rate=0.00001,
            cycle_life=None,  # No cycle limit
            degradation_rate=0.001,  # 0.1% per year
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_compressed_air(name: str, time_horizon: int = 8760) -> Storage:
        """Create a compressed air energy storage (CAES) system."""
        return Storage(
            name=name,
            storage_type='caes',
            storage_capacity=100,
            capex_per_kwh=50,  # $/kWh
            capex_per_kw=1000,  # $/kW
            opex_fraction=0.02,
            round_trip_efficiency=0.70,
            max_c_rate=0.25,
            max_d_rate=0.25,
            min_soc=0.2,
            max_soc=0.9,
            self_discharge_rate=0.002,
            cycle_life=None,
            degradation_rate=0.005,
            time_horizon=time_horizon
        )
    
    @staticmethod
    def create_hydrogen(name: str, time_horizon: int = 8760) -> Storage:
        """Create a hydrogen storage system (electrolyzer + fuel cell)."""
        return Storage(
            name=name,
            storage_type='hydrogen',
            storage_capacity=100,
            capex_per_kwh=20,   # $/kWh (tank)
            capex_per_kw=2000,  # $/kW (electrolyzer + fuel cell)
            opex_fraction=0.03,
            round_trip_efficiency=0.40,
            max_c_rate=0.5,
            max_d_rate=0.5,
            min_soc=0.1,
            max_soc=0.9,
            self_discharge_rate=0.0001,
            cycle_life=None,
            degradation_rate=0.02,
            time_horizon=time_horizon
        )