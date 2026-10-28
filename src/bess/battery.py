from dataclasses import dataclass, replace
from math import sqrt


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float          # nameplate energy capacity
    power_mw: float              # max charge/discharge power (symmetric)
    soc_min: float               # fraction of capacity, in [0, 1)
    soc_max: float               # fraction of capacity, in (0, 1]
    round_trip_efficiency: float # AC-AC round-trip efficiency, in (0, 1]
    initial_soc: float = 0.5     # fraction of capacity

    def __post_init__(self):
        if self.capacity_mwh <= 0:
            raise ValueError(f"capacity_mwh must be > 0, got {self.capacity_mwh}")
        if self.power_mw <= 0:
            raise ValueError(f"power_mw must be > 0, got {self.power_mw}")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError(
                f"need 0 <= soc_min < soc_max <= 1, got {self.soc_min} and {self.soc_max}"
            )
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError(
                f"round_trip_efficiency must be in (0, 1], got {self.round_trip_efficiency}"
            )
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError(
                f"initial_soc must be in [{self.soc_min}, {self.soc_max}], got {self.initial_soc}"
            )

    # Derived values
    @property
    def charge_efficiency(self) -> float:
        return sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return sqrt(self.round_trip_efficiency)

    @property
    def energy_min_mwh(self) -> float:
        return self.soc_min * self.capacity_mwh

    @property
    def energy_max_mwh(self) -> float:
        return self.soc_max * self.capacity_mwh

    @property
    def usable_energy_mwh(self) -> float:
        return self.energy_max_mwh - self.energy_min_mwh

    @property
    def initial_energy_mwh(self) -> float:
        return self.initial_soc * self.capacity_mwh

    @property
    def duration_h(self) -> float:
        return self.capacity_mwh / self.power_mw

    # Degradation hook: returns a NEW battery, the original is untouched
    def with_soh(self, soh: float) -> "Battery":
        """Battery with capacity scaled by state of health (0 < soh <= 1)."""
        if not 0 < soh <= 1:
            raise ValueError(f"soh must be in (0, 1], got {soh}")
        return replace(self, capacity_mwh=self.capacity_mwh * soh)


@dataclass(frozen=True)
class DegradationModel:
    capex_eur_per_mwh: float   # nameplate replacement cost
    cycle_life: float          # full equivalent cycles to end of life
    eol_soh: float = 0.8       # SoH considered end of life

    def cost_per_mwh_discharged(self, battery: Battery) -> float:
        return (self.capex_eur_per_mwh * battery.capacity_mwh
                / (self.cycle_life * battery.usable_energy_mwh))

    def soh_loss(self, battery: Battery, discharged_mwh: float) -> float:
        """Fraction of SoH lost for a given discharged energy."""
        fec = discharged_mwh / battery.usable_energy_mwh
        return (1 - self.eol_soh) * fec / self.cycle_life