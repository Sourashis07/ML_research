import numpy as np

class SpacecraftFaultInjector:
    """
    Fault & Space Weather Dynamic Injector for Satellite Telemetry Simulator.
    Allows real-time injection of:
    - Solar Flare / CME Event (Radiation & Current Spikes across solar channels)
    - Battery Cell Drop (abrupt voltage drop)
    - Solar Array Shunt (current output failure)
    - Radiator Valve Sticking (thermal runaway)
    - Thermistor Drift (sensor calibration decay)
    """
    FAULT_TYPES = [
        "NONE",
        "SOLAR_FLARE_CME",
        "BATTERY_CELL_DROP",
        "SOLAR_ARRAY_SHUNT",
        "RADIATOR_VALVE_STICK",
        "THERMISTOR_DRIFT"
    ]

    def __init__(self):
        self.active_fault = "NONE"
        self.fault_severity = 1.0
        self.injection_start_step = 0
        self.time_warp_factor = 1  # 1x, 5x, 10x

    def set_fault(self, fault_type: str, severity: float = 1.0, current_step: int = 0):
        """Sets active hardware/space weather fault."""
        if fault_type in self.FAULT_TYPES:
            self.active_fault = fault_type
            self.fault_severity = severity
            self.injection_start_step = current_step
            print(f"[!] Injecting Fault: {fault_type} (Severity: {severity}x) at step {current_step}")
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    def clear_fault(self):
        """Resets fault state to nominal."""
        self.active_fault = "NONE"

    def apply_fault_to_telemetry(self, telemetry_data: np.ndarray, step_offset: int = 0) -> np.ndarray:
        """
        Modifies input telemetry array [N, C] with active fault distortions.
        - Col 0: Main telemetry channel (Voltage / Temperature / Current)
        - Col 1..C: Subsystem channels (Power V_bus, Solar I_sa, Battery T_batt, Thermal T_bay)
        """
        if self.active_fault == "NONE":
            return telemetry_data

        corrupted = telemetry_data.copy()
        N, C = corrupted.shape
        rel_steps = np.arange(N)

        if self.active_fault == "SOLAR_FLARE_CME":
            # Solar Flare / Coronal Mass Ejection: High energy particle burst causing spikes across all channels
            flare_spike = self.fault_severity * (0.6 + 0.4 * np.random.normal(0, 0.1, size=N))
            corrupted[:, 0] += flare_spike
            if C > 1:
                corrupted[:, 1:min(C, 5)] += flare_spike[:, None] * 0.8

        elif self.active_fault == "BATTERY_CELL_DROP":
            # Abrupt voltage drop in battery bus channel
            drop_val = 0.4 * self.fault_severity
            corrupted[:, 0] -= drop_val
            if C > 1:
                corrupted[:, 1] -= drop_val * 0.7  # V_bus drop

        elif self.active_fault == "SOLAR_ARRAY_SHUNT":
            # Solar array shunt failure: Current output drops near zero
            corrupted[:, 0] *= (1.0 - 0.7 * self.fault_severity)
            if C > 2:
                corrupted[:, 2] = 0.05  # I_SA current drop

        elif self.active_fault == "RADIATOR_VALVE_STICK":
            # Thermal cooling valve stuck closed: Exponential thermal runaway
            thermal_runaway = 0.05 * self.fault_severity * np.exp(rel_steps / (N * 0.3))
            corrupted[:, 0] += thermal_runaway
            if C > 3:
                corrupted[:, 3] += thermal_runaway * 1.2  # T_batt runaway

        elif self.active_fault == "THERMISTOR_DRIFT":
            # Sensor calibration drift: Linear upward ramp on temperature telemetry
            drift_ramp = (rel_steps / N) * 0.5 * self.fault_severity
            corrupted[:, 0] += drift_ramp

        return corrupted
