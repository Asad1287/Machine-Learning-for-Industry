"""
In this notebook, we simulate and analyze the variations in key wastewater parameters over a period of 112 days, using a time resolution of four observations per day. This simulation takes into account different operating conditions—DRY, RAIN, and STORM—to capture realistic fluctuations in wastewater characteristics. Each operating regime has a distinct impact on the parameters, based on the variability in water inflow and pollutant concentration observed in wastewater treatment plants.


COD (Chemical Oxygen Demand): Indicates the organic matter present in water.
BOD5 (Biological Oxygen Demand): Measures the oxygen required by microorganisms to break down organic matter over five days.
TSS (Total Suspended Solids): Reflects the suspended particles in wastewater.
SNH4-N (Ammonium Nitrogen): Represents the nitrogen content from ammonium.
SNO-N (Nitrate Nitrogen): Indicates the nitrogen present in the form of nitrate.
Ntot (Total Nitrogen): The combined nitrogen content from various sources.
Each parameter has a baseline value, with fluctuations applied to simulate different operating conditions:

DRY Regime: Small fluctuations around the base value.
RAIN Regime: Moderate increase in values due to higher inflow.
STORM Regime: Larger fluctuations, reflecting peak pollutant levels.
Data Analysis
We plotted each parameter across the simulated period to observe how values change over time in response to the operating regime. The transitions between DRY, RAIN, and STORM conditions are evident from the chart, showing increasing trends in pollutant levels during more intense inflow events.

Observations
Parameter Trends: The parameters tend to increase in concentration from DRY to STORM regimes, reflecting the expected stress on wastewater treatment plants during high inflow conditions.
Regime Transitions: The plot shows smooth transitions in parameter values between different operating regimes, with the most significant spikes occurring during STORM periods.
This simulation provides insights into the typical performance of wastewater treatment processes under variable conditions and could serve as a basis for testing control algorithms or optimization techniques in wastewater treatment.

1. S_OD4, S_OD5, S_OD6 (Setpoints for Dissolved Oxygen in Tanks B4, B5, and B6)
Definition: These values represent the desired concentrations of dissolved oxygen (DO) in different tanks (B4, B5, and B6) within the biological treatment stage of the wastewater process.
Role: Dissolved oxygen is critical for the aerobic biological processes that help remove organic pollutants. Optimal DO levels allow aerobic bacteria to thrive, breaking down organic matter efficiently.
In Optimization: In the equation for J, the term J1 penalizes deviations of S_OD4, S_OD5, and S_OD6 from target values (e.g., 2.2 mg/L for S_OD4 and 2.0 mg/L for the others). This part of the cost function ensures DO levels stay close to optimal.
2. S_NO4 (Setpoint for Nitrate Concentration)
Definition: This setpoint represents the target concentration of nitrate in a specific stage of the treatment process (often the anoxic zone, where denitrification occurs).
Role: Nitrate (NO₃⁻) concentration control is vital for nitrogen removal through denitrification, a process where bacteria convert nitrate into nitrogen gas. Proper nitrate control reduces the nitrogen levels in effluent, helping the plant meet regulatory standards.
In Optimization: In J2, the deviation of S_NO4 from 0.5 mg/L is penalized. This term ensures that the nitrate level stays close to the desired concentration for optimal nitrogen removal efficiency.
3. k_RE (Recirculation Flow Coefficient)
Definition: This coefficient determines the proportion of the recirculated flow rate, which recycles part of the treated water back into the system for additional treatment.
Role: Recirculation helps maintain the concentration of microbes, improving the efficiency of biological processes. A proper flow rate is crucial to maintain stable conditions in the biological treatment tanks.
In Optimization: J2 also penalizes deviations of k_RE from 0.75, which is an optimal value in this model. This term helps ensure the recirculation rate is within an efficient range, balancing treatment quality and energy costs.
4. Q_PC (Primary Clarifier Sludge Flow Rate)
Definition: This setpoint defines the rate of sludge flow extracted from the primary clarifier, where solids settle out of the wastewater.
Role: Proper sludge removal prevents the buildup of solids in the clarifier, ensuring efficient treatment and preventing overflow. Sludge flow also affects downstream treatment stages, as excess sludge can lead to higher operational costs.
In Optimization: In J3, the deviation of Q_PC from 2250 m³/day is penalized, indicating an optimal range for maintaining clarifier performance and minimizing costs associated with excess sludge.
5. Q_EXC (Excess Sludge Flow Rate)
Definition: This parameter sets the flow rate for excess sludge removed from the system after secondary treatment.
Role: Excess sludge must be removed to prevent clogging and maintain the system's balance. This flow affects sludge processing costs and efficiency.
In Optimization: The term J3 penalizes deviations of Q_EXC from 450 m³/day, maintaining optimal sludge removal rates to avoid over- or under-processing of excess sludge.
Performance Criterion J
The cost function J aggregates three components:

J1: The sum of squared deviations of the dissolved oxygen setpoints S_OD4, S_OD5, and S_OD6 from their optimal values. This term focuses on maintaining proper oxygen levels.
J2: Penalizes deviations of the nitrate concentration S_NO4 and recirculation coefficient k_RE, balancing nitrogen removal and recirculation.
J3: Penalizes deviations of sludge flow rates Q_PC and Q_EXC, optimizing sludge management.
The weighting coefficients alpha_1, alpha_2, and alpha_3 (0.5, 1, and 100) prioritize these terms based on their relative importance. The high weight on J3 emphasizes the importance of controlling sludge flow rates due to their significant impact on treatment efficiency and cost.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the simulation parameters
np.random.seed(42)
days = 112
time_steps_per_day = 4  # Simulating four time points per day
total_steps = days * time_steps_per_day

# Parameters to simulate: COD, BOD5, TSS, SNH4-N, SNO-N, and Ntot
# Initialize time series for each parameter with base values and add random fluctuations
base_values = {
    "COD": 100,    # Base chemical oxygen demand in mg/L
    "BOD5": 20,    # Base biological oxygen demand in mg/L
    "TSS": 30,     # Total suspended solids in mg/L
    "SNH4-N": 1.5, # Ammonium concentration in mg/L
    "SNO-N": 4,    # Nitrate concentration in mg/L
    "Ntot": 8      # Total nitrogen in mg/L
}

# Set up fluctuations to simulate DRY, RAIN, and STORM conditions
fluctuations = {
    "DRY": (0.9, 1.1),
    "RAIN": (1.1, 1.3),
    "STORM": (1.3, 1.6)
}

# Define operating regime over time (e.g., DRY for the first 84 days, RAIN and STORM for the last 28 days)
operating_regime = ["DRY"] * (84 * time_steps_per_day) + \
                   ["RAIN"] * (14 * time_steps_per_day) + \
                   ["STORM"] * (14 * time_steps_per_day)

# Create a DataFrame to hold simulated data
time_series = pd.date_range(start="2023-01-01", periods=total_steps, freq="6H")
data = pd.DataFrame(index=time_series)

# Simulate data for each parameter
for param, base in base_values.items():
    simulated_values = []
    for i in range(total_steps):
        regime = operating_regime[i]
        fluctuation = np.random.uniform(*fluctuations[regime])
        simulated_value = base * fluctuation
        simulated_values.append(simulated_value)
    data[param] = simulated_values

# Plotting the simulated wastewater parameters
plt.figure(figsize=(14, 10))
for i, param in enumerate(base_values.keys()):
    plt.plot(data.index, data[param], label=param)
plt.xlabel("Time")
plt.ylabel("Concentration (mg/L)")
plt.title("Simulated Wastewater Parameters Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

