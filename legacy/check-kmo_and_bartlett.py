import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Load your Excel file
df = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")

# Drop non-numeric or ID columns
X = df.drop(columns=["Sample ID", 'pH',	'EC','TDS',	'Sal.'])

# KMO Test
kmo_all, kmo_model = calculate_kmo(X)
print("KMO Score (overall):", kmo_model)

# Print individual KMO values
kmo_series = pd.Series(kmo_all, index=X.columns)
print("\nIndividual KMO values (sorted):")
print(kmo_series.sort_values())

# Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(X)
print("\nBartlett’s Test p-value:", p_value)
