import pandas as pd
import numpy as np

# Load Excel file
data = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")
standards = pd.read_excel("water_quality_data.xlsx", sheet_name="Standards")

# Convert all numeric cells to floats where possible
def safe_float(x):
    try:
        return float(str(x).strip())
    except:
        return np.nan

# Apply to all data columns except Sample ID
for col in data.columns:
    if col != "Sample ID":
        data[col] = data[col].apply(safe_float)

# Clean and convert standards too
standards["Permissible Limit (Si)"] = standards["Permissible Limit (Si)"].apply(safe_float)
standards["Ideal Value (Ci)"] = standards["Ideal Value (Ci)"].apply(safe_float)

# Compute k for Wi
standards["Inverse_Si"] = 1 / standards["Permissible Limit (Si)"]
k = 1 / standards["Inverse_Si"].sum()
standards["Wi"] = k / standards["Permissible Limit (Si)"]

# Initialize result DataFrame
results_df = data.copy()
results_df["k"] = k

# Loop through each parameter and calculate Qi, Wi, Ii
for param in standards["Parameter"]:
    if param not in data.columns:
        continue

    Si = standards.loc[standards["Parameter"] == param, "Permissible Limit (Si)"].values[0]
    Ci = standards.loc[standards["Parameter"] == param, "Ideal Value (Ci)"].values[0]
    Wi = standards.loc[standards["Parameter"] == param, "Wi"].values[0]

    def calc_qi(Ca):
        if pd.isna(Ca):
            return np.nan
        if param.lower() == "pH":
            if Ca > 7:
                return ((Ca - 7) / (8.5 - 7)) * 100
            else:
                return ((7 - Ca) / (7 - 6.5)) * 100
        return ((Ca - Ci) / (Si - Ci)) * 100 if Si != Ci else 0

    # Calculate Qi and Ii
    results_df[f"{param}_Qi"] = results_df[param].apply(calc_qi)
    results_df[f"{param}_Wi"] = Wi
    results_df[f"{param}_Ii"] = results_df[f"{param}_Qi"] * Wi

# Identify columns for summing
ii_cols = [col for col in results_df.columns if col.endswith("_Ii")]
wi_cols = [col.replace("_Ii", "_Wi") for col in ii_cols]

# Row-wise WQI calculation to handle NaNs properly
wqi_values = []
for _, row in results_df.iterrows():
    sum_ii = 0
    sum_wi = 0
    for ii_col, wi_col in zip(ii_cols, wi_cols):
        Ii = row[ii_col]
        Wi = row[wi_col]
        if not pd.isna(Ii):
            sum_ii += Ii
            sum_wi += Wi
    wqi = sum_ii / sum_wi if sum_wi != 0 else np.nan
    wqi_values.append(wqi)

results_df["WQI"] = wqi_values

# Save final result
results_df.to_excel("Detailed_WQI_GroundWater_results.xlsx", index=False)
print("WQI calculation completed and exported to 'Detailed_WQI_GroundWater_results.xlsx'")
