import pandas as pd
import numpy as np

# Load Excel file
data = pd.read_excel("water_quality_data.xlsx", sheet_name="StorageWater")
standards = pd.read_excel("water_quality_data.xlsx", sheet_name="Standards")

# Prepare limits
limits = dict(zip(standards["Parameter"], standards["Standard Limit (Si)"]))
pH_low = limits.get("pH_low", 6.5)
pH_high = limits.get("pH_high", 8.5)

# Identify parameters to compute WPI for
parameters = [param for param in data.columns if param in limits and not param.startswith("pH_")]

# Initialize list to hold WPI values
wpi_results = []

for index, row in data.iterrows():
    pl_values = []

    for param in parameters:
        try:
            Ci = float(str(row[param]).strip())
        except:
            continue  # skip non-numeric or bad entries

        Si = limits[param]

        if Ci == 0:
            continue  # ignore zero as specified

        # Handle pH separately
        if param == "pH":
            if Ci < 7:
                pl = (7 - Ci) / (7 - pH_low)
            elif Ci > 7:
                pl = (Ci - 7) / (pH_high - 7)
            else:
                pl = 0
        else:
            pl = 1 + ((Ci - Si) / Si)

        pl_values.append(pl)

    wpi = np.mean(pl_values) if pl_values else np.nan
    wpi_results.append(wpi)

# Add WPI to the data
data["WPI"] = wpi_results

# Export to Excel
data.to_excel("WPI_StorageWater_results.xlsx", index=False)
print("WPI calculation completed. Results saved to 'WPI_StorageWater_results.xlsx'")
