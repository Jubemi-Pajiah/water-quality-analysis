import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

# Step 1: Load the Excel file
df = pd.read_excel("water_quality_data.xlsx", sheet_name="GroundWater")

# Step 2: Drop non-numeric and excluded columns
X = df.drop(columns=["Sample ID", "pH", "EC", "TDS", "Sal."]) 

# Step 3: Fit factor analyzer without limiting number of factors
fa = FactorAnalyzer()
fa.fit(X)

# Step 4: Get eigenvalues
eigenvalues, _ = fa.get_eigenvalues()

# Step 5: Plot scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title("Scree Plot for Factor Analysis")
plt.xlabel("Factor Number")
plt.ylabel("Eigenvalue")
plt.axhline(y=1, color='red', linestyle='--', label="Eigenvalue = 1")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
