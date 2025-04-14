# Water Quality Analysis Repository

## Repository Description
This repository contains Python scripts and analytical tools designed for comprehensive assessment and statistical analysis of water quality, specifically focusing on heavy metal contamination and general pollution indicators. It includes robust methodologies such as correlation analysis, cluster analysis (K-means clustering), the Elbow method, silhouette score analysis, Water Quality Index (WQI), and Water Pollution Index (WPI) calculations, making it ideal for researchers and environmental scientists working on water quality monitoring projects.

---

## 📁 Files and Scripts Description

| File Name                             | Description                                                        |
|---------------------------------------|--------------------------------------------------------------------|
| **correlation.py**                    | Computes Pearson correlation coefficients among heavy metal parameters, generating correlation matrices.|
| **cluster-analysis.py**               | Implements K-means clustering to classify water samples based on heavy metal concentrations.|
| **elbow-method-number-of-clusters.py**| Uses the Elbow method to determine the optimal number of clusters by plotting Within-Cluster Sum of Squares (WCSS).|
| **silhouette-score-confirmation.py**  | Validates the optimal number of clusters found via the Elbow method by calculating silhouette scores.|
| **WQI.py**                            | Calculates the Water Quality Index (WQI) using weighted arithmetic mean, assessing suitability of water samples for drinking based on standard guidelines.|
| **WPI.py**                            | Computes the Water Pollution Index (WPI) for generalized pollution assessment, providing insights into broader chemical pollution.|
| **columns-to-row.py**                 | Utility script for reshaping data from columns into rows (useful for data cleaning and preprocessing).|

---

## ⚙️ Prerequisites

- **Python 3.8 or higher**

### Required libraries:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- openpyxl (for Excel interaction)

Install dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

---

## 🚀 Getting Started

**Clone the repository:**

```bash
git clone [your-repository-link]
```

**Navigate into the repository:**

```bash
cd water-quality-analysis
```

**Prepare your dataset (see required formats below).**

**Execute scripts individually as needed:**

```bash
python correlation.py
python cluster-analysis.py
python WQI.py
python WPI.py
```

---

## 📑 Data Requirements

Your input Excel file (`water_quality_data.xlsx`) must contain the following sheets:

- **Data Sheet:** Named `"Data"`, includes raw data of measured concentrations.
- **Standards Sheet:** Named `"Standards"`, defines permissible limits for each parameter.

### Example (`Data` sheet):

| Sample ID | Fe  | Zn  | Cu  | Mn  | Ni   | Cr   | Co  | pH  | EC  | TDS |
|-----------|-----|-----|-----|-----|------|------|-----|-----|-----|-----|
| GW1       | 0.3 | 1.2 | 0.2 | 0.1 | 0.05 | 0.02 | 0.1 | 6.8 | 250 | 450 |

### Example (`Standards` sheet):

| Parameter | Permissible Limit (Si) |
|-----------|------------------------|
| Fe        | 0.3                    |
| Zn        | 3                      |
| pH_low    | 6.5                    |
| pH_high   | 8.5                    |
| ...       | ...                    |

---

## 📌 Analytical Methods Overview

- **Correlation Analysis:** Assesses inter-metal relationships to identify common sources or geochemical behaviors.
- **K-means Cluster Analysis:** Classifies samples based on similarity in heavy metal profiles.
- **Elbow Method and Silhouette Score:** Robust techniques for determining the ideal number of clusters.
- **Water Quality Index (WQI):** Evaluates drinking water quality, incorporating weighted health risks.
- **Water Pollution Index (WPI):** Measures broader chemical pollution without weighting or ideal value assumptions.

---

## 📘 Usage and Application

This toolkit supports environmental monitoring, public health assessments, water resource management, and academic research in water quality science. It simplifies complex analyses into reproducible, transparent, and scientifically rigorous processes.

---

## 🙌 Contributions

Contributions and improvements are welcome! Please open an issue or submit a pull request.

---

## 📞 Contact Information

For questions, collaboration, or feedback, please reach out via:

- **Email:** `[your email address]`
- **GitHub:** `[your GitHub profile link]`

---

## 📝 Citation

If you use this repository in your research or publications, kindly cite or acknowledge appropriately.

---

Enjoy analyzing your water quality data! 🚰✨
