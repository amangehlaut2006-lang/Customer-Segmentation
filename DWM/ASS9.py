import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

from tkinter import filedialog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------------------
# DWM EXPERIMENT 09: IMPLEMENTATION OF
# CLUSTERING USING K-MEANS ALGORITHM
# -----------------------------------------

print("======================================")
print("      DATA WAREHOUSING AND MINING")
print("       K-MEANS CLUSTERING")
print("======================================")
print("1. Student Performance Dataset")
print("2. Customer Dataset")
print("3. Import Your Own CSV Dataset")

choice = input("Select dataset (1/2/3): ").strip()

# Select CSV file
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

file_path = filedialog.askopenfilename(
    title="Select Dataset CSV File",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)

root.destroy()

if not file_path:
    raise SystemExit("No file selected.")

# Read CSV, automatically detecting common delimiters
df = pd.read_csv(file_path, sep=None, engine="python")
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

# -----------------------------------------
# SELECT FEATURES FOR CLUSTERING
# -----------------------------------------

numeric_df = df.select_dtypes(include=np.number)

if numeric_df.empty:
    raise SystemExit("The dataset contains no numeric columns.")

# Ignore numeric identifiers when possible
id_columns = [
    col for col in numeric_df.columns
    if col.lower() in ["id", "customerid", "studentid", "student_id"]
]

numeric_df = numeric_df.drop(columns=id_columns, errors="ignore")

if numeric_df.shape[1] == 0:
    raise SystemExit("No suitable numeric features found.")

if choice == "1":
    # Student performance: prefer marks and study-related features
    preferred = [
        "G1", "G2", "G3", "math score", "reading score",
        "writing score", "studytime", "absences"
    ]

elif choice == "2":
    # Customer segmentation: prefer income and spending
    preferred = [
        "Annual Income (k$)", "Spending Score (1-100)",
        "Annual Income", "Spending Score",
        "age", "income", "spending score"
    ]

else:
    # Use numeric columns from any uploaded CSV
    preferred = []

# Match columns without case sensitivity
column_lookup = {
    col.lower(): col for col in numeric_df.columns
}

features = []
for name in preferred:
    if name.lower() in column_lookup:
        actual_name = column_lookup[name.lower()]
        if actual_name not in features:
            features.append(actual_name)

# Use available numeric columns if preferred ones aren't found
if len(features) < 2:
    features = list(numeric_df.columns[:min(5, len(numeric_df.columns))])

if len(features) < 2:
    raise SystemExit("At least two numeric features are required.")

print("\nAvailable numeric columns:")
for col in numeric_df.columns:
    print("-", col)

print("\nFeatures selected for clustering:", features)

# -----------------------------------------
# DATA PREPROCESSING
# -----------------------------------------

X = numeric_df[features].copy()

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using column medians
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------------------------
# APPLY K-MEANS CLUSTERING
# -----------------------------------------

k = int(input("\nEnter number of clusters (e.g. 3): "))

if k < 2 or k >= len(df):
    raise SystemExit("Choose k >= 2 and smaller than the row count.")

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)

print("\nClustering completed successfully!")
print("\nCluster-wise record counts:")
print(df["Cluster"].value_counts().sort_index())

# -----------------------------------------
# CLUSTER QUALITY
# -----------------------------------------

if len(set(df["Cluster"])) > 1 and len(df) > k:
    score = silhouette_score(X_scaled, df["Cluster"])
    print("\nSilhouette Score:", round(score, 4))
    print("(A higher score generally indicates better-separated clusters.)")

# -----------------------------------------
# VISUALIZATION
# -----------------------------------------

# Use the first two selected features for plotting
plot_data = pd.DataFrame(X_imputed, columns=features)
plot_data["Cluster"] = df["Cluster"].to_numpy()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_data,
    x=features[0],
    y=features[1],
    hue="Cluster",
    palette="viridis",
    s=75,
    alpha=0.8
)

plt.title("K-Means Clustering - " +
          ("Student Performance" if choice == "1"
           else "Customer Segmentation" if choice == "2"
           else "Custom Dataset"))

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# -----------------------------------------
# PCA VISUALIZATION
# -----------------------------------------

# Project all selected features into two dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Cluster"],
    palette="viridis",
    s=65,
    alpha=0.8
)

plt.title("K-Means Clustering - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

print(
    "\nPCA variance explained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# -----------------------------------------
# DISPLAY CLUSTER SUMMARY
# -----------------------------------------

print("\nCluster Summary:")
print(df.groupby("Cluster")[features].mean().round(2))

# -----------------------------------------
# SAVE RESULTS
# -----------------------------------------

output_path = file_path.rsplit(".", 1)[0] + "_clustered.csv"
df.to_csv(output_path, index=False)

print("\nClustered dataset saved to:")
print(output_path)

print("\nExperiment completed successfully!")
