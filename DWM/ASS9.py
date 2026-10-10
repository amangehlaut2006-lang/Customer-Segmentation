
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 K-Means Clustering Dashboard")
st.caption("Data Warehousing and Mining | Experiment 09")

st.markdown("""
Explore datasets, identify groups using K-Means clustering,
evaluate cluster quality, and download your results.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Configuration")

dataset_type = st.sidebar.selectbox(
    "Select dataset type",
    [
        "Student Performance",
        "Customer Segmentation",
        "Custom Dataset"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to get started.")

    st.markdown("### Supported datasets")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎓 Student Performance")
        st.write("Analyze marks, study time, absences, and other numeric features.")

    with col2:
        st.subheader("🛍️ Customer Segmentation")
        st.write("Group customers by income, spending score, age, and other features.")

    with col3:
        st.subheader("📁 Custom Dataset")
        st.write("Upload your own CSV containing at least two numeric features.")

    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    df.columns = df.columns.astype(str).str.strip()

except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if len(df) < 3:
    st.error("Upload at least 3 rows to perform clustering.")
    st.stop()

st.success("Dataset loaded successfully!")

# --------------------------------------------------
# DATA OVERVIEW
# --------------------------------------------------

st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Total Columns", len(df.columns))
c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview dataset", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Column information"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

# --------------------------------------------------
# PREPARE NUMERIC FEATURES
# --------------------------------------------------

st.header("2. Select Features")

numeric_df = df.select_dtypes(include=np.number).copy()

# Remove numeric identifiers when possible
id_columns = [
    col for col in numeric_df.columns
    if col.lower().replace(" ", "").replace("_", "")
    in ["id", "customerid", "studentid", "recordid"]
]

numeric_df = numeric_df.drop(columns=id_columns, errors="ignore")

# Remove columns containing only missing values
numeric_df = numeric_df.dropna(axis=1, how="all")

if numeric_df.shape[1] < 2:
    st.error(
        "Your dataset needs at least two usable numeric columns. "
        "Check that numeric values are stored as numbers."
    )
    st.stop()

if dataset_type == "Student Performance":
    preferred = [
        "G1", "G2", "G3", "math score",
        "reading score", "writing score",
        "studytime", "absences"
    ]

elif dataset_type == "Customer Segmentation":
    preferred = [
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Annual Income",
        "Spending Score",
        "Age",
        "age",
        "income",
        "spending score"
    ]

else:
    preferred = []

column_lookup = {
    col.lower().strip(): col
    for col in numeric_df.columns
}

default_features = []

for name in preferred:
    actual_name = column_lookup.get(name.lower().strip())
    if actual_name and actual_name not in default_features:
        default_features.append(actual_name)

if len(default_features) < 2:
    default_features = list(numeric_df.columns[:min(5, len(numeric_df.columns))])

selected_features = st.multiselect(
    "Choose numeric columns for clustering",
    options=list(numeric_df.columns),
    default=default_features
)

if len(selected_features) < 2:
    st.warning("Select at least two numeric features.")
    st.stop()

# Convert infinite values into missing values
X = numeric_df[selected_features].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Remove rows where every selected feature is missing
valid_rows = X.notna().any(axis=1)

X = X.loc[valid_rows]
working_df = df.loc[valid_rows].copy()

if len(X) < 3:
    st.error("Not enough usable records after removing empty rows.")
    st.stop()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

try:
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

except Exception as e:
    st.error(f"Data preprocessing failed: {e}")
    st.stop()

st.caption(
    f"Using {len(X)} records and {len(selected_features)} features. "
    "Missing values are filled with column medians, then features are standardized."
)

# --------------------------------------------------
# CLUSTER SETTINGS
# --------------------------------------------------

st.header("3. Configure K-Means")

max_k = min(10, len(X) - 1)

k = st.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=max_k,
    value=min(3, max_k)
)

if st.button("🚀 Run Clustering", type="primary"):
    st.session_state["run_clustering"] = True

if not st.session_state.get("run_clustering", False):
    st.info("Choose your features and K, then click Run Clustering.")
    st.stop()

# --------------------------------------------------
# APPLY K-MEANS
# --------------------------------------------------

try:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

except Exception as e:
    st.error(f"Clustering failed: {e}")
    st.stop()

working_df["Cluster"] = labels

# --------------------------------------------------
# CLUSTER QUALITY
# --------------------------------------------------

st.header("4. Clustering Results")

unique_clusters = len(np.unique(labels))

if unique_clusters > 1 and unique_clusters < len(labels):
    score = silhouette_score(X_scaled, labels)
else:
    score = None

r1, r2, r3 = st.columns(3)

r1.metric("Clusters Found", unique_clusters)
r2.metric("Records Clustered", f"{len(labels):,}")
r3.metric(
    "Silhouette Score",
    f"{score:.4f}" if score is not None else "Unavailable"
)

if score is not None:
    if score >= 0.5:
        st.success("The clusters are reasonably well separated.")
    elif score >= 0.25:
        st.info("The clusters show some separation, with possible overlap.")
    else:
        st.warning(
            "The clusters overlap substantially. Try different features or K values."
        )

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.header("5. Cluster Distribution")

cluster_counts = (
    working_df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Records")
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Records",
        color="Cluster",
        title="Records in Each Cluster",
        text="Records",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Records",
        title="Cluster Proportions",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FEATURE SCATTER PLOT
# --------------------------------------------------

st.header("6. Feature-Based Visualization")

plot_df = pd.DataFrame(
    X_imputed,
    columns=selected_features,
    index=working_df.index
)

plot_df["Cluster"] = labels.astype(str)

x_feature = st.selectbox(
    "X-axis feature",
    selected_features,
    index=0
)

y_feature = st.selectbox(
    "Y-axis feature",
    selected_features,
    index=1
)

fig = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="Cluster",
    title=f"Clusters: {x_feature} vs {y_feature}",
    hover_data=selected_features,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------

st.header("7. PCA Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "Principal Component 1": X_pca[:, 0],
    "Principal Component 2": X_pca[:, 1],
    "Cluster": labels.astype(str)
})

fig = px.scatter(
    pca_df,
    x="Principal Component 1",
    y="Principal Component 2",
    color="Cluster",
    title="K-Means Clusters in Two-Dimensional PCA Space",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

variance = sum(pca.explained_variance_ratio_) * 100

st.metric("Variance Explained by Two Components", f"{variance:.2f}%")

# --------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------

st.header("8. Cluster Summary")

summary = (
    plot_df
    .assign(Cluster=labels)
    .groupby("Cluster")[selected_features]
    .mean()
    .round(2)
)

st.dataframe(summary, use_container_width=True)

# --------------------------------------------------
# CLUSTERED DATA AND DOWNLOAD
# --------------------------------------------------

st.header("9. Export Results")

st.dataframe(working_df, use_container_width=True)

csv_data = working_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Clustered CSV",
    data=csv_data,
    file_name="clustered_results.csv",
    mime="text/csv"
)

# Download cluster summary
summary_csv = summary.to_csv().encode("utf-8")

st.download_button(
    label="⬇️ Download Cluster Summary",
    data=summary_csv,
    file_name="cluster_summary.csv",
    mime="text/csv"
)

st.success("Clustering analysis completed successfully!")

st.caption(
    "DWM Experiment 09 | K-Means Clustering | "
    "Built with Python, Streamlit, Pandas, Scikit-learn and Plotly"
)
