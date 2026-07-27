

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import CAT_COLS, CHILDREN_COLS, EDUCATION_MAP, LIVING_WITH_MAP, SPENDING_COLS

DATA_PATH = "smartcart_customers.csv"
MODEL_DIR = "models"
ASSET_DIR = "assets"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)

# Load + preprocess -
    df = pd.read_csv(DATA_PATH)
    df["Income"] = df["Income"].fillna(df["Income"].median())

    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Children"] = df[CHILDREN_COLS].sum(axis=1)

    df["Education"] = df["Education"].replace(EDUCATION_MAP)
    df["Living_With"] = df["Marital_Status"].replace(LIVING_WITH_MAP)

    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"] + SPENDING_COLS
    df_cleaned = df.drop(columns=cols_to_drop)

    # Outlier removal
    df_cleaned = df_cleaned[(df_cleaned["Age"] < 90)]
    df_cleaned = df_cleaned[(df_cleaned["Income"] < 600_000)]
    df_cleaned = df_cleaned.reset_index(drop=True)

    # Encode + scale + PCA 
    ohe = OneHotEncoder()
    enc_cols = ohe.fit_transform(df_cleaned[CAT_COLS])
    enc_df = pd.DataFrame(
        enc_cols.toarray(), columns=ohe.get_feature_names_out(CAT_COLS), index=df_cleaned.index
    )
    df_encoded = pd.concat([df_cleaned.drop(columns=CAT_COLS), enc_df], axis=1)
    training_column_order = df_encoded.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    #  Elbow + silhouette curves (for Model Insights) 
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        wcss.append(km.inertia_)

    sil_scores = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        labels_k = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels_k))

    #  Fit all 3 algorithms for comparison -
    kmeans_final = KMeans(n_clusters=4, random_state=42)
    labels_kmeans = kmeans_final.fit_predict(X_pca)
    kmeans_silhouette = silhouette_score(X_pca, labels_kmeans)

    dbscan = DBSCAN(eps=0.7, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_pca)
    n_dbscan_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_dbscan_noise = int(list(labels_dbscan).count(-1))

    agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels_agg = agg.fit_predict(X_pca)
    agg_silhouette = silhouette_score(X_pca, labels_agg)

    print(f"KMeans (k=4) silhouette:        {kmeans_silhouette:.3f}")
    print(f"DBSCAN clusters/noise:          {n_dbscan_clusters} clusters, {n_dbscan_noise} noise points")
    print(f"Agglomerative (k=4) silhouette: {agg_silhouette:.3f}  <-- FINAL MODEL")


    centroids = np.array([X_pca[labels_agg == c].mean(axis=0) for c in sorted(set(labels_agg))])

  
    df_encoded["cluster"] = labels_agg
    profile_cols = [
        "Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases",
        "NumCatalogPurchases", "NumDealsPurchases", "Recency", "Response",
        "Age", "Total_Children",
    ]
    cluster_stats = df_encoded.groupby("cluster")[profile_cols].mean().round(2)
    cluster_sizes = df_encoded["cluster"].value_counts().sort_index()


    joblib.dump(ohe, os.path.join(MODEL_DIR, "ohe.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    np.save(os.path.join(MODEL_DIR, "centroids.npy"), centroids)

    with open(os.path.join(MODEL_DIR, "training_column_order.json"), "w") as f:
        json.dump(training_column_order, f, indent=2)

    with open(os.path.join(MODEL_DIR, "cluster_metrics.json"), "w") as f:
        json.dump(
            {
                "kmeans_k4_silhouette": round(float(kmeans_silhouette), 4),
                "agglomerative_k4_silhouette": round(float(agg_silhouette), 4),
                "dbscan_n_clusters": n_dbscan_clusters,
                "dbscan_n_noise": n_dbscan_noise,
                "dbscan_total_points": int(len(labels_dbscan)),
                "elbow_k": list(range(1, 11)),
                "elbow_wcss": [round(float(v), 2) for v in wcss],
                "silhouette_k": list(range(2, 11)),
                "silhouette_scores": [round(float(v), 4) for v in sil_scores],
                "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            },
            f,
            indent=2,
        )

    cluster_stats_dict = cluster_stats.reset_index().to_dict(orient="records")
    for row in cluster_stats_dict:
        row["size"] = int(cluster_sizes[row["cluster"]])
    with open(os.path.join(MODEL_DIR, "cluster_stats.json"), "w") as f:
        json.dump(cluster_stats_dict, f, indent=2)


    explorer_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "PC3": X_pca[:, 2],
            "cluster": labels_agg,
            "Income": df_cleaned["Income"].values,
            "Total_Spending": df_cleaned["Total_Spending"].values,
            "Age": df_cleaned["Age"].values,
            "Total_Children": df_cleaned["Total_Children"].values,
            "NumWebPurchases": df_cleaned["NumWebPurchases"].values,
            "NumStorePurchases": df_cleaned["NumStorePurchases"].values,
            "NumCatalogPurchases": df_cleaned["NumCatalogPurchases"].values,
            "Recency": df_cleaned["Recency"].values,
            "Response": df_cleaned["Response"].values,
        }
    )
    explorer_df.to_csv(os.path.join(ASSET_DIR, "explorer_data.csv"), index=False)

    print(f"\nSaved model artifacts to '{MODEL_DIR}/' and explorer data to '{ASSET_DIR}/'")


if __name__ == "__main__":
    main()
