import json
import os
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
from model_utils import CAT_COLS, PERSONAS, build_customer_row
 
MODEL_DIR = "models"
ASSET_DIR = "assets"
 
st.set_page_config(page_title="SmartCart Customer Segmentation", page_icon="🛒", layout="wide")
 
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, rgba(201,101,103,0.10), rgba(76,110,245,0.06));
        border: 1px solid rgba(201,101,103,0.18);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0 0 0.25rem 0;
        font-size: 1.9rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.8;
    }
    .stat-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(0,0,0,0.06);
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-top: 0.75rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        opacity: 0.75;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }
    .persona-card {
        border-radius: 1rem;
        padding: 1.5rem;
        border-top: 6px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        height: 100%;
        background: rgba(255,255,255,0.55);
    }
    .persona-card h3 {
        margin-top: 0;
    }
    .persona-tagline {
        font-style: italic;
        opacity: 0.75;
        margin-bottom: 0.75rem;
    }
    .persona-stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        padding: 0.15rem 0;
        border-bottom: 1px dashed rgba(0,0,0,0.08);
    }
    .result-banner {
        border-radius: 1rem;
        padding: 1.75rem 2rem;
        border-left: 8px solid;
        box-shadow: 0 2px 14px rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
@st.cache_resource
def load_artifacts():
    ohe = joblib.load(os.path.join(MODEL_DIR, "ohe.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    centroids = np.load(os.path.join(MODEL_DIR, "centroids.npy"))
    with open(os.path.join(MODEL_DIR, "training_column_order.json")) as f:
        column_order = json.load(f)
    return ohe, scaler, pca, centroids, column_order
 
 
@st.cache_data
def load_explorer_data():
    path = os.path.join(ASSET_DIR, "explorer_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None
 
 
@st.cache_data
def load_cluster_stats():
    path = os.path.join(MODEL_DIR, "cluster_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
 
 
@st.cache_data
def load_cluster_metrics():
    path = os.path.join(MODEL_DIR, "cluster_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
 
 
explorer_df_preview = load_explorer_data()
metrics_preview = load_cluster_metrics()
n_customers = len(explorer_df_preview) if explorer_df_preview is not None else "-"
sil_score = f"{metrics_preview['agglomerative_k4_silhouette']:.3f}" if metrics_preview else "-"
 
st.markdown(
    f"""
    <div class="hero">
        <h1>🛒 SmartCart Customer Segmentation</h1>
        <p>Groups customers into behavioral segments using unsupervised learning
        (PCA + Agglomerative Clustering), to support targeted marketing decisions.</p>
        <span class="stat-pill">👥 {n_customers} customers analyzed</span>
        <span class="stat-pill">🧩 4 segments</span>
        <span class="stat-pill">📈 Silhouette score {sil_score}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 

# Classify a New Customer tab

st.markdown('<div class="section-title">🔎 Classify a New Customer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Enter a customer\'s profile below. Since clustering has no '
    "built-in way to score brand-new data, this assigns them to the segment whose average "
    "profile (centroid) is closest in the model's feature space.</div>",
    unsafe_allow_html=True,
)
 
with st.form("customer_form"):
    profile_col, spend_col, activity_col = st.columns(3)
 
    with profile_col:
        st.markdown("**Profile**")
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=700000, value=50000, step=1000)
        education = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"], index=2)
        marital_status = st.selectbox(
            "Marital Status", ["Married", "Together", "Single", "Divorced", "Widow"]
        )
        kidhome = st.number_input("Kids at home", min_value=0, max_value=5, value=0)
        teenhome = st.number_input("Teens at home", min_value=0, max_value=5, value=0)
        customer_since = st.date_input("Customer Since", value=pd.Timestamp("2013-01-01"))
 
    with spend_col:
        st.markdown("**Spending (last 2 years)**")
        mnt_wines = st.number_input("Wine spending ($)", min_value=0, value=200)
        mnt_fruits = st.number_input("Fruit spending ($)", min_value=0, value=20)
        mnt_meat = st.number_input("Meat spending ($)", min_value=0, value=150)
        mnt_fish = st.number_input("Fish spending ($)", min_value=0, value=30)
        mnt_sweets = st.number_input("Sweets spending ($)", min_value=0, value=20)
        mnt_gold = st.number_input("Gold product spending ($)", min_value=0, value=30)
 
    with activity_col:
        st.markdown("**Shopping Activity**")
        recency = st.slider("Days since last purchase", 0, 100, 30)
        num_deals = st.number_input("Deal purchases", min_value=0, value=2)
        num_web = st.number_input("Web purchases", min_value=0, value=4)
        num_catalog = st.number_input("Catalog purchases", min_value=0, value=2)
        num_store = st.number_input("Store purchases", min_value=0, value=5)
        num_web_visits = st.number_input("Web visits / month", min_value=0, value=5)
        complain = st.checkbox("Filed a complaint in the last 2 years")
        responded = st.checkbox("Responded to the last marketing campaign")
 
    submitted = st.form_submit_button("Classify Customer", type="primary", width="stretch")
 
if submitted:
    ohe, scaler, pca, centroids, column_order = load_artifacts()
 
    raw = {
        "Income": income,
        "Recency": recency,
        "NumDealsPurchases": num_deals,
        "NumWebPurchases": num_web,
        "NumCatalogPurchases": num_catalog,
        "NumStorePurchases": num_store,
        "NumWebVisitsMonth": num_web_visits,
        "Complain": int(complain),
        "Response": int(responded),
        "Year_Birth": 2026 - age,
        "Dt_Customer": customer_since.strftime("%Y-%m-%d"),
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweets,
        "MntGoldProds": mnt_gold,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Education": education,
        "Marital_Status": marital_status,
    }
 
    row_df = build_customer_row(raw)
    enc = ohe.transform(row_df[CAT_COLS])
    enc_df = pd.DataFrame(enc.toarray(), columns=ohe.get_feature_names_out(CAT_COLS))
    df_encoded_new = pd.concat(
        [row_df.drop(columns=CAT_COLS).reset_index(drop=True), enc_df], axis=1
    )[column_order]
 
    X_pca_new = pca.transform(scaler.transform(df_encoded_new))
    distances = np.linalg.norm(centroids - X_pca_new, axis=1)
    cluster_id = int(np.argmin(distances))
    persona = PERSONAS[cluster_id]
 
    st.divider()
    result_col, chart_col = st.columns([1.3, 1])
 
    with result_col:
        st.markdown(
            f"""
            <div class="result-banner" style="border-left-color: {persona['color']};">
                <div style="font-size:0.85rem; opacity:0.7; text-transform:uppercase; letter-spacing:0.05em;">Predicted Segment</div>
                <h2 style="margin:0.2rem 0;">🏷️ {persona['name']}</h2>
                <div class="persona-tagline">{persona['tagline']}</div>
                <p>{persona['description']}</p>
                <p><b>📣 Recommendation:</b> {persona['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    with chart_col:
        st.markdown("**Distance to each segment centroid**")
        st.caption("Lower = closer match")
        dist_df = pd.DataFrame(
            {
                "Segment": [PERSONAS[c]["name"] for c in sorted(PERSONAS.keys())],
                "Distance": distances,
            }
        ).sort_values("Distance")
        fig = go.Figure(
            go.Bar(
                x=dist_df["Distance"], y=dist_df["Segment"], orientation="h",
                marker_color=[PERSONAS[c]["color"] for c in sorted(PERSONAS.keys())],
            )
        )
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Distance")
        st.plotly_chart(fig, width="stretch")
 
st.divider()
 

# Secondary content

explorer_tab, persona_tab, insights_tab = st.tabs(
    ["🗺️ Cluster Explorer", "🧑‍🤝‍🧑 Customer Personas", "📊 Model Insights"]
)
 
with explorer_tab:
    st.markdown('<div class="section-title">Interactive Customer Map</div>', unsafe_allow_html=True)
    explorer_df = load_explorer_data()
 
    if explorer_df is None:
        st.info("Run `train_model.py` to generate the explorer dataset.")
    else:
        explorer_df = explorer_df.copy()
        explorer_df["Persona"] = explorer_df["cluster"].map(lambda c: PERSONAS[c]["name"])
 
        persona_options = [PERSONAS[c]["name"] for c in sorted(PERSONAS.keys())]
        selected = st.multiselect(
            "Filter by persona", options=persona_options, default=persona_options
        )
        filtered = explorer_df[explorer_df["Persona"].isin(selected)]
 
        color_map = {PERSONAS[c]["name"]: PERSONAS[c]["color"] for c in PERSONAS}
 
        fig = px.scatter_3d(
            filtered,
            x="PC1", y="PC2", z="PC3",
            color="Persona",
            color_discrete_map=color_map,
            hover_data={
                "Income": ":,.0f", "Total_Spending": ":,.0f", "Age": True,
                "Total_Children": True, "NumWebPurchases": True,
                "NumStorePurchases": True, "PC1": False, "PC2": False, "PC3": False,
            },
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            height=650,
            legend_title_text="Segment",
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, width="stretch")
 
        st.caption(
            "Each point is a customer, projected from 18 engineered features down to 3 "
            "principal components. Points close together behave similarly."
        )
 
with persona_tab:
    st.markdown('<div class="section-title">Segment Personas & Recommendations</div>', unsafe_allow_html=True)
    stats = load_cluster_stats()
 
    if stats is None:
        st.info("Run `train_model.py` to generate cluster profiles.")
    else:
        stats_by_cluster = {s["cluster"]: s for s in stats}
        cols = st.columns(4)
 
        for i, cluster_id in enumerate(sorted(PERSONAS.keys())):
            persona = PERSONAS[cluster_id]
            s = stats_by_cluster.get(cluster_id, {})
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="persona-card" style="border-top-color: {persona['color']};">
                        <h3>{persona['name']}</h3>
                        <div class="persona-tagline">{persona['tagline']}</div>
                        <div class="persona-stat-row"><span>Segment size</span><b>{s.get('size', '-')}</b></div>
                        <div class="persona-stat-row"><span>Avg. Income</span><b>${s.get('Income', 0):,.0f}</b></div>
                        <div class="persona-stat-row"><span>Avg. Spending</span><b>${s.get('Total_Spending', 0):,.0f}</b></div>
                        <div class="persona-stat-row"><span>Campaign Response</span><b>{s.get('Response', 0) * 100:.0f}%</b></div>
                        <div class="persona-stat-row"><span>Avg. Age</span><b>{s.get('Age', 0):.0f}</b></div>
                        <p style="margin-top:0.75rem; font-size:0.88rem;">{persona['description']}</p>
                        <p style="font-size:0.88rem;"><b>📣 Recommendation:</b> {persona['recommendation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
 
with insights_tab:
    st.markdown('<div class="section-title">Why Agglomerative Clustering?</div>', unsafe_allow_html=True)
    metrics = load_cluster_metrics()
 
    if metrics is None:
        st.info("Run `train_model.py` to generate model comparison metrics.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Agglomerative Silhouette (final model)", f"{metrics['agglomerative_k4_silhouette']:.3f}")
        m2.metric("KMeans Silhouette (k=4)", f"{metrics['kmeans_k4_silhouette']:.3f}")
        m3.metric("DBSCAN Result", f"{metrics['dbscan_n_clusters']} clusters, {metrics['dbscan_n_noise']} noise pts")
 
        st.write(
            "**Agglomerative Clustering** was selected as the final model — it achieved "
            f"the highest silhouette score ({metrics['agglomerative_k4_silhouette']:.3f}) "
            "and produced visibly better-separated, more compact clusters in the 3D PCA "
            "projection than KMeans or DBSCAN."
        )
        st.write(
            "**Why DBSCAN underperformed:** DBSCAN is a density-based algorithm, well "
            "suited to irregularly shaped clusters or noisy data. After PCA compression, "
            "this dataset's customer segments form compact, roughly spherical (globular) "
            "clusters of similar density — exactly the shape DBSCAN struggles to separate, "
            "since it looks for density *gaps* rather than spherical boundaries. It found "
            f"only {metrics['dbscan_n_clusters']} real clusters and flagged "
            f"{metrics['dbscan_n_noise']} points as noise, versus the more informative "
            "4-segment structure from Agglomerative Clustering."
        )
 
        st.divider()
        st.markdown("#### Choosing k: Elbow Method + Silhouette Score")
 
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["elbow_k"], y=metrics["elbow_wcss"], name="WCSS (Elbow)",
                mode="lines+markers", yaxis="y1", line=dict(color="#C96567"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["silhouette_k"], y=metrics["silhouette_scores"], name="Silhouette Score",
                mode="lines+markers", yaxis="y2", line=dict(color="#4C6EF5", dash="dash"),
            )
        )
        fig.update_layout(
            height=400,
            xaxis_title="Number of Clusters (k)",
            yaxis=dict(title="WCSS"),
            yaxis2=dict(title="Silhouette Score", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"PCA explained variance (3 components): "
            f"{sum(metrics['pca_explained_variance']) * 100:.1f}% of total variance retained."
        )