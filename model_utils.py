

import pandas as pd


REFERENCE_DATE = pd.Timestamp("2014-06-29")

SPENDING_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

CHILDREN_COLS = ["Kidhome", "Teenhome"]

EDUCATION_MAP = {
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate",
}

LIVING_WITH_MAP = {
    "Married": "Partner",
    "Together": "Partner",
    "Single": "Alone",
    "Divorced": "Alone",
    "Widow": "Alone",
    "Alone": "Alone",
    "Absurd": "Alone",
    "YOLO": "Alone",
}

# Columns of df_cleaned (post feature-engineering, pre one-hot-encoding),
# in the exact order used at training time.
CLEANED_COLUMNS = [
    "Education",
    "Income",
    "Recency",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Complain",
    "Response",
    "Age",
    "Customer_Tenure_Days",
    "Total_Spending",
    "Total_Children",
    "Living_With",
]

CAT_COLS = ["Education", "Living_With"]


PERSONAS = {
    0: {
        "name": "Family Shoppers",
        "tagline": "Budget-conscious households",
        "description": (
            "Lower income and lower total spending, but the highest average "
            "number of children at home. They shop deal-driven and buy "
            "across all channels only occasionally."
        ),
        "recommendation": (
            "Target with family-size bundle deals, discount coupons, and "
            "value-tier promotions rather than premium products."
        ),
        "color": "#F4A259",
    },
    1: {
        "name": "Loyal Customers",
        "tagline": "Highest-value, most active segment",
        "description": (
            "Highest income and highest total spending, actively purchasing "
            "through web, catalog, and store channels alike."
        ),
        "recommendation": (
            "Enroll in a premium loyalty program with early access to new "
            "products and VIP perks — protect and grow this segment's "
            "lifetime value."
        ),
        "color": "#5B8C5A",
    },
    2: {
        "name": "Target for Reactivation",
        "tagline": "Low-income, low-engagement",
        "description": (
            "Lowest total spending of all segments, infrequent purchases "
            "across every channel, despite a comparable household size to "
            "Family Shoppers."
        ),
        "recommendation": (
            "Re-engage with win-back email campaigns, entry-level product "
            "offers, and low-cost personalized discounts to reactivate "
            "purchasing."
        ),
        "color": "#C96567",
    },
    3: {
        "name": "Best ROI",
        "tagline": "High spend, highest campaign responsiveness",
        "description": (
            "Income and spending nearly match the Loyal Customers segment, "
            "but this group responds to marketing campaigns at roughly "
            "double the rate — the highest Response rate of any cluster."
        ),
        "recommendation": (
            "Prioritize marketing spend here first — this segment converts "
            "campaigns most efficiently, making it the highest-ROI target "
            "for new promotions."
        ),
        "color": "#4C6EF5",
    },
}


def build_customer_row(raw: dict) -> pd.DataFrame:
    """Takes raw customer inputs (matching the original dataset's raw
    columns) and returns a single-row DataFrame with the same engineered
    columns as df_cleaned in the notebook, in CLEANED_COLUMNS order.
    """
    dt_customer = pd.Timestamp(raw["Dt_Customer"])
    tenure_days = (REFERENCE_DATE - dt_customer).days

    total_spending = sum(raw[c] for c in SPENDING_COLS)
    total_children = raw["Kidhome"] + raw["Teenhome"]
    age = 2026 - raw["Year_Birth"]

    education = EDUCATION_MAP.get(raw["Education"], raw["Education"])
    living_with = LIVING_WITH_MAP.get(raw["Marital_Status"], raw["Marital_Status"])

    row = {
        "Education": education,
        "Income": raw["Income"],
        "Recency": raw["Recency"],
        "NumDealsPurchases": raw["NumDealsPurchases"],
        "NumWebPurchases": raw["NumWebPurchases"],
        "NumCatalogPurchases": raw["NumCatalogPurchases"],
        "NumStorePurchases": raw["NumStorePurchases"],
        "NumWebVisitsMonth": raw["NumWebVisitsMonth"],
        "Complain": raw["Complain"],
        "Response": raw["Response"],
        "Age": age,
        "Customer_Tenure_Days": tenure_days,
        "Total_Spending": total_spending,
        "Total_Children": total_children,
        "Living_With": living_with,
    }
    return pd.DataFrame([row])[CLEANED_COLUMNS]
