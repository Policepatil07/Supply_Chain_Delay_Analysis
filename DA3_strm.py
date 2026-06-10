import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Supply Chain Delay Dashboard",layout="wide")

sns.set_theme(style="whitegrid")

st.title("📦 Supply Chain Delay Dashboard")

# Load Data
df = pd.read_csv("scda.csv")

df["purchase_date"] = pd.to_datetime(df["purchase_date"],errors="coerce")

# Data Cleaning

df = df.dropna(subset=["purchase_date"])

# Fill missing values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna("Unknown")

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())


# KPI Metrics

total_orders = len(df)

delayed_orders = (
    df["delivery_status"].astype(str).str.lower() == "delayed").sum()

on_time_orders = total_orders - delayed_orders

delay_rate = (
    delayed_orders / total_orders * 100
    if total_orders > 0
    else 0)

avg_shipping = df["shipping_time_days"].mean()

st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Delayed Orders", f"{delayed_orders:,}")
col3.metric("Delay Rate (%)", f"{delay_rate:.2f}")
col4.metric("Avg Shipping Days", f"{avg_shipping:.2f}")


# Monthly Delay Trend

st.subheader("📈 Monthly Delay Trend")

df["month"] = df["purchase_date"].dt.to_period("M")

monthly_delay = (
    df.groupby("month")["delivery_status"]
    .apply(lambda x: (
            x.astype(str).str.lower().eq("delayed").mean()* 100)))

fig, ax = plt.subplots(figsize=(10, 5))

monthly_delay.plot(marker="o",linewidth=2,ax=ax)

ax.set_title("Monthly Delay Percentage")
ax.set_xlabel("Month")
ax.set_ylabel("Delay (%)")
ax.grid(True)

st.pyplot(fig)


# Top Delay Locations

st.subheader("🌍 Top Delay Locations")

location_delay = (
    df.groupby("location")["delivery_status"]
    .apply(lambda x: (
            x.astype(str).str.lower().eq("delayed").sum()
        )).sort_values(ascending=False).head(10))

fig2, ax2 = plt.subplots(figsize=(10, 5))

sns.barplot(x=location_delay.values,y=location_delay.index,palette="Reds_r",ax=ax2)

ax2.set_xlabel("Delayed Orders")
ax2.set_ylabel("Location")
ax2.set_title("Top 10 Delay Locations")

st.pyplot(fig2)

# Shipping Time by Category

st.subheader("📦 Shipping Time by Category")

fig3, ax3 = plt.subplots(figsize=(12, 6))

sns.boxplot(data=df,x="category",y="shipping_time_days",ax=ax3)

ax3.set_title("Shipping Time Distribution by Category")
plt.xticks(rotation=45)

st.pyplot(fig3)

# Seller Performance & Risk Score

df["is_delayed"] = np.where(df["delivery_status"].astype(str).str.lower() == "delayed",1,0)

# Risk Score Calculation
df["risk_score"] = ((10 - df["seller_rating"])+ ((100 - df["stock"]) / 10)+ df["shipping_time_days"])

st.subheader("🛒 Seller Performance & Risk Score")

top_sellers = (df.groupby("seller_id")
    .agg(
        avg_shipping_days=("shipping_time_days", "mean"),
        avg_seller_rating=("seller_rating", "mean"),
        delay_rate=("is_delayed", "mean"),
        avg_risk_score=("risk_score", "mean"),)
    .sort_values(by="avg_risk_score",ascending=False).head(10))

top_sellers["delay_rate"] = (top_sellers["delay_rate"] * 100)

st.dataframe(top_sellers.reset_index(),use_container_width=True)

# Machine Learning: Delay Prediction

st.subheader("🤖 Delay Risk Prediction")

features = ["category","subcategory","brand","location","device","payment_method","price","discount","final_price","rating","review_count","stock",
    "seller_rating","shipping_time_days","is_returned",]

df_ml = df.copy()

# Encode categorical columns
categorical_cols = ["category","subcategory","brand","location","device","payment_method",]

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Convert Yes/No values
if df_ml["is_returned"].dtype == "object":
    df_ml["is_returned"] = (
        df_ml["is_returned"]
        .astype(str).str.lower().map({"yes": 1,"no": 0}).fillna(0))

X = df_ml[features]
y = df_ml["is_delayed"]

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Model Training

model = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
model.fit(X_train, y_train)

# Accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

st.metric("Model Accuracy",f"{accuracy * 100:.2f}%")

# Predict Risk for All Records

df_ml["predicted_delay_risk"] = (model.predict_proba(X)[:, 1] * 100)

st.subheader("🚨 Highest Delay Risk Orders")

high_risk_orders = (df_ml[[
            "user_id",
            "product_id",
            "predicted_delay_risk"]].sort_values(by="predicted_delay_risk",ascending=False).head(20))

st.dataframe(high_risk_orders,use_container_width=True)

# Feature Importance

st.subheader("📌 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_}).sort_values(by="Importance",ascending=False)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df,x="Importance",y="Feature",palette="viridis",ax=ax4)
ax4.set_title("Random Forest Feature Importance")
st.pyplot(fig4)