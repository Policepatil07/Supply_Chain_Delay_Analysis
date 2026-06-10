import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("scda.csv")
print(df.shape)
print(df.head())
df.isnull().sum()

df.drop_duplicates(inplace=True)

df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df.describe()

# Method 1: Using delivery_status

df['is_delayed'] = np.where(df['delivery_status'].str.lower() == 'delayed',1,0)

print("\nDelay Distribution")
print(df['is_delayed'].value_counts())

delay_rate = df['is_delayed'].mean() * 100
print(f"\nOverall Delay Rate: {delay_rate:.2f}%")

#Visualisation

#shipping time analysis

plt.figure(figsize=(8,5))
sns.histplot(df['shipping_time_days'], bins=20, kde=True)
plt.title("Shipping Time Distribution")
plt.show()

#delay by location

location_delay = (df.groupby('location')['is_delayed'].mean().sort_values(ascending=False)* 100)

plt.figure(figsize=(10,5))
location_delay.plot(kind='bar')
plt.title("Delay Percentage by Location")
plt.ylabel("Delay %")
plt.show()

print(location_delay)

#delay by category

category_delay = (df.groupby('category')['is_delayed'].mean().sort_values(ascending=False)* 100)

plt.figure(figsize=(10,5))
category_delay.plot(kind='bar', color='orange')
plt.title("Delay Percentage by Category")
plt.ylabel("Delay %")
plt.show()

print(category_delay)

#delay by brand

brand_delay = (df.groupby('brand')['is_delayed'].mean().sort_values(ascending=False).head(10)* 100)

plt.figure(figsize=(10,5))
brand_delay.plot(kind='bar', color='green')
plt.title("Top Brands with Delays")
plt.ylabel("Delay %")
plt.show()

#delay by seller

seller_delay = (df.groupby('seller_id')['is_delayed'].mean().sort_values(ascending=False).head(10)* 100)

plt.figure(figsize=(10,5))
seller_delay.plot(kind='bar', color='red')
plt.title("Top Sellers with Delays")
plt.ylabel("Delay %")
plt.show()

#stock vs delay

plt.figure(figsize=(8,5))
sns.boxplot(x='is_delayed',y='stock',data=df)
plt.title("Stock Availability vs Delay")
plt.show()

#seller rating vs delay

plt.figure(figsize=(8,5))
sns.boxplot(x='is_delayed',y='seller_rating',data=df)
plt.title("Seller Rating vs Delay")
plt.show()

#payment method

payment_delay = (df.groupby('payment_method')['is_delayed'].mean()* 100)

plt.figure(figsize=(8,5))
payment_delay.plot(kind='bar')
plt.title("Delay by Payment Method")
plt.ylabel("Delay %")
plt.show()

#device analysis

device_delay = (df.groupby('device')['is_delayed'].mean()* 100)

plt.figure(figsize=(8,5))
device_delay.plot(kind='bar')
plt.title("Delay by Device")
plt.ylabel("Delay %")
plt.show()

#returned orders analysis

return_delay = pd.crosstab(df['is_returned'],df['is_delayed'])

print("\nReturned Orders vs Delays")
print(return_delay)

#correlation analysis

numeric_columns = ['price','discount','final_price','rating','review_count','stock','seller_rating','shipping_time_days','is_delayed']
corr = df[numeric_columns].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#pareto analysis

pareto = (df.groupby('location')['is_delayed'].sum().sort_values(ascending=False))

pareto.plot(kind='bar',figsize=(10,5))

plt.title("Pareto Analysis of Delays by Location")
plt.ylabel("Delayed Orders")
plt.show()

#feature

df_ml = df.copy()

features = ['category','subcategory','brand','location','device','payment_method']

le = LabelEncoder()

for col in features:
    df_ml[col] = le.fit_transform(df_ml[col])

# Convert return status if needed

if df_ml['is_returned'].dtype == 'object':
    df_ml['is_returned'] = le.fit_transform(df_ml['is_returned'])

#machine learning model

categorical_cols = ['category','subcategory','brand','location','device','payment_method']

X = pd.get_dummies(df_ml[features], columns=categorical_cols, drop_first=True)

y = df_ml['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

#predictions

y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

#confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#feature importance

importance = pd.DataFrame({'Feature': X.columns,'Importance': model.feature_importances_})

importance = importance.sort_values(by='Importance',ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance',y='Feature',data=importance)
plt.title("Feature Importance for Delivery Delays")
plt.show()

#root cause analysis

print("\nTop Factors Affecting Delays:")
print(importance.head(10))

print("\nAnalysis Completed Successfully!")

total_orders = len(df)
delayed_orders = df['is_delayed'].sum()
on_time_orders = total_orders - delayed_orders
delay_rate = (delayed_orders / total_orders) * 100
avg_shipping = df['shipping_time_days'].mean()

print("="*50)
print("SUPPLY CHAIN KPI DASHBOARD")
print("="*50)
print("Total Orders:", total_orders)
print("Delayed Orders:", delayed_orders)
print("On-Time Orders:", on_time_orders)
print("Delay Rate (%):", round(delay_rate,2))
print("Average Shipping Days:", round(avg_shipping,2))

df['month'] = df['purchase_date'].dt.to_period('M')
monthly_delay = df.groupby('month')['is_delayed'].mean()*100

plt.figure(figsize=(12,5))
monthly_delay.plot(marker='o')
plt.title('Monthly Delay Trend')
plt.ylabel('Delay Percentage')
plt.grid(True)
plt.show()

top_locations = df.groupby('location')['is_delayed'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_locations.values,y=top_locations.index)
plt.title("Top Delay Locations")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='category',y='shipping_time_days',data=df)
plt.xticks(rotation=45)
plt.title("Shipping Time by Category")
plt.show()

seller_score = df.groupby('seller_id').agg({'shipping_time_days':'mean','seller_rating':'mean','is_delayed':'mean'})

seller_score['delay_percent'] = seller_score['is_delayed']*100
seller_score = seller_score.sort_values(by='delay_percent',ascending=False)
print(seller_score.head(20))

df['risk_score'] = ((10-df['seller_rating'])+(100 - df['stock'])/10+df['shipping_time_days'])

print(df[['seller_id','stock','seller_rating','shipping_time_days','risk_score']].head())
