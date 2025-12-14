import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
stats = df[selected_features].agg(['mean', 'median', 'var', 'std']).transpose()
print(stats.round(2))

print("1. Plot: Churn Distribution (Pie Chart)")
plt.figure(figsize=(6, 5))
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=["#3b97f4", "#e22a2a"])
plt.title('Churn Rate (Yes vs No)')
plt.ylabel('')
plt.show()

print("2. Plot: Histogram")
plt.figure(figsize=(8, 5))
sns.histplot(df['tenure'], kde=True, color='skyblue')
plt.title('Tenure Distribution (Histogram)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.show()

print("3. Plot: Correlation Matrix")
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Heatmap)')
plt.show()

print("4. Plot: Box Plot")
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('Monthly Charges by Churn Status (Box Plot)')
plt.show()

