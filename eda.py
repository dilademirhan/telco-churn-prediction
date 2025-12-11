import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

selected_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("--- İstatistiksel Özet (Mean, Median, Variance, Std) ---")
stats = df[selected_features].agg(['mean', 'median', 'var', 'std']).transpose()
print(stats.round(2))

print("1.Grafik: Histogram")
plt.figure(figsize=(8, 5))
sns.histplot(df['tenure'], kde=True, color='skyblue')
plt.title('Müşteri Süresi (Tenure) Histogramı')
plt.xlabel('Ay (Müşteri ne kadar süredir bizde?)')
plt.ylabel('Kişi Sayısı')
plt.show()

print("2. Grafik: Korelasyon Matrisi")
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi (Heatmap)')
plt.show()

print("3. Grafik: Box Plot ")
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set2')
plt.title('Churn Durumuna Göre Aylık Ücret (Box Plot)')
plt.show()

print("4. Grafik: Scatter Plot ")
plt.figure(figsize=(8, 5))
sns.scatterplot(x='tenure', y='TotalCharges', hue='Churn', data=df, alpha=0.6)
plt.title('Süre ve Toplam Ücret İlişkisi (Scatter Plot)')
plt.show()
print("EDA tamamlandi.")