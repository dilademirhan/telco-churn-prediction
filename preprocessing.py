import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

# 1. Veriyi Oku ve Gereksizleri At
df = pd.read_csv("data.csv")
df.drop('customerID', axis=1, inplace=True)

# 2. Sayısal Düzeltme ve Eksik Veri
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# 3. Kategorik Verileri Sayıya Çevir (Encoding)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 4. Sayıları Küçült (0-1 Arası Normalizasyon)
scaler = MinMaxScaler()
cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols] = scaler.fit_transform(df[cols])

# 5. Dengesizliği Gider (SMOTE)
X = df.drop('Churn', axis=1)
y = df['Churn']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Adım 1: Özellikleri tablo yap
X_df = pd.DataFrame(X_res, columns=X.columns)

# Adım 2: Hedef değişkeni tablo yap
y_df = pd.DataFrame(y_res, columns=['Churn'])

# Adım 3: İkisini yan yana birleştir
clean_data = pd.concat([X_df, y_df], axis=1)

clean_data.to_csv("clean_data.csv", index=False)
print("İşlem tamam. Veri temizlendi, dengelendi ve kaydedildi.")