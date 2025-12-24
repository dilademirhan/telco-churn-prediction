import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data.csv")
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

scaler = MinMaxScaler()
cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols] = scaler.fit_transform(df[cols])

X = df.drop('Churn', axis=1)
y = df['Churn']

smote = SMOTE(random_state=30)
X_res, y_res = smote.fit_resample(X, y)

X_df = pd.DataFrame(X_res, columns=X.columns)

y_df = pd.DataFrame(y_res, columns=['Churn'])

clean_data = pd.concat([X_df, y_df], axis=1)

clean_data.to_csv("clean_data.csv", index=False)
