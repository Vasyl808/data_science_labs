import pandas as pd
import numpy as np


input_file = "d:/ai_labs/ml_project/superstore_data.csv"
df_raw = pd.read_csv(input_file)

df = df_raw.copy()

before = len(df)
df = df.drop_duplicates(subset=df.columns.difference(['Id']))
print(f'[Дедублікація]  {before} → {len(df)}  (видалено {before - len(df)})')

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%m/%d/%Y')

CURRENT_YEAR = 2014  # snapshot рік датасету
df['Age'] = CURRENT_YEAR - df['Year_Birth']

before = len(df)
df.dropna(inplace=True)
print(f'[dropna]        {before} → {len(df)}  (видалено {before - len(df)})')

bad_statuses = ['Alone', 'YOLO', 'Absurd']
before = len(df)
df = df[~df['Marital_Status'].isin(bad_statuses)]
print(f'[Marital_Status] {before} → {len(df)}  (видалено {before - len(df)})')

Q1, Q3 = df['Age'].quantile(0.25), df['Age'].quantile(0.75)
IQR = Q3 - Q1
age_mask = (df['Age'] >= Q1 - 1.5*IQR) & (df['Age'] <= Q3 + 1.5*IQR)

income_mask = df['Income'] != 666666.00

before = len(df)
df = df[age_mask & income_mask].reset_index(drop=True)
print(f'[Outliers]      {before} → {len(df)}  (видалено {before - len(df)})')

print(f'\n Фінальний розмір після очищення: {df.shape[0]} × {df.shape[1]}')

SNAPSHOT_DATE = df['Dt_Customer'].max() + pd.DateOffset(days=1)
print(f'Snapshot date: {SNAPSHOT_DATE.date()}')

df['Customer_Tenure_Days'] = (SNAPSHOT_DATE - df['Dt_Customer']).dt.days

df['Enroll_Year']  = df['Dt_Customer'].dt.year
df['Enroll_Month'] = df['Dt_Customer'].dt.month

df['Enroll_Quarter'] = df['Dt_Customer'].dt.quarter

df['Enrolled_H1'] = (df['Enroll_Month'] <= 6).astype(int)

print('Часові фічі:')
print(df[['Dt_Customer','Customer_Tenure_Days','Enroll_Year','Enroll_Month','Enroll_Quarter','Enrolled_H1']].head())

df = df.drop(columns=['Enroll_Year','Enroll_Month','Enroll_Quarter','Enrolled_H1'])

df.drop(columns=['NumWebVisitsMonth', 'NumDealsPurchases'], inplace=True)

df.drop(columns=['Year_Birth'], inplace=True)

mnt_cols = ['MntWines', 'MntMeatProducts', 'MntFruits',
            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

df['TotalMnt'] = df[mnt_cols].sum(axis=1)
df['TotalPurchases'] = df[['NumWebPurchases', 
                            'NumCatalogPurchases', 
                            'NumStorePurchases']].sum(axis=1)

df['WineRatio'] = df['MntWines'] / (df['Income'] + 1)
df['MeatRatio'] = df['MntMeatProducts'] / (df['Income'] + 1)
df['PremiumRatio'] = (df['MntWines'] + df['MntMeatProducts']) / (df['TotalMnt'] + 1)

df['CatalogShare'] = df['NumCatalogPurchases'] / (df['TotalPurchases'] + 1)

log_cols = [
    'MntWines', 'MntMeatProducts', 'MntFruits', 'MntFishProducts', 
    'MntSweetProducts', 'MntGoldProds', 'NumCatalogPurchases', 'NumWebPurchases'
]
for col in log_cols:
    df[f'{col}_log'] = np.log1p(df[col])

drop_cols = [
    'MntWines',
    'MntMeatProducts', 
    'MntFruits',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds',
    'NumCatalogPurchases',
    'NumWebPurchases',
    'Dt_Customer'
]

df.drop(columns=drop_cols, inplace=True)

df.drop(columns=['Complain'], inplace=True)

edu_order = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
df['Education'] = df['Education'].map(edu_order)

df['is_alone'] = df['Marital_Status'].isin(
    ['Single', 'Widow', 'Divorced']
).astype(int)

df.drop(columns=['Marital_Status'], inplace=True)

final_cols = [
    'Id', 'Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 
    'NumStorePurchases', 'Response', 'Age', 'Customer_Tenure_Days', 
    'MntWines_log', 'MntMeatProducts_log', 'MntFruits_log', 
    'MntFishProducts_log', 'MntSweetProducts_log', 'MntGoldProds_log', 
    'NumCatalogPurchases_log', 'NumWebPurchases_log', 'TotalMnt', 
    'TotalPurchases', 'WineRatio', 'MeatRatio', 'PremiumRatio', 
    'CatalogShare', 'is_alone'
]
df = df[final_cols]

print(f"\nFinal dataset shape: {df.shape}")
print(df.head())
