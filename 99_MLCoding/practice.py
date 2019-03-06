import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline

# データの読み込み
df = pd.read_csv('./data/av_loan_u6lujuX_CVtuZ9i.csv', header=0)
X = df.iloc[:, 1:-1]
y = df.iloc[:, [-1]]

# 正解ラベルの数値数値変換
class_mapping = {'N': 1, 'Y': 0}
y = y.copy()
y.loc[:, 'Loan_Status'] = y['Loan_Status'].map(class_mapping)

# 設問1：正解ラベルyに欠損値がないことを確認せよ。
# yに欠損値がないことを確認
# → 和がゼロなので、isnullの戻り値は全部False
# print(y.isnull().sum())

# Xに欠損値あり
# print(X.isnull().any(axis=0))
# print(X.isnull().sum())

# # X.keys()
# # >> Index(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
# #        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
# #        'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
# #       dtype='object')


# 設問2：特徴量Xをone-hotエンコーディングし、結果をX_oheにセットせよ。
# 設問3：X_oheに含まれる欠損値を補完し、X_finの名前でセットしてください。
# 欠損値を落としたデータセットX_dnaでImputerを学習
# X_impに欠損値を補完したデータをセット
imputer_mf = SimpleImputer(strategy="most_frequent")
X_dna = X.dropna()
X_imp = pd.DataFrame()
imputer_mf.fit(X_dna)

X_ohe = imputer_mf.transform(X)
X_ohe = pd.DataFrame(X_ohe, columns=X.columns)

# OneHotEncodingと標準化を行って辞書に格納
continuous = np.array(['ApplicantIncome',
                       'CoapplicantIncome',
                       'LoanAmount',
                       'Loan_Amount_Term'])

ohe = OneHotEncoder(sparse=False)
mms = MinMaxScaler()
enc = {}
ans = 0
for col in X:
    if any(continuous == col):
        enc[col] = mms.fit_transform(np.array(X_ohe[col]).reshape(-1, 1).astype(np.float64))
        ans += enc[col].shape[1]
    else:
        tmp = ohe.fit_transform(X_ohe[col].factorize()[0].reshape(-1, 1)).astype(np.int64)
        enc[col] = tmp
        ans += enc[col].shape[1]

X_fin = np.hstack([enc['Gender'],
                   enc['Married'],
                   enc['Dependents'],
                   enc['Education'],
                   enc['Self_Employed'],
                   enc['ApplicantIncome'],
                   enc['CoapplicantIncome'],
                   enc['LoanAmount'],
                   enc['Loan_Amount_Term'],
                   enc['Credit_History'],
                   enc['Property_Area']])

# "#### 設問5：ローン審査の予測モデルとして、以下のパイプラインを構成して下さい。\n",
# "- 標準化とロジスティック回帰\n",
# "- 標準化とランダムフォレスト\n",
# "- 標準化と主成分分析とランダムフォレスト"

# num_pipeline = 






