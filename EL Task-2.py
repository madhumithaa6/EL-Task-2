import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Titanic-Dataset.csv")
print("Summary Statistics:")
print(df.describe())
numeric_cols = ['Age','Fare','SibSp','Parch']
for col in numeric_cols:
    plt.figure(figsize=(6,3))
    plt.hist(df[col],bins=20,color='skyblue',edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
for col in numeric_cols:
    plt.figure(figsize=(6,2))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
numeric_df=df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
sns.pairplot(df[['Survived','Age','Fare','SibSp','Parch']])
plt.suptitle("Pairplot of Key Features",y=1.02)
plt.show()
print("Observations:")
print("Fare is positively skewed (some high-ticket passengers).")
print("Age has a normal distribution with some outliers.")
print("'Sex_male=0',survived more  (females survived more).")
print("Family size is mostly around 0 or 1.")
print("Feature Inference:")
print("Higher fare passengers had higher survival chances.")
print("Younger passengers had higher survival chances.")
print("Males were less likely to survive.")
print(":identified from boxplot,histogram and correlation")
