import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./customer_churn.csv")

print("Structure: ", df.shape)

print("Summary Statistics: \n", df.describe())

# Identify and handle missing data
print(df.isna().sum())
print(df['Membership_Type'].unique())
print(df['Payment_Method'].unique())
print(df['Preferred_Content_Type'].unique())
# Data set looks good no missing values


# Exploratory data analysis
df = df.drop(['CustomerID'], axis=1)
df.hist(figsize=(10, 8), bins=20)
plt.tight_layout()
plt.show()

numerical_cols = [
    "Age",
    "Subscription_Length_Months",
    "Watch_Time_Hours",
    "Number_of_Logins",
    "Number_of_Complaints",
    "Resolution_Time_Days"
]

for col in numerical_cols:
    df.boxplot(column=col, by='Churn')
    plt.suptitle("")
    plt.title(f"{col} By Churn")
    plt.ylabel(col)
    plt.show()

correlation = df.corr(numeric_only=True)
plt.figure(figsize=(8,8))
sns.heatmap(correlation, annot=True)
plt.show()