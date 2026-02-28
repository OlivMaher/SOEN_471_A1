import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./customer_churn.csv")

print("Structure: ", df.shape)
print("Summary Statistics: \n", df.describe())

# Identify and handle missing data
print(df.isna().sum())
print("Negative values: ", sum([i for i in df if i.isnumeric() and i < 0]))
print(df['Membership_Type'].unique())
print(df['Payment_Method'].unique())
print(df['Preferred_Content_Type'].unique())
# Data set looks good no missing values or negative values
# Save the cleaned dataset
df.to_csv("./customer_churn_cleaned.csv", index=False)

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

fig, axes = plt.subplots(2, 3, figsize=(15, 8))  
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    df.boxplot(column=col, by='Churn', ax=axes[i])
    axes[i].set_title(col)

plt.suptitle("") 
plt.tight_layout()
plt.show()

correlation = df.corr(numeric_only=True)
plt.figure(figsize=(8,8))
sns.heatmap(correlation, annot=True)
plt.show()