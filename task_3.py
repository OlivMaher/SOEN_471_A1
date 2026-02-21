import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


df = pd.read_csv("customer_churn_cleaned.csv")
target_key = "Churn"
if target_key not in df.columns:
    raise ValueError(f"Target key '{target_key}' not found in the dataset.")

X = df.drop(columns=[target_key]) # churn is our target variable
y = df[target_key]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.columns.difference(cat_cols)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])

print("Training Random Forest Classifier...")
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

importances = clf.named_steps["model"].feature_importances_
feature_names = clf.named_steps["preprocess"].get_feature_names_out()
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)





print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.title("Confusion Matrix for Random Forest Classifier")
disp.plot()
plt.savefig("figures/random_forest_confusion_matrix.png")
plt.show()

# Plot the results
fig, ax = plt.subplots()
feature_importance_df.set_index("Feature")["Importance"].plot.bar(ax=ax)
ax.set_title("Feature Importances (MDI)")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig("figures/random_forest_feature_importance.png")
plt.show()

with open("results_random.txt", "w") as f:
    f.write("Random Forest Classifier Results\n")
    f.write(str(classification_report(y_test, predictions)))
    f.write(f"\nAccuracy: {accuracy_score(y_test, predictions)}")
    f.write(f"\nPrecision: {precision_score(y_test, predictions)}")
    f.write(f"\nRecall: {recall_score(y_test, predictions)}")
    f.write(f"\nF1 Score: {f1_score(y_test, predictions)}")
    f.write("\n\nFeature Importances:\n")
    f.write(feature_importance_df.to_string(index=False))