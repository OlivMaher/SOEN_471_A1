import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    ("model", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "model__max_depth": [3,4,5, 6,7, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Cross-Validation Accuracy:", grid.best_score_)
print("Best Hyperparameters:", grid.best_params_)

best_model.fit(X_train, y_train)

# Predict + Evaluate
y_pred = best_model.predict(X_test)

plt.figure(figsize=(50, 10))
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
plot_tree(
    best_model.named_steps["model"],
    feature_names=list(feature_names),
    class_names=[str(c) for c in best_model.named_steps["model"].classes_],
    filled=True,
    rounded=True,
    max_depth=5,
    fontsize=10,
    impurity=False,      # hides gini/entropy
    proportion=True,     # show proportions instead of raw counts (optional)
    label="none",        # removes "gini = ..." style labels line; keeps split text
)
plt.savefig("figures/decision_tree_scaled.png", bbox_inches="tight", dpi=300)
plt.savefig("figures/decision_tree.png")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("Recall Score:", recall)
cm = confusion_matrix(y_test, y_pred)  # use labels=... if you want fixed order
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

disp.plot(values_format="d")
plt.savefig("figures/confusion_matrix.png")
plt.show()

with open("results_decision.txt", "w") as f:
    f.write("Decision Tree Classifier Results\n")
    f.write("===============================\n")
    f.write(f"Best Hyperparameters: {grid.best_params_}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Recall Score: {recall}\n",)