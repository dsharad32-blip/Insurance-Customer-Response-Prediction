# load pre-trained data
# Imputing and encoding
# Train Test Split
# Model training And Hyperparameter tuning
# Model Evaluation & select the best model
# dump the best model

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import joblib

print("Imports Successful")


# Load Clean Data
CLEANED_DATA_PATH = r"C:\Users\dshar\OneDrive\Pictures\Desktop\Insurance_Customer_Response_Pred\DATA\Cleaned_data.csv"
data = pd.read_csv(CLEANED_DATA_PATH)

#Copy of Pre-processed data
df = data.copy()


# Feature and Target
X = df.drop(columns=["Response"])
y = df["Response"]


# Column Categorization
num_cols = X.select_dtypes(include="number").columns.to_list()
cat_cols = X.select_dtypes(include="object").columns.to_list()


# Numerical Pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Pipeline
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"))
])


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num_transformer", num_pipeline, num_cols),
        ("cat_transformer", cat_pipeline, cat_cols)
    ]
)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Model Dictionary
models = {

    "LogisticRegression":(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        ),
        {
            "model__C":[0.001,0.01,0.1,1,10],
            "model__penalty":["l1","l2"],
            "model__solver":["liblinear"],
            "model__class_weight":[None,"balanced"]
        }
    ),

    "DecisionTree":(
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__max_depth":[None,5,10,20],
            "model__min_samples_split":[2,5,10]
        }
    ),

    "RandomForest":(
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "model__n_estimators":[100,200],
            "model__max_depth":[None,10,20]
        }
    ),

    "XGBoost":(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=7
        ),
        {
            "model__n_estimators":[100,200],
            "model__learning_rate":[0.01,0.1],
            "model__max_depth":[3,6],
            "model__subsample":[0.8,1],
            "model__colsample_bytree":[0.8,1]
        }
    )
}

#-------------------------------------------------------------------------------------------------------
result = []
best_model = None
best_score = 0


# Training Loop
for name,(model,param) in models.items():

    print(f"\nTraining {name}...")

    pipe = Pipeline(steps=[
        ("preprocess",preprocessor),
        ("model",model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train,y_train)

    print("Best Parameters:", grid.best_params_)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:,1]


    # Metrics
    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_prob)


    result.append({
        "Model":name,
        "Accuracy":acc,
        "Precision":precision,
        "Recall":recall,
        "F1_Score":f1,
        "ROC_AUC":roc_auc
    })


    # Best Model Selection
    if f1 > best_score:
        best_score = f1
        best_model = grid.best_estimator_
        best_model_name = name
        best_y_pred = y_pred
        best_y_prob = y_prob


# Model Comparison
result_df = pd.DataFrame(result)
result_df = result_df.sort_values(by="F1_Score",ascending=False)

print("\nModel Comparison")
print(result_df)

print("\nBest Model:",result_df.iloc[0]["Model"])


# Confusion Matrix
cm = confusion_matrix(y_test,best_y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Response","Response"],
            yticklabels=["No Response","Response"])

plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()


# ROC Curve
fpr,tpr,thresholds = roc_curve(y_test,best_y_prob)
roc_auc = roc_auc_score(y_test,best_y_prob)

plt.figure(figsize=(6,5))

plt.plot(fpr,tpr,
         label=f"AUC = {roc_auc:.3f}",
         linewidth=2)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({best_model_name})")

plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# Feature Importance
model = best_model.named_steps["model"]

if hasattr(model,"feature_importances_"):

    importances = model.feature_importances_

    feature_names = num_cols + list(
        best_model.named_steps["preprocess"]
        .named_transformers_["cat_transformer"]
        .named_steps["ohe"]
        .get_feature_names_out(cat_cols)
    )

    importance_df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":importances
    }).sort_values(by="Importance",ascending=False).head(15)

    plt.figure(figsize=(8,6))
    sns.barplot(data=importance_df,
                x="Importance",
                y="Feature")

    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


# Save Best Model
MODEL_PATH = r"C:\Users\dshar\OneDrive\Pictures\Desktop\Insurance_Customer_Response_Pred\Models\Best_Model.pkl"
joblib.dump(best_model, MODEL_PATH, compress=3)  

print("\nBest Model Saved Successfully")


# Save Model Comparison for Dashboard
RESULT_PATH = r"C:\Users\dshar\OneDrive\Pictures\Desktop\Insurance_Customer_Response_Pred\Models\model_results.csv"

result_df.to_csv(RESULT_PATH,index=False)

print("Model Results Saved")