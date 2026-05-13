#%% 
# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#%%
# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv(r"C:\Users\91932\OneDrive\Desktop\HR Analysis\HR_Analytics.csv")

print(df.head())
print(df.info())
#%%
# ===============================
# 3. DATA CLEANING
# ===============================

# Drop employee number (no predictive power)
df.drop("EmployeeNumber", axis=1, inplace=True)

# Encode target variable
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Label Encode categorical columns
le= LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

#Onehot Encoding
df = pd.get_dummies(df, drop_first=True)
#%%
# ===============================
# 4. FEATURE / TARGET SPLIT
# ===============================
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
#%%
# ===============================
# 5. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#%%
# ===============================
# 6. FEATURE SCALING
# ===============================


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#%%
# ===============================
# 7. MODEL TRAINING
# ===============================
#Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)



#%%
# ===============================
# 8. MODEL EVALUATION
# ===============================
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
from sklearn.metrics import classification_report



# %%
# ===============================
# LOGISTIC REGRESSION MODEL
# ===============================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scaling ONLY for Logistic Regression
scaler_lr = StandardScaler()
X_train_scaled = scaler_lr.fit_transform(X_train)
X_test_scaled = scaler_lr.transform(X_test)

from sklearn.impute import SimpleImputer

# Create imputer
imputer = SimpleImputer(strategy='mean')

# Apply on train and test
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# Train Logistic Regression
lr = LogisticRegression(max_iter=2000, solver='liblinear')
lr.fit(X_train_scaled, y_train)

# Prediction
y_pred_lr = lr.predict(X_test_scaled)

# ===============================
# EVALUATION - LOGISTIC REGRESSION
# ===============================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#%%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Create pipeline
# ===============================
# 9. CROSS Validation
# ===============================

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # handle NaN
    ('scaler', StandardScaler()),                          # scale data
    ('model', LogisticRegression(max_iter=2000, solver='liblinear'))
])

# Apply cross validation
scores = cross_val_score(pipeline, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

#%%
# ===============================
# 10. ADA Boost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Base model
lr_base = LogisticRegression(max_iter=2000, solver='liblinear')

# AdaBoost with Logistic Regression
ada_model = AdaBoostClassifier(
    estimator=lr_base,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train
ada_model.fit(X_train_scaled, y_train)

# Predict
y_pred_ada = ada_model.predict(X_test_scaled)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report

print("AdaBoost + LR Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Classification Report:\n", classification_report(y_test, y_pred_ada))
# ===============================

#%%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_ada = confusion_matrix(y_test, y_pred_ada)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm_ada,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Attrition", "Attrition"],
    yticklabels=["No Attrition", "Attrition"]
)

plt.title("Confusion Matrix - AdaBoost + Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#%%
# ===============================
# 11. FEATURE IMPORTANCE
# ===============================
feature_importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[:10], y=feature_importance.index[:10])
plt.title("Top 10 Important Features for Attrition")
plt.show()


# %%
