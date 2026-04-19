# ============================================================
# Fraudulent Loan Application Detection using Logistic Regression
# Dataset: Kaggle Loan Dataset
# Author: ML Mini Project
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')
 
# ============================================================
# STEP 1: Load Dataset
# ============================================================
# Download from Kaggle: https://www.kaggle.com/datasets/
#   altruistdelhite04/loan-prediction-problem-dataset
 
df = pd.read_csv('loan_train.csv')
 
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape          : {df.shape}")
print(f"Total Records  : {df.shape[0]}")
print(f"Total Features : {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
 
# ============================================================
# STEP 2: Data Preprocessing
# ============================================================
 
# Drop Loan_ID column (not useful for prediction)
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)
 
# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
 
print("\nMissing values after imputation:")
print(df.isnull().sum())
 
# Encode Dependents
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
 
# Label Encoding for categorical variables
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed',
                    'Property_Area', 'Loan_Status']
 
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
 
# Loan_Status: Y=1 (Approved), N=0 (Rejected/Fraudulent risk)
print("\nEncoded Dataset:")
print(df.head())
 
# Feature Engineering
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)
 
# ============================================================
# STEP 3: Exploratory Data Analysis (EDA)
# ============================================================
 
plt.figure(figsize=(16, 12))
 
# Loan Status Distribution
plt.subplot(2, 3, 1)
df['Loan_Status'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Loan Status Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
plt.ylabel('Count')
plt.xticks(rotation=0)
 
# Credit History vs Loan Status
plt.subplot(2, 3, 2)
sns.countplot(x='Credit_History', hue='Loan_Status', data=df,
              palette=['#e74c3c', '#2ecc71'])
plt.title('Credit History vs Loan Status', fontsize=13, fontweight='bold')
plt.legend(['Rejected', 'Approved'])
 
# Gender vs Loan Status
plt.subplot(2, 3, 3)
sns.countplot(x='Gender', hue='Loan_Status', data=df,
              palette=['#e74c3c', '#2ecc71'])
plt.title('Gender vs Loan Status', fontsize=13, fontweight='bold')
 
# Applicant Income Distribution
plt.subplot(2, 3, 4)
df['ApplicantIncome'].plot(kind='hist', bins=40, color='#3498db', edgecolor='black')
plt.title('Applicant Income Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Applicant Income')
 
# Loan Amount Distribution
plt.subplot(2, 3, 5)
df['LoanAmount'].plot(kind='hist', bins=40, color='#9b59b6', edgecolor='black')
plt.title('Loan Amount Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Loan Amount (in thousands)')
 
# Property Area vs Loan Status
plt.subplot(2, 3, 6)
sns.countplot(x='Property_Area', hue='Loan_Status', data=df,
              palette=['#e74c3c', '#2ecc71'])
plt.title('Property Area vs Loan Status', fontsize=13, fontweight='bold')
 
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nEDA plots saved as 'eda_plots.png'")
 
# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, annot_kws={"size": 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved.")
 
# ============================================================
# STEP 4: Feature Selection & Train-Test Split
# ============================================================
 
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
 
print(f"\nFeatures used: {list(X.columns)}")
print(f"Target variable: Loan_Status")
print(f"Class distribution:\n{y.value_counts()}")
 
# Handle class imbalance using oversampling
X_combined = pd.concat([X, y], axis=1)
minority = X_combined[X_combined['Loan_Status'] == 0]
majority = X_combined[X_combined['Loan_Status'] == 1]
 
minority_upsampled = resample(minority, replace=True,
                               n_samples=len(majority), random_state=42)
upsampled = pd.concat([majority, minority_upsampled])
 
X = upsampled.drop('Loan_Status', axis=1)
y = upsampled['Loan_Status']
 
print(f"\nAfter oversampling - Class distribution:\n{y.value_counts()}")
 
# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
 
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
 
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# ============================================================
# STEP 5: Model Training - Logistic Regression
# ============================================================
 
model = LogisticRegression(
    C=1.0,               # Regularization parameter
    solver='lbfgs',      # Optimization algorithm
    max_iter=1000,       # Maximum iterations
    class_weight='balanced',
    random_state=42
)
 
model.fit(X_train_scaled, y_train)
print("\nLogistic Regression Model trained successfully!")
 
# ============================================================
# STEP 6: Model Evaluation
# ============================================================
 
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
 
accuracy  = accuracy_score(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_proba)
 
print("\n" + "=" * 60)
print("MODEL PERFORMANCE RESULTS")
print("=" * 60)
print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"ROC-AUC Score  : {roc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Fraudulent/Rejected', 'Legitimate/Approved']))
 
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['Rejected', 'Approved'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
 
# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='#2ecc71', lw=2.5,
             label=f'ROC Curve (AUC = {roc_score:.4f})')
axes[1].plot([0, 1], [0, 1], color='#e74c3c', linestyle='--', lw=1.5,
             label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#2ecc71')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nModel result plots saved as 'model_results.png'")
 
# Feature Importance (Coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(model.coef_[0])
}).sort_values('Coefficient', ascending=False)
 
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance,
            palette='Blues_r')
plt.title('Feature Importance (Logistic Regression Coefficients)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved.")
 
# ============================================================
# STEP 7: Sample Prediction (Single Applicant)
# ============================================================
 
print("\n" + "=" * 60)
print("SAMPLE FRAUD DETECTION PREDICTION")
print("=" * 60)
 
# Sample: Married male, graduate, no self-employment,
#         income=5000, coapplicant=1500, loan=150, term=360,
#         credit history=0 (NO CREDIT HISTORY - high fraud risk)
sample = pd.DataFrame([[
    1, 1, 2, 0, 4000, 1500, 128, 360, 0, 2,
    5500, 128/360, 5500 - (128/360)*1000
]], columns=X.columns)
 
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)
 
print(f"Prediction  : {'APPROVED (Legitimate)' if prediction[0] == 1 else 'REJECTED (Possible Fraud/Risk)'}")
print(f"Probability of Approval : {probability[0][1]*100:.2f}%")
print(f"Probability of Rejection: {probability[0][0]*100:.2f}%")
 
print("\n" + "=" * 60)
print("EXECUTION COMPLETE")
print("=" * 60)

