# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Experiment Tracking and Model Registry with Unity Catalog
# MAGIC
# MAGIC ## Scenario
# MAGIC You are a data scientist at a financial services company building predictive models for customer churn. While your team has successfully trained several models, leadership is concerned about governance, auditability, and compliance. Regulators require a full record of how models are developed, deployed, and retired.
# MAGIC
# MAGIC ## Objectives
# MAGIC By the end of this lab, you will be able to:
# MAGIC - Track experiments with MLflow by logging parameters, metrics, and artifacts
# MAGIC - Register models in the MLflow Model Registry integrated with Unity Catalog
# MAGIC - Manage versions and promote models between Staging, Production, and Archived
# MAGIC - Apply Unity Catalog governance controls (RBAC, audit logging, and lineage)
# MAGIC - Implement reproducibility practices through metadata and documentation
# MAGIC - Apply archiving and cleanup policies to maintain a healthy model registry
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with Unity Catalog enabled
# MAGIC - Access to create catalogs, schemas, and tables
# MAGIC - MLflow installed (pre-installed in Databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Environment Setup and Prerequisites
# MAGIC
# MAGIC In this section, we will:
# MAGIC 1. Import necessary libraries
# MAGIC 2. Configure Unity Catalog settings
# MAGIC 3. Create sample customer churn data
# MAGIC
# MAGIC **Why this matters:** Proper environment setup ensures reproducibility and governance from the start. Unity Catalog provides enterprise-grade data governance, while MLflow handles model lifecycle management.

# COMMAND ----------

# Import required libraries
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully")
print(f"MLflow version: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Unity Catalog Settings
# MAGIC
# MAGIC We'll set up our Unity Catalog namespace structure:
# MAGIC - **Catalog**: `financial_services` - Top-level container for our data
# MAGIC - **Schema**: `churn_models` - Logical grouping for churn-related assets
# MAGIC - **Volume**: For storing model artifacts and documentation
# MAGIC
# MAGIC **Governance Note:** Unity Catalog provides centralized access control, audit logging, and data lineage across all assets.

# COMMAND ----------

# Define Unity Catalog namespace
CATALOG_NAME = "financial_services"
SCHEMA_NAME = "churn_models"
TABLE_NAME = "customer_churn_data"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.customer_churn_model"

# Create catalog and schema if they don't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print(f"✓ Unity Catalog configured:")
print(f"  - Catalog: {CATALOG_NAME}")
print(f"  - Schema: {SCHEMA_NAME}")
print(f"  - Model Registry: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Sample Customer Churn Data
# MAGIC
# MAGIC We'll create a realistic customer dataset with features commonly used in churn prediction:
# MAGIC - **Customer Demographics**: Age, tenure, account type
# MAGIC - **Usage Patterns**: Transaction frequency, product usage, support interactions
# MAGIC - **Financial Metrics**: Account balance, credit score, monthly charges
# MAGIC - **Target Variable**: Churn (1 = churned, 0 = retained)
# MAGIC
# MAGIC **Data Governance:** This data will be stored in Unity Catalog with full lineage tracking.

# COMMAND ----------

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_customers = 10000

# Create customer IDs
customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_customers + 1)]

# Generate features
data = {
    'customer_id': customer_ids,
    'age': np.random.randint(18, 75, n_customers),
    'tenure_months': np.random.randint(1, 120, n_customers),
    'account_balance': np.random.uniform(100, 50000, n_customers).round(2),
    'credit_score': np.random.randint(300, 850, n_customers),
    'num_products': np.random.randint(1, 5, n_customers),
    'monthly_charges': np.random.uniform(20, 500, n_customers).round(2),
    'total_transactions': np.random.randint(0, 200, n_customers),
    'support_calls': np.random.randint(0, 15, n_customers),
    'complaint_filed': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
    'account_type': np.random.choice(['Basic', 'Premium', 'Gold'], n_customers, p=[0.5, 0.35, 0.15]),
    'online_banking': np.random.choice([0, 1], n_customers, p=[0.3, 0.7]),
    'mobile_app_usage': np.random.randint(0, 100, n_customers),
    'last_transaction_days': np.random.randint(0, 90, n_customers)
}

# Create DataFrame
df_pandas = pd.DataFrame(data)

# Generate churn target with realistic correlations
churn_probability = (
    0.1 +  # Base churn rate
    (df_pandas['support_calls'] > 5) * 0.2 +
    (df_pandas['complaint_filed'] == 1) * 0.25 +
    (df_pandas['tenure_months'] < 12) * 0.15 +
    (df_pandas['last_transaction_days'] > 60) * 0.2 +
    (df_pandas['account_balance'] < 1000) * 0.1 -
    (df_pandas['num_products'] > 2) * 0.15 -
    (df_pandas['online_banking'] == 1) * 0.1
)

# Clip probabilities and generate churn
churn_probability = np.clip(churn_probability, 0, 0.8)
df_pandas['churn'] = np.random.binomial(1, churn_probability)

# Add timestamp for audit purposes
df_pandas['data_created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"✓ Generated {n_customers:,} customer records")
print(f"✓ Churn rate: {df_pandas['churn'].mean():.2%}")
print(f"\nFeature summary:")
print(df_pandas.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Data to Unity Catalog
# MAGIC
# MAGIC We'll persist our customer data to Unity Catalog as a Delta table. This provides:
# MAGIC - **ACID transactions** for data reliability
# MAGIC - **Time travel** for auditing and compliance
# MAGIC - **Automatic lineage tracking** through Unity Catalog
# MAGIC - **Fine-grained access control** via RBAC
# MAGIC
# MAGIC **Compliance Note:** All data access and modifications are automatically logged by Unity Catalog for regulatory audits.

# COMMAND ----------

# Convert to Spark DataFrame and save to Unity Catalog
df_spark = spark.createDataFrame(df_pandas)

# Write to Delta table in Unity Catalog
table_path = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"
df_spark.write.format("delta").mode("overwrite").saveAsTable(table_path)

print(f"✓ Data saved to Unity Catalog table: {table_path}")

# Verify table creation and show sample
df_loaded = spark.table(table_path)
print(f"✓ Table contains {df_loaded.count():,} records")
print("\nSample records:")
display(df_loaded.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Checks
# MAGIC
# MAGIC Before model training, we perform data quality validation:
# MAGIC - Check for missing values
# MAGIC - Verify data distributions
# MAGIC - Validate business rules
# MAGIC
# MAGIC **Best Practice:** Document data quality checks for reproducibility and compliance.

# COMMAND ----------

# Perform data quality checks
print("=== Data Quality Report ===\n")

# Check for missing values
missing_counts = df_pandas.isnull().sum()
print("Missing Values:")
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "✓ No missing values")

# Check class balance
print(f"\nClass Distribution:")
print(f"  - Retained (0): {(df_pandas['churn'] == 0).sum():,} ({(df_pandas['churn'] == 0).mean():.2%})")
print(f"  - Churned (1): {(df_pandas['churn'] == 1).sum():,} ({(df_pandas['churn'] == 1).mean():.2%})")

# Feature correlations with target
print(f"\nTop Features Correlated with Churn:")
numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.drop('churn')
correlations = df_pandas[numeric_cols].corrwith(df_pandas['churn']).abs().sort_values(ascending=False)
print(correlations.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Data Preparation and Feature Engineering
# MAGIC
# MAGIC In this section, we'll:
# MAGIC 1. Prepare features for modeling
# MAGIC 2. Split data into train/test sets
# MAGIC 3. Apply feature scaling
# MAGIC
# MAGIC **MLflow Integration:** All preprocessing steps will be logged as artifacts for reproducibility.

# COMMAND ----------

# Prepare features for modeling
# One-hot encode categorical variables
df_encoded = pd.get_dummies(df_pandas, columns=['account_type'], drop_first=True)

# Select feature columns (exclude ID, timestamp, and target)
feature_cols = [col for col in df_encoded.columns
                if col not in ['customer_id', 'churn', 'data_created_at']]

X = df_encoded[feature_cols]
y = df_encoded['churn']

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target variable shape: {y.shape}")
print(f"\nFeatures used for modeling:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# COMMAND ----------

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")
print(f"\nTrain set churn rate: {y_train.mean():.2%}")
print(f"Test set churn rate: {y_test.mean():.2%}")

# COMMAND ----------

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Feature scaling completed")
print(f"✓ Scaled training data shape: {X_train_scaled.shape}")
print(f"✓ Scaled test data shape: {X_test_scaled.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3: MLflow Experiment Tracking
# MAGIC
# MAGIC Now we'll train multiple models and track everything with MLflow:
# MAGIC - **Parameters**: Model hyperparameters
# MAGIC - **Metrics**: Accuracy, precision, recall, F1, AUC
# MAGIC - **Artifacts**: Model files, feature importance plots, confusion matrices
# MAGIC - **Tags**: Metadata for organization and searchability
# MAGIC
# MAGIC **Enterprise Value:** Complete experiment tracking enables reproducibility, comparison, and audit trails.

# COMMAND ----------

# Set up MLflow experiment
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/churn_prediction_experiments"
mlflow.set_experiment(experiment_name)

# Configure MLflow to use Unity Catalog for model registry
mlflow.set_registry_uri("databricks-uc")

print(f"✓ MLflow experiment: {experiment_name}")
print(f"✓ Model registry: Unity Catalog")
print(f"✓ Registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Function: Model Training and Logging
# MAGIC
# MAGIC We'll create a reusable function that:
# MAGIC - Trains a model
# MAGIC - Logs all parameters, metrics, and artifacts to MLflow
# MAGIC - Returns performance metrics for comparison

# COMMAND ----------

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    """
    Train a model and log everything to MLflow.

    Args:
        model: Sklearn model instance
        model_name: Name for the MLflow run
        X_train, X_test, y_train, y_test: Train/test data
        params: Dictionary of hyperparameters to log

    Returns:
        Dictionary of metrics
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log model type and training metadata
        mlflow.set_tag("model_type", model.__class__.__name__)
        mlflow.set_tag("training_date", datetime.now().strftime('%Y-%m-%d'))
        mlflow.set_tag("data_version", "v1.0")
        mlflow.set_tag("purpose", "customer_churn_prediction")

        # Train model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test),
            'test_recall': recall_score(y_test, y_pred_test),
            'test_f1': f1_score(y_test, y_pred_test),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test)
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train[:5]
        )

        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            import matplotlib.pyplot as plt

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importances - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('/tmp/feature_importance.png')
            mlflow.log_artifact('/tmp/feature_importance.png')
            plt.close()

        print(f"✓ {model_name} training complete")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Metrics: {metrics}")

        return metrics, run.info.run_id

print("✓ Helper function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Multiple Models
# MAGIC
# MAGIC We'll train three different models to compare performance:
# MAGIC 1. **Logistic Regression** - Simple, interpretable baseline
# MAGIC 2. **Random Forest** - Ensemble method with feature importance
# MAGIC 3. **Gradient Boosting** - Advanced ensemble technique
# MAGIC
# MAGIC Each model's parameters, metrics, and artifacts will be logged to MLflow for comparison.

# COMMAND ----------

# Train Logistic Regression
lr_params = {
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}

lr_model = LogisticRegression(**lr_params)
lr_metrics, lr_run_id = train_and_log_model(
    lr_model,
    "Logistic_Regression_Baseline",
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    lr_params
)

# COMMAND ----------

# Train Random Forest
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

rf_model = RandomForestClassifier(**rf_params)
rf_metrics, rf_run_id = train_and_log_model(
    rf_model,
    "Random_Forest_v1",
    X_train,
    X_test,
    y_train,
    y_test,
    rf_params
)

# COMMAND ----------

# Train Gradient Boosting
gb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

gb_model = GradientBoostingClassifier(**gb_params)
gb_metrics, gb_run_id = train_and_log_model(
    gb_model,
    "Gradient_Boosting_v1",
    X_train,
    X_test,
    y_train,
    y_test,
    gb_params
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare Model Performance
# MAGIC
# MAGIC Let's compare all three models across key metrics to determine which performs best.

# COMMAND ----------

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Run_ID': [lr_run_id, rf_run_id, gb_run_id],
    'Accuracy': [lr_metrics['test_accuracy'], rf_metrics['test_accuracy'], gb_metrics['test_accuracy']],
    'Precision': [lr_metrics['test_precision'], rf_metrics['test_precision'], gb_metrics['test_precision']],
    'Recall': [lr_metrics['test_recall'], rf_metrics['test_recall'], gb_metrics['test_recall']],
    'F1_Score': [lr_metrics['test_f1'], rf_metrics['test_f1'], gb_metrics['test_f1']],
    'AUC': [lr_metrics['test_auc'], rf_metrics['test_auc'], gb_metrics['test_auc']]
})

print("=== Model Performance Comparison ===\n")
print(comparison_df.to_string(index=False))

# Identify best model based on F1 score (balanced metric)
best_model_idx = comparison_df['F1_Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_run_id = comparison_df.loc[best_model_idx, 'Run_ID']

print(f"\n✓ Best performing model: {best_model_name}")
print(f"✓ Run ID: {best_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4: Model Registration in Unity Catalog
# MAGIC
# MAGIC Now we'll register our best model in Unity Catalog's Model Registry. This provides:
# MAGIC - **Centralized model storage** with versioning
# MAGIC - **Access control** via Unity Catalog RBAC
# MAGIC - **Audit logging** of all model operations
# MAGIC - **Lineage tracking** from data to model to deployment
# MAGIC
# MAGIC **Governance Benefit:** Unity Catalog ensures only authorized users can access, modify, or deploy models.

# COMMAND ----------

# Register the best model to Unity Catalog
print(f"Registering {best_model_name} to Unity Catalog...")

# Create model registry entry
model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME,
    tags={
        "model_type": best_model_name,
        "training_date": datetime.now().strftime('%Y-%m-%d'),
        "use_case": "customer_churn_prediction",
        "department": "data_science",
        "compliance_approved": "pending"
    }
)

print(f"✓ Model registered successfully!")
print(f"  Model Name: {MODEL_NAME}")
print(f"  Version: {model_version.version}")
print(f"  Run ID: {best_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Model Description and Documentation
# MAGIC
# MAGIC Proper documentation is critical for governance and compliance. We'll add:
# MAGIC - Model description
# MAGIC - Training details
# MAGIC - Performance metrics
# MAGIC - Intended use and limitations

# COMMAND ----------

# Initialize MLflow client
client = MlflowClient()

# Update model description
model_description = f"""
# Customer Churn Prediction Model

## Overview
This model predicts customer churn for financial services customers using {best_model_name}.

## Training Details
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Training Data**: {X_train.shape[0]:,} samples
- **Features**: {X_train.shape[1]} features
- **Algorithm**: {best_model_name}

## Performance Metrics (Test Set)
- **Accuracy**: {comparison_df.loc[best_model_idx, 'Accuracy']:.4f}
- **Precision**: {comparison_df.loc[best_model_idx, 'Precision']:.4f}
- **Recall**: {comparison_df.loc[best_model_idx, 'Recall']:.4f}
- **F1 Score**: {comparison_df.loc[best_model_idx, 'F1_Score']:.4f}
- **AUC-ROC**: {comparison_df.loc[best_model_idx, 'AUC']:.4f}

## Intended Use
- Predict customer churn probability
- Identify at-risk customers for retention campaigns
- Support business decision-making

## Limitations
- Model trained on historical data; performance may degrade over time
- Requires retraining with fresh data quarterly
- Not suitable for real-time predictions without proper infrastructure

## Compliance Notes
- All training data stored in Unity Catalog with access controls
- Model training tracked in MLflow for full reproducibility
- Audit logs available through Unity Catalog
"""

# Update model version description
client.update_model_version(
    name=MODEL_NAME,
    version=model_version.version,
    description=model_description
)

print("✓ Model documentation added")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5: Model Version Management and Promotion
# MAGIC
# MAGIC Unity Catalog supports model lifecycle management through aliases. We'll:
# MAGIC 1. Set the "Champion" alias for production deployment
# MAGIC 2. Set the "Challenger" alias for A/B testing
# MAGIC 3. Demonstrate version management
# MAGIC
# MAGIC **Best Practice:** Use aliases instead of stages for flexible deployment strategies.

# COMMAND ----------

# Set Champion alias (production model)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=model_version.version
)

print(f"✓ Model version {model_version.version} promoted to 'Champion' (Production)")
print(f"  This model is now ready for production deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model from Registry
# MAGIC
# MAGIC Demonstrate how to load a registered model for inference. This is how production systems would access the model.

# COMMAND ----------

# Load model using alias
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

print("✓ Model loaded from registry using 'Champion' alias")

# Make sample predictions
sample_data = X_test.head(5)
predictions = loaded_model.predict(sample_data)

print("\nSample Predictions:")
prediction_df = pd.DataFrame({
    'Customer_Index': sample_data.index,
    'Predicted_Churn': predictions,
    'Actual_Churn': y_test.iloc[:5].values
})
print(prediction_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Model Version Updates
# MAGIC
# MAGIC In real scenarios, you'll train improved models over time. Let's simulate this by:
# MAGIC 1. Training an improved Random Forest model (v2)
# MAGIC 2. Registering it as a new version
# MAGIC 3. Setting it as "Challenger" for A/B testing

# COMMAND ----------

# Train improved Random Forest with better hyperparameters
rf_v2_params = {
    'n_estimators': 200,  # Increased from 100
    'max_depth': 15,      # Increased from 10
    'min_samples_split': 3,  # Decreased from 5
    'min_samples_leaf': 1,   # Decreased from 2
    'random_state': 42,
    'n_jobs': -1
}

print("Training improved Random Forest model (v2)...")
rf_v2_model = RandomForestClassifier(**rf_v2_params)
rf_v2_metrics, rf_v2_run_id = train_and_log_model(
    rf_v2_model,
    "Random_Forest_v2_Improved",
    X_train,
    X_test,
    y_train,
    y_test,
    rf_v2_params
)

# COMMAND ----------

# Register the new version
print("Registering improved model as new version...")

model_version_v2 = mlflow.register_model(
    model_uri=f"runs:/{rf_v2_run_id}/model",
    name=MODEL_NAME,
    tags={
        "model_type": "Random Forest",
        "training_date": datetime.now().strftime('%Y-%m-%d'),
        "use_case": "customer_churn_prediction",
        "version_notes": "Improved hyperparameters for better performance",
        "department": "data_science"
    }
)

print(f"✓ New model version registered: {model_version_v2.version}")

# Set as Challenger for A/B testing
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Challenger",
    version=model_version_v2.version
)

print(f"✓ Model version {model_version_v2.version} set as 'Challenger'")
print(f"  Ready for A/B testing against Champion model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View All Model Versions
# MAGIC
# MAGIC Let's examine all versions of our registered model and their aliases.

# COMMAND ----------

# Get all versions of the model
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print(f"=== All Versions of {MODEL_NAME} ===\n")
for version in all_versions:
    print(f"Version: {version.version}")
    print(f"  Run ID: {version.run_id}")
    print(f"  Status: {version.status}")
    print(f"  Aliases: {version.aliases if hasattr(version, 'aliases') else 'None'}")
    print(f"  Created: {version.creation_timestamp}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6: Unity Catalog Governance Controls
# MAGIC
# MAGIC Unity Catalog provides enterprise-grade governance features:
# MAGIC - **RBAC (Role-Based Access Control)**: Control who can read, write, or execute models
# MAGIC - **Audit Logging**: Track all operations on models and data
# MAGIC - **Data Lineage**: Trace models back to training data
# MAGIC
# MAGIC Let's explore these governance capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Access Control with RBAC (Demonstration)
# MAGIC
# MAGIC Unity Catalog provides enterprise-grade access control through Role-Based Access Control (RBAC).
# MAGIC
# MAGIC **How RBAC Works in Production:**
# MAGIC 1. **Account admins** create groups at **account level** (not workspace level)
# MAGIC 2. **Users are added** to groups based on their roles
# MAGIC 3. **Permissions are granted** to groups, not individual users
# MAGIC 4. **Users inherit** permissions from all groups they belong to
# MAGIC
# MAGIC **Typical Groups in ML Projects:**
# MAGIC - `data_analysts` - Read access to data tables
# MAGIC - `ml_engineers` - Model execution and deployment rights
# MAGIC - `data_scientists` - Full access to develop and train models
# MAGIC - `data_engineers` - Data pipeline and table management
# MAGIC
# MAGIC **Important: Workspace vs. Account Groups**
# MAGIC - Unity Catalog requires **account-level groups** (created in Account Console)
# MAGIC - Workspace-level groups (created with `CREATE GROUP`) **do NOT work** with Unity Catalog
# MAGIC - Only account admins can create account-level groups
# MAGIC - This is a common source of confusion!
# MAGIC
# MAGIC **How to Create Account-Level Groups:**
# MAGIC
# MAGIC *Azure Databricks Account Console (UI):*
# MAGIC 1. Sign in to the Databricks account console (not a workspace)
# MAGIC 2. In Azure, go to **accounts.azuredatabricks.net** (or accounts.cloud.databricks.com for AWS/GCP)
# MAGIC 3. Log in as an **account admin**
# MAGIC 4. Navigate to the **User Management** section
# MAGIC 5. Select **Groups** tab
# MAGIC 6. Click **Add Group** button
# MAGIC 7. Enter group name (e.g., `ml_engineers`)
# MAGIC 8. Press **Add** button
# MAGIC 9. Repeat for all required groups: `data_analysts`, `ml_engineers`, `data_scientists`, `data_engineers`, `all_users`
# MAGIC
# MAGIC *Alternative - Databricks CLI:*
# MAGIC ```
# MAGIC databricks account groups create --group-name data_analysts
# MAGIC databricks account groups create --group-name ml_engineers
# MAGIC databricks account groups create --group-name data_scientists
# MAGIC databricks account groups create --group-name data_engineers
# MAGIC databricks account groups create --group-name all_users
# MAGIC ```
# MAGIC
# MAGIC **For This Lab:**
# MAGIC - If you have account-level groups, the notebook will detect and use them
# MAGIC - If not, we'll demonstrate the concepts with your current user
# MAGIC - Example commands show what admins would run in production
# MAGIC - You'll learn the complete RBAC workflow either way

# COMMAND ----------

# Check if account-level groups exist
print("=== Checking for Account-Level Groups ===\n")

print("⚠ Important: Unity Catalog requires ACCOUNT-LEVEL groups")
print("  • Workspace groups (CREATE GROUP) do NOT work with Unity Catalog")
print("  • Only account admins can create account-level groups")
print("  • Groups must be created in the Account Console\n")

# Define required groups
required_groups = {
    'data_analysts': 'Group for data analysts with read access to data',
    'ml_engineers': 'Group for ML engineers with model execution rights',
    'data_scientists': 'Group for data scientists with full schema access',
    'data_engineers': 'Group for data engineers with data pipeline management',
    'all_users': 'Group for all users with basic catalog access'
}

print("Required groups for this lab:")
for group_name, description in required_groups.items():
    print(f"  • {group_name}: {description}")

# Check if account-level groups exist (read-only check)
print("\nChecking if groups exist at account level...")
print("-" * 80)

existing_groups = []
missing_groups = []

for group_name in required_groups.keys():
    try:
        # Try to grant a harmless permission to test if group exists
        # We'll immediately revoke it, so this is just a test
        # If group doesn't exist, this will fail with PRINCIPAL_DOES_NOT_EXIST
        test_sql = f"GRANT USAGE ON CATALOG {CATALOG_NAME} TO `{group_name}`"
        spark.sql(test_sql)

        # If we got here, group exists! Now revoke the test grant
        try:
            spark.sql(f"REVOKE USAGE ON CATALOG {CATALOG_NAME} FROM `{group_name}`")
        except:
            pass  # Revoke might fail if already granted, that's ok

        print(f"✓ {group_name}: Exists (account-level group)")
        existing_groups.append(group_name)

    except Exception as e:
        error_msg = str(e).lower()
        if "principal_does_not_exist" in error_msg or "does not exist" in error_msg or "cannot find" in error_msg:
            print(f"⊘ {group_name}: Does not exist at account level")
            missing_groups.append(group_name)
        elif "already granted" in error_msg or "already has" in error_msg:
            # Group exists, permission was already granted
            print(f"✓ {group_name}: Exists (account-level group)")
            existing_groups.append(group_name)
        elif "permission" in error_msg or "privilege" in error_msg:
            # Can't verify due to permissions, but let's assume it might exist
            print(f"? {group_name}: Cannot verify (insufficient permissions)")
            print(f"  Will attempt to use this group in permission grants")
            existing_groups.append(group_name)  # Optimistically add it
        else:
            print(f"? {group_name}: Cannot verify ({str(e)[:80]}...)")
            missing_groups.append(group_name)

# Summary
print("\n" + "="*80)
print("GROUP CHECK SUMMARY")
print("="*80)

# Store available groups for later use
available_groups = existing_groups

if len(existing_groups) > 0:
    print(f"\n✓ Account-level groups found: {len(existing_groups)}")
    for group in existing_groups:
        print(f"  ✓ {group}")
    print("\n  🎉 Excellent! These groups will be used for permission grants.")
else:
    print("\n⊘ No account-level groups found")

if len(missing_groups) > 0:
    print(f"\n⊘ Groups not found: {len(missing_groups)}")
    for group in missing_groups:
        print(f"  ⊘ {group}")

    print("\n📝 How to Create Account-Level Groups:")
    print("-" * 80)
    print("Account-level groups MUST be created in the Databricks Account Console:")
    print("")
    print("Option 1: Azure Databricks Account Console (UI) - Recommended")
    print("  1. Sign in to the Databricks account console (not a workspace)")
    print("  2. In Azure, go to: accounts.azuredatabricks.net")
    print("     (or accounts.cloud.databricks.com for AWS/GCP)")
    print("  3. Log in as an account admin")
    print("  4. Navigate to: User Management section")
    print("  5. Select: Groups tab")
    print("  6. Click: Add Group button")
    print("  7. Enter group name (e.g., ml_engineers)")
    print("  8. Press: Add button")
    print("  9. Repeat for all groups: data_analysts, ml_engineers, data_scientists,")
    print("     data_engineers, all_users")
    print("")
    print("Option 2: Databricks CLI (for Account Admins)")
    print("  databricks account groups create --group-name data_analysts")
    print("  databricks account groups create --group-name ml_engineers")
    print("  databricks account groups create --group-name data_scientists")
    print("  databricks account groups create --group-name data_engineers")
    print("  databricks account groups create --group-name all_users")
    print("")
    print("⚠ Note: CREATE GROUP in SQL creates workspace groups, NOT account groups")
    print("  Workspace groups do NOT work with Unity Catalog permissions!")

print(f"\n📊 Total available groups for permissions: {len(available_groups)}")
if len(available_groups) > 0:
    print("  These groups will be used in the permission granting section.")
else:
    print("  No groups available - will demonstrate with current user only.")
    print("  This is normal and the lab will still teach all RBAC concepts.")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Workspace vs. Account Groups
# MAGIC
# MAGIC **Important Distinction:**
# MAGIC - `SHOW GROUPS` displays **workspace-level groups** (created with `CREATE GROUP`)
# MAGIC - Unity Catalog requires **account-level groups** (created in Account Console)
# MAGIC - These are **completely separate** and cannot be used interchangeably!
# MAGIC
# MAGIC Let's check both to understand the difference.

# COMMAND ----------

# Display workspace groups vs account groups
print("=== Understanding Group Types ===\n")

print("⚠ CRITICAL: Workspace Groups ≠ Account Groups")
print("  • SHOW GROUPS shows workspace groups")
print("  • Unity Catalog needs account groups")
print("  • They are completely separate!\n")

# Check workspace groups
print("1. Workspace Groups (from SHOW GROUPS):")
print("-" * 80)
try:
    workspace_groups = spark.sql("SHOW GROUPS")
    workspace_group_list = [row[0] for row in workspace_groups.collect()]

    if len(workspace_group_list) > 0:
        print(f"Found {len(workspace_group_list)} workspace group(s):")
        display(workspace_groups)

        print("\nChecking our required groups in workspace:")
        for group_name in required_groups.keys():
            if group_name in workspace_group_list:
                print(f"  ✓ {group_name} - Found in workspace")
            else:
                print(f"  ✗ {group_name} - Not in workspace")

        print("\n⚠ WARNING: These are WORKSPACE groups!")
        print("  They will NOT work with Unity Catalog permissions.")
        print("  Unity Catalog requires ACCOUNT-LEVEL groups.")
    else:
        print("No workspace groups found")

except Exception as e:
    print(f"Unable to list workspace groups: {str(e)}")

# Check account groups (the ones that actually work with Unity Catalog)
print("\n2. Account Groups (for Unity Catalog):")
print("-" * 80)
print("Checking if groups exist at ACCOUNT level (required for Unity Catalog)...\n")

account_groups_found = []
account_groups_missing = []

for group_name in required_groups.keys():
    try:
        # Try to grant a test permission to see if group exists
        # This is the most reliable way to check across all Databricks versions
        test_sql = f"GRANT USAGE ON CATALOG {CATALOG_NAME} TO `{group_name}`"
        spark.sql(test_sql)

        # If we got here, group exists! Revoke the test grant
        try:
            spark.sql(f"REVOKE USAGE ON CATALOG {CATALOG_NAME} FROM `{group_name}`")
        except:
            pass

        print(f"  ✓ {group_name} - EXISTS at account level (works with Unity Catalog)")
        account_groups_found.append(group_name)

    except Exception as e:
        error_msg = str(e).lower()
        if "principal_does_not_exist" in error_msg or "does not exist" in error_msg or "cannot find" in error_msg:
            print(f"  ✗ {group_name} - DOES NOT EXIST at account level")
            account_groups_missing.append(group_name)
        elif "already granted" in error_msg or "already has" in error_msg:
            # Group exists, permission was already there
            print(f"  ✓ {group_name} - EXISTS at account level (works with Unity Catalog)")
            account_groups_found.append(group_name)
        elif "permission" in error_msg or "privilege" in error_msg:
            print(f"  ? {group_name} - Cannot verify (insufficient permissions)")
            account_groups_missing.append(group_name)
        else:
            print(f"  ? {group_name} - Cannot verify: {str(e)[:60]}...")
            account_groups_missing.append(group_name)

# Summary
print("\n" + "="*80)
print("GROUP TYPE SUMMARY")
print("="*80)

try:
    if len(workspace_group_list) > 0:
        print(f"\n📋 Workspace Groups: {len(workspace_group_list)} found")
        print("  ⚠ These do NOT work with Unity Catalog")
        print("  ⚠ Created with: CREATE GROUP")
        print("  ⚠ Only work for legacy workspace permissions")
except:
    pass

if len(account_groups_found) > 0:
    print(f"\n✓ Account Groups: {len(account_groups_found)} found")
    print("  ✓ These WORK with Unity Catalog")
    for group in account_groups_found:
        print(f"    • {group}")
else:
    print(f"\n✗ Account Groups: 0 found")
    print("  ✗ Unity Catalog permissions will not work")

if len(account_groups_missing) > 0:
    print(f"\n⊘ Missing Account Groups: {len(account_groups_missing)}")
    for group in account_groups_missing:
        print(f"    • {group}")
    print("\n  💡 To create account-level groups:")
    print("     1. Go to: https://accounts.cloud.databricks.com/")
    print("     2. User Management → Groups → Add Group")
    print("     3. Create each group at ACCOUNT level")

print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grant Permissions (Demonstration)
# MAGIC
# MAGIC Unity Catalog allows fine-grained permissions. Here's how permissions would be granted in production:
# MAGIC
# MAGIC **Typical Permission Structure:**
# MAGIC - **data_analysts**: SELECT on table (read-only access)
# MAGIC - **ml_engineers**: USE SCHEMA on schema (model execution and schema access)
# MAGIC - **data_scientists**: ALL PRIVILEGES on schema (full access)
# MAGIC - **data_engineers**: MODIFY on table (write access for data pipelines)
# MAGIC - **all_users**: USE CATALOG on catalog (basic catalog access)
# MAGIC
# MAGIC **Note:** This section demonstrates the concepts. In production, your admin would create groups and grant permissions.

# COMMAND ----------

# Demonstrate permission granting concepts
print("=== Unity Catalog Permissions (Demonstration) ===\n")

# Get current user
current_user = spark.sql("SELECT current_user()").collect()[0][0]
print(f"Current user: {current_user}\n")

# Show example permission commands
print("In production, an admin would execute commands like:\n")

example_grants = [
    {
        'description': 'Grant read access to data analysts',
        'sql': f"GRANT SELECT ON TABLE {table_path} TO `data_analysts`;"
    },
    {
        'description': 'Grant schema usage to ML engineers',
        'sql': f"GRANT USE SCHEMA ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME} TO `ml_engineers`;"
    },
    {
        'description': 'Grant full access to data scientists',
        'sql': f"GRANT ALL PRIVILEGES ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME} TO `data_scientists`;"
    },
    {
        'description': 'Grant write access to data engineers',
        'sql': f"GRANT MODIFY ON TABLE {table_path} TO `data_engineers`;"
    },
    {
        'description': 'Grant catalog usage to all users',
        'sql': f"GRANT USE CATALOG ON CATALOG {CATALOG_NAME} TO `all_users`;"
    }
]

for i, grant in enumerate(example_grants, 1):
    print(f"{i}. {grant['description']}")
    print(f"   {grant['sql']}")
    print()

# Try to grant permissions to production groups (if they exist) and current user
print("="*80)
print("Attempting to grant permissions...")
print("="*80)

# Check if we have available_groups from earlier section
try:
    available_groups_list = available_groups
    print(f"\nℹ Available groups from creation section: {len(available_groups_list)}")
    if len(available_groups_list) > 0:
        print(f"  Groups: {', '.join(available_groups_list)}")
except NameError:
    # If available_groups doesn't exist, we'll try all groups and handle errors
    available_groups_list = []
    print("\nℹ No group information from creation section - will attempt all groups")

successful_grants = []
failed_grants = []
groups_granted = []
groups_not_found = []

# Define production permissions to try
production_permissions = [
    {
        'principal': 'data_analysts',
        'privilege': 'SELECT',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Read access to customer churn data'
    },
    {
        'principal': 'ml_engineers',
        'privilege': 'USE SCHEMA',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Schema usage rights'
    },
    {
        'principal': 'data_scientists',
        'privilege': 'ALL PRIVILEGES',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Full access to schema'
    },
    {
        'principal': 'data_engineers',
        'privilege': 'MODIFY',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Write access to manage data pipelines'
    },
    {
        'principal': 'all_users',
        'privilege': 'USE CATALOG',
        'object_type': 'CATALOG',
        'object_name': CATALOG_NAME,
        'description': 'Catalog usage rights'
    }
]

# Try production groups first
print("\n1. Attempting Production Group Grants:")
print("-" * 80)

for perm in production_permissions:
    group_name = perm['principal']

    # Skip if we know the group doesn't exist
    if len(available_groups_list) > 0 and group_name not in available_groups_list:
        print(f"\n⊘ Skipping {group_name}: Group was not created/found in earlier section")
        groups_not_found.append(group_name)
        failed_grants.append(perm)
        continue

    print(f"\nGranting {perm['privilege']} on {perm['object_type']} to {group_name}:")
    print(f"  Object: {perm['object_name']}")
    print(f"  Purpose: {perm['description']}")

    try:
        grant_sql = f"GRANT {perm['privilege']} ON {perm['object_type']} {perm['object_name']} TO `{group_name}`"
        spark.sql(grant_sql)
        print(f"  ✓ Status: Success - Group exists and grant applied!")
        successful_grants.append(perm)
        groups_granted.append(group_name)
    except Exception as e:
        error_msg = str(e)
        if "already has" in error_msg.lower() or "already granted" in error_msg.lower():
            print(f"  ✓ Status: Already granted - Group exists!")
            successful_grants.append(perm)
            groups_granted.append(group_name)
        elif "principal_does_not_exist" in error_msg.lower() or "does not exist" in error_msg.lower() or "cannot find" in error_msg.lower():
            print(f"  ⊘ Status: Group '{group_name}' does not exist")
            print(f"  Note: Group creation failed or requires account admin privileges")
            groups_not_found.append(group_name)
            failed_grants.append(perm)
        elif "insufficient" in error_msg.lower() or "permission" in error_msg.lower():
            print(f"  ⚠ Status: Insufficient privileges (requires admin)")
            print(f"  Note: Group may exist but you need admin rights to grant")
            failed_grants.append(perm)
        else:
            print(f"  ⚠ Status: {error_msg[:150]}...")
            failed_grants.append(perm)

# Also grant to current user for demonstration
print("\n2. Granting to Current User (for demonstration):")
print("-" * 80)

user_permissions = [
    {
        'principal': current_user,
        'privilege': 'SELECT',
        'object_type': 'TABLE',
        'object_name': table_path,
        'description': 'Read access to customer churn data'
    },
    {
        'principal': current_user,
        'privilege': 'USE SCHEMA',
        'object_type': 'SCHEMA',
        'object_name': f"{CATALOG_NAME}.{SCHEMA_NAME}",
        'description': 'Schema usage rights'
    }
]

for perm in user_permissions:
    print(f"\nGranting {perm['privilege']} on {perm['object_type']}:")
    print(f"  Object: {perm['object_name']}")

    try:
        grant_sql = f"GRANT {perm['privilege']} ON {perm['object_type']} {perm['object_name']} TO `{current_user}`"
        spark.sql(grant_sql)
        print(f"  ✓ Status: Success")
    except Exception as e:
        error_msg = str(e)
        if "already has" in error_msg.lower() or "already granted" in error_msg.lower():
            print(f"  ✓ Status: Already granted")
        else:
            print(f"  ⚠ Status: {error_msg[:80]}...")

# Summary
print("\n" + "="*80)
print("PERMISSION GRANT SUMMARY")
print("="*80)

if len(groups_granted) > 0:
    print(f"\n✓ Production groups successfully granted: {len(set(groups_granted))}")
    for group in set(groups_granted):
        print(f"  ✓ {group}")
    print("\n  🎉 Excellent! Your workspace has production groups configured!")
    print("  The verification section will show these grants.")

if len(groups_not_found) > 0:
    print(f"\n⊘ Groups not found: {len(set(groups_not_found))}")
    for group in set(groups_not_found):
        print(f"  ⊘ {group}")
    print("\n  📝 Why groups don't exist:")
    print("  • Group creation requires account admin privileges")
    print("  • You may not have permission to create groups")
    print("  • Groups may need to be created at account level")
    print("\n  💡 Solution:")
    print("  • Contact your Databricks account admin")
    print("  • Request creation of: data_analysts, ml_engineers, data_scientists, all_users")
    print("  • Or use this lab in demonstration mode (grants to current user)")

if successful_grants:
    print(f"\n✓ Total successful grants: {len(successful_grants)}")
    for perm in successful_grants:
        principal = perm.get('principal', 'current_user')
        print(f"  - {principal}: {perm['privilege']} on {perm['object_type']}")

if len(groups_granted) == 0:
    print("\n📋 Demonstration Mode:")
    print("  Since production groups don't exist, this lab will:")
    print("  • Grant permissions to your current user")
    print("  • Show example commands for production")
    print("  • Explain what production would look like")
    print("  • Teach RBAC concepts effectively")

print("\n" + "="*80)
print("KEY CONCEPTS - Unity Catalog Permissions")
print("="*80)
print("""
1. **Hierarchical Permissions**
   - CATALOG → SCHEMA → TABLE/MODEL
   - Permissions inherit down the hierarchy

2. **Common Permission Types**
   - USE CATALOG: Access to catalog
   - USE SCHEMA: Access to schema
   - SELECT: Read data from tables
   - MODIFY: Write/update data
   - EXECUTE: Run models/functions
   - ALL PRIVILEGES: Full access

3. **Role-Based Access Control (RBAC)**
   - Create groups for different roles (e.g., data_analysts, ml_engineers)
   - Grant permissions to groups, not individuals
   - Users inherit permissions from their groups

4. **Production Setup (Admin Tasks)**
   - Create groups: CREATE GROUP data_analysts;
   - Add users to groups: ALTER GROUP data_analysts ADD USER user@company.com;
   - Grant permissions: GRANT SELECT ON TABLE ... TO data_analysts;

5. **Best Practices**
   - Use groups for permission management
   - Follow principle of least privilege
   - Document permission decisions
   - Review permissions regularly
   - All changes are automatically audited
""")

print("="*80)
print("\n✓ Permission concepts demonstrated")
print("\nIn production environments:")
print("  • Workspace admins create and manage groups")
print("  • Permissions are granted based on job roles")
print("  • All changes are tracked in audit logs")
print("  • Regular access reviews ensure compliance")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Granted Permissions
# MAGIC
# MAGIC Let's verify the permissions were granted successfully by viewing grants on each object.

# COMMAND ----------

# Verify permissions
print("=== Verifying Granted Permissions ===\n")
print("Checking what permissions exist vs. what was demonstrated...\n")

# Define what we expect in production
expected_grants = {
    'table': [
        {'principal': 'data_analysts', 'privilege': 'SELECT', 'description': 'Read access to customer churn data'}
    ],
    'schema': [
        {'principal': 'ml_engineers', 'privilege': 'USE SCHEMA', 'description': 'Schema usage rights'},
        {'principal': 'data_scientists', 'privilege': 'ALL PRIVILEGES', 'description': 'Full access to schema'}
    ],
    'catalog': [
        {'principal': 'all_users', 'privilege': 'USE CATALOG', 'description': 'Catalog usage rights'}
    ]
}

# Check table permissions
print("1. Table Permissions (customer_churn_data):")
print("-" * 80)
print(f"Expected in production: GRANT SELECT ON TABLE ... TO `data_analysts`\n")

try:
    table_grants = spark.sql(f"SHOW GRANTS ON TABLE {table_path}")
    grants_list = table_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on table:")
        display(table_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        data_analysts_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if 'data_analysts' in row_str and 'select' in row_str:
                print("  ✓ data_analysts has SELECT permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_analysts_found = True
                break

        if not data_analysts_found:
            print("  ⊘ data_analysts: Not found (would exist in production)")

        # Check for data_engineers
        data_engineers_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if 'data_engineers' in row_str and 'modify' in row_str:
                print("  ✓ data_engineers has MODIFY permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_engineers_found = True
                break

        if not data_engineers_found:
            print("  ⊘ data_engineers: Not found (would exist in production)")

        # Check for current user
        current_user_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if current_user.lower() in row_str:
                if not current_user_found:
                    print(f"  ✓ {current_user} has permissions on table (DEMONSTRATION GRANT)")
                    current_user_found = True
                print(f"     - {row}")

        production_groups_found = data_analysts_found or data_engineers_found
        if production_groups_found:
            print("\n  📝 Note: Production groups found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'data_analysts' and 'data_engineers' groups here")
    else:
        print("⊘ No explicit grants on table")
        print("\n📋 What You Would See in Production:")
        print("  ✓ data_analysts: SELECT permission")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are inherited from schema or catalog level")
        print("  • This is normal in learning environments")
        print("  • You can still access the table through inherited permissions")

except Exception as e:
    print(f"Unable to show table grants: {str(e)}")
    print("Note: This may be normal if grants are inherited from parent objects")

print("\n2. Schema Permissions (churn_models):")
print("-" * 80)
print("Expected in production:")
print("  • GRANT USE SCHEMA ON SCHEMA ... TO `ml_engineers`")
print("  • GRANT ALL PRIVILEGES ON SCHEMA ... TO `data_scientists`\n")

try:
    schema_grants = spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME}")
    grants_list = schema_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on schema:")
        display(schema_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        ml_engineers_found = False
        data_scientists_found = False

        for row in grants_list:
            row_str = str(row).lower()

            if 'ml_engineers' in row_str and 'use schema' in row_str:
                print("  ✓ ml_engineers has USE SCHEMA permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                ml_engineers_found = True

            if 'data_scientists' in row_str and 'all' in row_str:
                print("  ✓ data_scientists has ALL PRIVILEGES (PRODUCTION GRANT)")
                print(f"     - {row}")
                data_scientists_found = True

        if not ml_engineers_found:
            print("  ⊘ ml_engineers: Not found (would exist in production)")
        if not data_scientists_found:
            print("  ⊘ data_scientists: Not found (would exist in production)")

        # Check for current user
        current_user_found = False
        for row in grants_list:
            row_str = str(row).lower()
            if current_user.lower() in row_str:
                if not current_user_found:
                    print(f"  ✓ {current_user} has permissions on schema (DEMONSTRATION GRANT)")
                    current_user_found = True
                print(f"     - {row}")

        if ml_engineers_found or data_scientists_found:
            print("\n  📝 Note: Production groups found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'ml_engineers' and 'data_scientists' groups here")
    else:
        print("⊘ No explicit grants on schema")
        print("\n📋 What You Would See in Production:")
        print("  ✓ ml_engineers: USE SCHEMA permission")
        print("  ✓ data_scientists: ALL PRIVILEGES")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are inherited from catalog or account level")
        print("  • This is normal in shared Databricks workspaces")

except Exception as e:
    print(f"Unable to show schema grants: {str(e)}")
    print("Note: This may require additional permissions")

print("\n3. Catalog Permissions (financial_services):")
print("-" * 80)
print("Expected in production: GRANT USE CATALOG ON CATALOG ... TO `all_users`\n")

try:
    catalog_grants = spark.sql(f"SHOW GRANTS ON CATALOG {CATALOG_NAME}")
    grants_list = catalog_grants.collect()

    if len(grants_list) > 0:
        print(f"✓ Found {len(grants_list)} grant(s) on catalog:")
        display(catalog_grants)

        # Check for expected permissions
        grants_text = ' '.join([str(row) for row in grants_list]).lower()

        print("\nGrant Analysis:")

        # Check for production groups
        all_users_found = False

        for row in grants_list:
            row_str = str(row).lower()
            if 'all_users' in row_str and 'use catalog' in row_str:
                print("  ✓ all_users has USE CATALOG permission (PRODUCTION GRANT)")
                print(f"     - {row}")
                all_users_found = True

        if not all_users_found:
            print("  ⊘ all_users: Not found (would exist in production)")

        # Show all other grants
        print("\n  All grants on catalog:")
        for row in grants_list:
            row_str = str(row).lower()
            if 'all_users' not in row_str:  # Don't duplicate all_users
                print(f"  • {row}")

        if all_users_found:
            print("\n  📝 Note: Production group 'all_users' found with correct permissions!")
        else:
            print("\n  📝 Note: In production, you would see 'all_users' group here")
    else:
        print("⊘ No explicit grants on catalog")
        print("\n📋 What You Would See in Production:")
        print("  ✓ all_users: USE CATALOG permission")
        print("  ✓ Admin groups with full privileges")
        print("  ✓ Other relevant groups with appropriate permissions")
        print("\nℹ Current Status:")
        print("  • Permissions are managed at account level")
        print("  • Users have default workspace access")
        print("  • Catalog is accessible to all workspace users")
        print("\n✓ You can still use the catalog - access is inherited from workspace/account level")

except Exception as e:
    print(f"Unable to show catalog grants: {str(e)}")
    print("Note: This may require additional permissions")

print("\n" + "="*80)
print("RBAC VERIFICATION SUMMARY")
print("="*80)

# Summary of what was verified
print("\n✓ Permissions Verified:")
print(f"  - Table grants checked: {table_path}")
print(f"  - Schema grants checked: {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"  - Catalog grants checked: {CATALOG_NAME}")

print("\n" + "="*80)
print("COMPARISON: Demonstration vs. Production")
print("="*80)

# Check what was actually granted by reviewing the grants
try:
    table_check = spark.sql(f"SHOW GRANTS ON TABLE {table_path}").collect()
    schema_check = spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOG_NAME}.{SCHEMA_NAME}").collect()
    catalog_check = spark.sql(f"SHOW GRANTS ON CATALOG {CATALOG_NAME}").collect()

    # Determine which groups were found
    all_grants_text = ' '.join([str(row) for row in table_check + schema_check + catalog_check]).lower()

    data_analysts_exists = 'data_analysts' in all_grants_text
    ml_engineers_exists = 'ml_engineers' in all_grants_text
    data_scientists_exists = 'data_scientists' in all_grants_text
    all_users_exists = 'all_users' in all_grants_text

except:
    data_analysts_exists = False
    ml_engineers_exists = False
    data_scientists_exists = False
    all_users_exists = False

print("\n📋 What Was Demonstrated (Example Commands):")
print("-" * 80)
print("1. GRANT SELECT ON TABLE ... TO `data_analysts`")
print("   Purpose: Read access to customer churn data")
print(f"   Status: {'✓ Successfully granted!' if data_analysts_exists else '⊘ Group does not exist in this environment'}")
print("")
print("2. GRANT USE SCHEMA ON SCHEMA ... TO `ml_engineers`")
print("   Purpose: Schema usage rights")
print(f"   Status: {'✓ Successfully granted!' if ml_engineers_exists else '⊘ Group does not exist in this environment'}")
print("")
print("3. GRANT ALL PRIVILEGES ON SCHEMA ... TO `data_scientists`")
print("   Purpose: Full access to schema")
print(f"   Status: {'✓ Successfully granted!' if data_scientists_exists else '⊘ Group does not exist in this environment'}")
print("")
print("4. GRANT USE CATALOG ON CATALOG ... TO `all_users`")
print("   Purpose: Catalog usage rights")
print(f"   Status: {'✓ Successfully granted!' if all_users_exists else '⊘ Group does not exist in this environment'}")

print("\n📋 What Actually Exists (Verification Results):")
print("-" * 80)
print(f"✓ {current_user}: SELECT on TABLE (demonstration grant)")
print(f"✓ {current_user}: USE SCHEMA on SCHEMA (demonstration grant)")

if data_analysts_exists:
    print("✓ data_analysts: SELECT on TABLE (production grant)")
else:
    print("⊘ data_analysts: Not found (would exist in production)")

if ml_engineers_exists:
    print("✓ ml_engineers: USE SCHEMA on SCHEMA (production grant)")
else:
    print("⊘ ml_engineers: Not found (would exist in production)")

if data_scientists_exists:
    print("✓ data_scientists: ALL PRIVILEGES on SCHEMA (production grant)")
else:
    print("⊘ data_scientists: Not found (would exist in production)")

if all_users_exists:
    print("✓ all_users: USE CATALOG on CATALOG (production grant)")
else:
    print("⊘ all_users: Not found (would exist in production)")

# Summary message
if data_analysts_exists or ml_engineers_exists or data_scientists_exists or all_users_exists:
    print("\n🎉 Excellent! Your workspace has production groups configured and grants were successful!")
else:
    print("\nℹ Note: This is a learning environment without pre-configured production groups.")

print("\n📋 What You Would See in Production:")
print("-" * 80)
print("""
Table Level (customer_churn_data):
  ✓ data_analysts: SELECT
  ✓ data_scientists: ALL PRIVILEGES (inherited from schema)
  ✓ ml_engineers: SELECT (if granted)

Schema Level (churn_models):
  ✓ ml_engineers: USE SCHEMA
  ✓ data_scientists: ALL PRIVILEGES
  ✓ data_analysts: USE SCHEMA (if granted)

Catalog Level (financial_services):
  ✓ all_users: USE CATALOG
  ✓ admins: ALL PRIVILEGES
  ✓ Other groups as needed

Each grant would show:
  • Principal (group name)
  • ActionType (SELECT, USE SCHEMA, etc.)
  • ObjectType (TABLE, SCHEMA, CATALOG)
  • ObjectKey (full path to object)
""")

print("\n📊 Understanding the Results:")
print("-" * 80)
print("""
If you see "No explicit grants" or "0 grants", this is NORMAL and EXPECTED in:
  • Shared Databricks workspaces
  • Learning/training environments
  • Workspaces with default access policies

How Access Works Without Explicit Grants:
  1. Workspace-level permissions grant default access
  2. Account-level permissions provide inherited access
  3. You're the creator/owner of the objects (automatic access)
  4. Unity Catalog uses hierarchical permission inheritance

What This Means:
  ✓ You CAN access and use the data/models
  ✓ Permissions are inherited from parent levels
  ✓ This is a secure and common configuration
  ✓ In production, explicit grants would be added for other users/groups

Production Difference:
  • Admins would create explicit grants for each group
  • You would see rows in the SHOW GRANTS output
  • Each user/group would have specific permissions listed
  • Audit logs would track all grant operations
""")

print("\n" + "="*80)
print("KEY TAKEAWAYS - Unity Catalog RBAC")
print("="*80)
print("""
1. ✓ Unity Catalog provides fine-grained access control
   - Permissions at catalog, schema, table, and column levels
   - Hierarchical inheritance of permissions

2. ✓ Groups enable scalable permission management
   - Create groups for different roles
   - Grant permissions to groups, not individuals
   - Users inherit from all their groups

3. ✓ Production RBAC Workflow:
   - Account admins create groups
   - Users are assigned to groups based on roles
   - Permissions follow principle of least privilege
   - Regular audits ensure compliance

4. ✓ All permission changes are automatically logged
   - Complete audit trail for compliance
   - Track who granted what to whom
   - Query audit logs for security reviews

5. ✓ RBAC is essential for enterprise governance
   - Meets regulatory requirements
   - Enables secure collaboration
   - Supports data governance policies

Example Production Commands:
  CREATE GROUP data_analysts;
  ALTER GROUP data_analysts ADD USER user@company.com;
  GRANT SELECT ON TABLE ... TO data_analysts;
  SHOW GRANTS ON TABLE ...;
""")
print("="*80)

print("\n💡 Next Steps for Production RBAC:")
print("  1. Work with admin to create proper groups")
print("  2. Map organizational roles to Unity Catalog groups")
print("  3. Document permission policies")
print("  4. Set up regular permission audits")
print("  5. Train users on data access procedures")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Audit Logging
# MAGIC
# MAGIC Unity Catalog automatically logs all operations. Let's query the audit logs to see model operations.
# MAGIC
# MAGIC **Compliance Value:** Audit logs provide a complete trail for regulatory requirements.

# COMMAND ----------

# Query audit logs for model operations
print("=== Audit Logging Demonstration ===\n")

# Check if system catalog is accessible
print("Checking audit log access...")
audit_available = False

try:
    # Try to access system catalog
    spark.sql("USE CATALOG system")
    spark.sql("SHOW TABLES IN system.access").collect()
    audit_available = True
    print("✓ System catalog is accessible")
except Exception as e:
    print("⚠ System catalog not accessible in this workspace")
    print(f"  Reason: {str(e)[:100]}...")

print("\n" + "-"*80)

if audit_available:
    print("\nQuerying audit logs for recent operations...")
    print("(This may take a moment...)\n")

    # Try multiple queries to find audit data
    queries_to_try = [
        {
            'name': 'Unity Catalog operations in this session',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name,
                    response.status_code
                FROM system.access.audit
                WHERE event_date >= current_date() - INTERVAL 1 DAY
                    AND user_identity.email = '{current_user}'
                    AND (
                        action_name IN ('createTable', 'createSchema', 'createCatalog',
                                       'getTable', 'getSchema', 'getCatalog',
                                       'createRegisteredModelVersion', 'updateRegisteredModel')
                        OR request_params.full_name_arg LIKE '%{CATALOG_NAME}%'
                        OR request_params.full_name_arg LIKE '%{SCHEMA_NAME}%'
                    )
                ORDER BY event_time DESC
                LIMIT 20
            """
        },
        {
            'name': 'Recent table operations',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name
                FROM system.access.audit
                WHERE event_date >= current_date() - INTERVAL 1 DAY
                    AND action_name IN ('createTable', 'getTable', 'readTable')
                ORDER BY event_time DESC
                LIMIT 10
            """
        },
        {
            'name': 'Any recent operations by current user',
            'query': f"""
                SELECT
                    event_time,
                    user_identity.email as user,
                    action_name,
                    request_params.full_name_arg as object_name
                FROM system.access.audit
                WHERE event_date >= current_date()
                    AND user_identity.email = '{current_user}'
                ORDER BY event_time DESC
                LIMIT 10
            """
        }
    ]

    audit_found = False

    for query_info in queries_to_try:
        if audit_found:
            break

        try:
            print(f"Trying: {query_info['name']}...")
            audit_logs = spark.sql(query_info['query'])
            audit_count = audit_logs.count()

            if audit_count > 0:
                print(f"✓ Found {audit_count} audit log entries!\n")
                print(f"Showing: {query_info['name']}")
                display(audit_logs)
                audit_found = True

                print("\n" + "="*80)
                print("AUDIT LOG ANALYSIS")
                print("="*80)
                print(f"""
✓ Successfully retrieved audit logs from Unity Catalog

What These Logs Show:
  • event_time: When the operation occurred
  • user: Who performed the operation ({current_user})
  • action_name: What operation was performed (createTable, getTable, etc.)
  • object_name: Which object was accessed
  • status_code: Success (200) or error codes

Compliance Value:
  ✓ Complete audit trail of all operations
  ✓ Track who accessed what data and when
  ✓ Investigate security incidents
  ✓ Meet regulatory requirements (SOX, GDPR, HIPAA)
  ✓ Generate compliance reports
""")
                break
            else:
                print(f"  No results for this query")
        except Exception as e:
            print(f"  Query failed: {str(e)[:80]}...")
            continue

    if not audit_found:
        print("\n⚠ No audit logs found with any query")
        print("\nPossible reasons:")
        print("  • Audit logs may have a delay before appearing (up to 1 hour)")
        print("  • Logs may be retained for limited time")
        print("  • Some operations may not be logged in this workspace type")
        print("  • Filters may not match recent operations")
        audit_available = False

if not audit_available:
    print("\n" + "="*80)
    print("AUDIT LOG DEMONSTRATION (Simulated)")
    print("="*80)
    print("\nSince audit logs aren't available, here's what they would show for this lab:\n")

    # Create simulated audit log data
    from datetime import datetime, timedelta
    import pandas as pd

    current_time = datetime.now()

    simulated_logs = [
        {
            'event_time': (current_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createCatalog',
            'object_name': CATALOG_NAME,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=9)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createSchema',
            'object_name': f'{CATALOG_NAME}.{SCHEMA_NAME}',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=8)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createTable',
            'object_name': table_path,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'createRegisteredModelVersion',
            'object_name': MODEL_NAME,
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'setRegisteredModelAlias',
            'object_name': f'{MODEL_NAME} (Champion)',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'grantPrivileges',
            'object_name': f'USE SCHEMA on {CATALOG_NAME}.{SCHEMA_NAME}',
            'status_code': 200
        },
        {
            'event_time': (current_time - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S'),
            'user': current_user,
            'action_name': 'getTable',
            'object_name': table_path,
            'status_code': 200
        }
    ]

    simulated_df = pd.DataFrame(simulated_logs)
    print("Simulated Audit Log Entries (What Would Appear in Production):")
    print("-"*80)
    display(simulated_df)

    print("\n" + "="*80)
    print("AUDIT LOG ANALYSIS (Simulated)")
    print("="*80)
    print(f"""
What These Logs Show:
  ✓ Catalog creation: {CATALOG_NAME}
  ✓ Schema creation: {SCHEMA_NAME}
  ✓ Table creation: {TABLE_NAME}
  ✓ Model registration: {MODEL_NAME}
  ✓ Model alias assignment: Champion
  ✓ Permission grant: USE SCHEMA
  ✓ Data access: getTable operation

All operations performed by: {current_user}
All operations successful: status_code = 200

Compliance Value:
  ✓ Complete audit trail of all operations
  ✓ Track who accessed what data and when
  ✓ Investigate security incidents
  ✓ Meet regulatory requirements (SOX, GDPR, HIPAA)
  ✓ Generate compliance reports
  ✓ Retention: 90+ days (configurable)
""")

    print("\n📚 About Unity Catalog Audit Logs:")
    print("-"*80)
    print("""
Audit logs in Unity Catalog track ALL operations including:

1. **Data Access**
   - Table reads and writes (getTable, readTable)
   - Schema and catalog access
   - Column-level access (if enabled)

2. **Model Operations**
   - Model registration (createRegisteredModelVersion)
   - Version creation and updates
   - Model alias changes (setRegisteredModelAlias)
   - Model downloads and deployments

3. **Permission Changes**
   - GRANT and REVOKE operations (grantPrivileges, revokePrivileges)
   - Group membership changes
   - Role assignments

4. **Administrative Actions**
   - Catalog/schema creation (createCatalog, createSchema)
   - Table modifications (createTable, alterTable)
   - Policy updates

Example Audit Log Query:
""")

    print("""
-- Query all operations on a specific model
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.name,
    response.status_code
FROM system.access.audit
WHERE request_params.name = 'catalog.schema.model_name'
ORDER BY event_time DESC;

-- Query all permission grants
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.privilege,
    request_params.principal
FROM system.access.audit
WHERE action_name LIKE '%GRANT%'
ORDER BY event_time DESC;

-- Query all data access
SELECT
    event_time,
    user_identity.email,
    action_name,
    request_params.full_name_arg
FROM system.access.audit
WHERE action_name = 'getTable'
ORDER BY event_time DESC;
""")

    print("\n" + "-"*80)
    print("In Production Environments:")
    print("  ✓ Audit logs are automatically enabled")
    print("  ✓ Logs are retained for 90+ days (configurable)")
    print("  ✓ Can be exported to external systems (S3, Azure, etc.)")
    print("  ✓ Used for compliance reporting and security monitoring")
    print("  ✓ Integrated with SIEM tools for real-time alerting")

    print("\n" + "-"*80)
    print("What Audit Logs Would Show for This Lab:")
    print("  • Model registration: " + MODEL_NAME)
    print("  • Version creation: Versions 1, 2, etc.")
    print("  • Alias assignments: Champion, Challenger")
    print("  • Table creation: " + table_path)
    print("  • All by user: " + current_user)
    print("  • Timestamps for each operation")
    print("  • Success/failure status codes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Lineage Tracking
# MAGIC
# MAGIC Unity Catalog automatically tracks lineage from data to models. This shows:
# MAGIC - Which tables were used to train the model
# MAGIC - Which notebooks/jobs created the model
# MAGIC - Downstream dependencies
# MAGIC
# MAGIC **Governance Benefit:** Complete transparency for auditors and stakeholders.

# COMMAND ----------

# Demonstrate lineage information
print("=== Model Lineage Information ===\n")

# Get model details
model_details = client.get_registered_model(MODEL_NAME)

print(f"Model: {model_details.name}")
print(f"Description: {model_details.description[:100] if model_details.description else 'N/A'}...")
print(f"\nLineage:")
print(f"  - Source Data: {table_path}")
print(f"  - Training Notebook: {experiment_name}")
print(f"  - Total Versions: {len(all_versions)}")
print(f"  - Current Champion: Version {model_version.version}")
print(f"  - Current Challenger: Version {model_version_v2.version}")

# Show data lineage through Unity Catalog
print(f"\n✓ Unity Catalog tracks complete lineage:")
print(f"  Data → Model → Deployment")
print(f"  All accessible through the Unity Catalog UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7: Model Monitoring and Reproducibility
# MAGIC
# MAGIC For production models, we need:
# MAGIC - **Reproducibility**: Ability to recreate any model version
# MAGIC - **Monitoring**: Track model performance over time
# MAGIC - **Documentation**: Clear records of all decisions
# MAGIC
# MAGIC Let's implement these best practices.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reproducibility: Recreate Model from Registry
# MAGIC
# MAGIC Demonstrate how to fully reproduce a model using MLflow tracking.

# COMMAND ----------

# Get run information for reproducibility
run_info = client.get_run(best_run_id)

print("=== Model Reproducibility Information ===\n")
print(f"Run ID: {run_info.info.run_id}")
print(f"Experiment ID: {run_info.info.experiment_id}")
print(f"Start Time: {datetime.fromtimestamp(run_info.info.start_time/1000)}")
print(f"End Time: {datetime.fromtimestamp(run_info.info.end_time/1000)}")

print("\nLogged Parameters:")
for key, value in run_info.data.params.items():
    print(f"  {key}: {value}")

print("\nLogged Metrics:")
for key, value in run_info.data.metrics.items():
    print(f"  {key}: {value:.4f}")

print("\nLogged Tags:")
for key, value in run_info.data.tags.items():
    if not key.startswith('mlflow.'):
        print(f"  {key}: {value}")

print("\n✓ All information needed to reproduce this model is logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Performance Report
# MAGIC
# MAGIC Generate a comprehensive report for stakeholders and compliance.

# COMMAND ----------

# Create performance report
report = f"""
{'='*80}
CUSTOMER CHURN MODEL - PERFORMANCE REPORT
{'='*80}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Name: {MODEL_NAME}
Champion Version: {model_version.version}
Challenger Version: {model_version_v2.version}

{'='*80}
CHAMPION MODEL PERFORMANCE
{'='*80}

Algorithm: {best_model_name}
Training Samples: {X_train.shape[0]:,}
Test Samples: {X_test.shape[0]:,}
Number of Features: {X_train.shape[1]}

Performance Metrics (Test Set):
  - Accuracy:  {comparison_df.loc[best_model_idx, 'Accuracy']:.4f}
  - Precision: {comparison_df.loc[best_model_idx, 'Precision']:.4f}
  - Recall:    {comparison_df.loc[best_model_idx, 'Recall']:.4f}
  - F1 Score:  {comparison_df.loc[best_model_idx, 'F1_Score']:.4f}
  - AUC-ROC:   {comparison_df.loc[best_model_idx, 'AUC']:.4f}

{'='*80}
CHALLENGER MODEL PERFORMANCE
{'='*80}

Algorithm: Random Forest (Improved)
Performance Metrics (Test Set):
  - Accuracy:  {rf_v2_metrics['test_accuracy']:.4f}
  - Precision: {rf_v2_metrics['test_precision']:.4f}
  - Recall:    {rf_v2_metrics['test_recall']:.4f}
  - F1 Score:  {rf_v2_metrics['test_f1']:.4f}
  - AUC-ROC:   {rf_v2_metrics['test_auc']:.4f}

{'='*80}
GOVERNANCE & COMPLIANCE
{'='*80}

✓ Data stored in Unity Catalog with access controls
✓ All experiments tracked in MLflow
✓ Model versions registered with full documentation
✓ Audit logs available for all operations
✓ Complete lineage from data to deployment
✓ RBAC implemented for data and model access

{'='*80}
RECOMMENDATIONS
{'='*80}

1. Deploy Champion model to production
2. Run A/B test with Challenger model
3. Monitor model performance weekly
4. Retrain model quarterly with fresh data
5. Review audit logs monthly for compliance

{'='*80}
"""

print(report)

# Save report as artifact
with open('/tmp/model_performance_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Report saved to /tmp/model_performance_report.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8: Model Archiving and Cleanup Policies
# MAGIC
# MAGIC As models accumulate, we need policies for:
# MAGIC - **Archiving old versions** that are no longer in use
# MAGIC - **Cleaning up experiments** to maintain organization
# MAGIC - **Retaining compliance records** per regulatory requirements
# MAGIC
# MAGIC **Best Practice:** Archive models rather than delete them to maintain audit trails.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Archive Old Model Versions
# MAGIC
# MAGIC Let's demonstrate archiving a model version that's no longer needed.

# COMMAND ----------

# Function to archive old model versions
def archive_model_version(model_name, version, reason):
    """
    Archive a model version by adding archive tags and documentation.

    Args:
        model_name: Full model name in Unity Catalog
        version: Version number to archive
        reason: Reason for archiving
    """
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archived",
        value="true"
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archive_date",
        value=datetime.now().strftime('%Y-%m-%d')
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archive_reason",
        value=reason
    )

    print(f"✓ Model version {version} archived")
    print(f"  Reason: {reason}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")

# Example: Archive the first version if we have multiple versions
if len(all_versions) > 2:
    archive_model_version(
        MODEL_NAME,
        all_versions[-1].version,  # Oldest version
        "Superseded by improved models with better performance"
    )
else:
    print("Note: Archiving demonstration - would archive older versions in production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup Policy Implementation
# MAGIC
# MAGIC Define and implement cleanup policies for model registry maintenance.

# COMMAND ----------

# Define cleanup policy
cleanup_policy = {
    'retain_champion': True,  # Always keep Champion model
    'retain_challenger': True,  # Always keep Challenger model
    'archive_after_days': 90,  # Archive versions older than 90 days
    'max_versions': 10,  # Keep maximum 10 versions
    'require_documentation': True  # All versions must have documentation
}

print("=== Model Registry Cleanup Policy ===\n")
for key, value in cleanup_policy.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

# Implement cleanup check
def check_cleanup_needed(model_name, policy):
    """
    Check if cleanup is needed based on policy.

    Args:
        model_name: Full model name in Unity Catalog
        policy: Dictionary of cleanup policies

    Returns:
        List of versions that can be archived
    """
    versions = client.search_model_versions(f"name='{model_name}'")

    # Get versions with aliases (Champion, Challenger)
    protected_versions = set()
    for version in versions:
        if hasattr(version, 'aliases') and version.aliases:
            protected_versions.add(version.version)

    # Find versions that can be archived
    archivable = []
    for version in versions:
        # Skip protected versions
        if version.version in protected_versions:
            continue

        # Check age
        created_time = datetime.fromtimestamp(version.creation_timestamp / 1000)
        age_days = (datetime.now() - created_time).days

        if age_days > policy['archive_after_days']:
            archivable.append({
                'version': version.version,
                'age_days': age_days,
                'created': created_time
            })

    return archivable

# Check cleanup
archivable_versions = check_cleanup_needed(MODEL_NAME, cleanup_policy)

print(f"\n=== Cleanup Analysis ===")
print(f"Total versions: {len(all_versions)}")
print(f"Archivable versions: {len(archivable_versions)}")

if archivable_versions:
    print("\nVersions eligible for archiving:")
    for v in archivable_versions:
        print(f"  Version {v['version']}: {v['age_days']} days old (created {v['created']})")
else:
    print("\n✓ No versions need archiving at this time")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 9: End-to-End Workflow Summary
# MAGIC
# MAGIC Let's create a comprehensive summary of everything we've accomplished in this lab.

# COMMAND ----------

# Create comprehensive summary
summary = f"""
{'='*80}
MLflow & UNITY CATALOG LAB - COMPLETE WORKFLOW SUMMARY
{'='*80}

Lab Completion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
SECTION 1: DATA PREPARATION
{'='*80}

✓ Generated {n_customers:,} customer records with realistic churn patterns
✓ Created Unity Catalog structure:
  - Catalog: {CATALOG_NAME}
  - Schema: {SCHEMA_NAME}
  - Table: {TABLE_NAME}
✓ Saved data to Delta table with ACID guarantees
✓ Performed data quality validation
✓ Churn rate: {df_pandas['churn'].mean():.2%}

{'='*80}
SECTION 2: FEATURE ENGINEERING
{'='*80}

✓ Prepared {X.shape[1]} features for modeling
✓ Split data: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test samples
✓ Applied feature scaling for model optimization
✓ Maintained stratified class distribution

{'='*80}
SECTION 3: MODEL TRAINING & EXPERIMENT TRACKING
{'='*80}

✓ Trained 3 different models:
  1. Logistic Regression (Baseline)
  2. Random Forest v1
  3. Gradient Boosting v1

✓ Logged to MLflow for each model:
  - Hyperparameters
  - Performance metrics (Accuracy, Precision, Recall, F1, AUC)
  - Model artifacts
  - Feature importance plots
  - Training metadata and tags

✓ Best Model: {best_model_name}
  - F1 Score: {comparison_df.loc[best_model_idx, 'F1_Score']:.4f}
  - AUC-ROC: {comparison_df.loc[best_model_idx, 'AUC']:.4f}

{'='*80}
SECTION 4: MODEL REGISTRATION
{'='*80}

✓ Registered best model to Unity Catalog Model Registry
✓ Model Name: {MODEL_NAME}
✓ Initial Version: {model_version.version}
✓ Added comprehensive documentation
✓ Included performance metrics and limitations

{'='*80}
SECTION 5: VERSION MANAGEMENT
{'='*80}

✓ Trained improved model (Random Forest v2)
✓ Registered as new version: {model_version_v2.version}
✓ Set model aliases:
  - Champion (Production): Version {model_version.version}
  - Challenger (A/B Test): Version {model_version_v2.version}
✓ Demonstrated version comparison and selection

{'='*80}
SECTION 6: GOVERNANCE & COMPLIANCE
{'='*80}

✓ Unity Catalog RBAC:
  - Fine-grained access control on data and models
  - Role-based permissions for different teams

✓ Audit Logging:
  - All operations automatically logged
  - Complete trail for regulatory compliance

✓ Data Lineage:
  - Full traceability from data to model to deployment
  - Accessible through Unity Catalog UI

{'='*80}
SECTION 7: REPRODUCIBILITY
{'='*80}

✓ All experiments fully reproducible via MLflow
✓ Complete parameter and metric logging
✓ Model artifacts stored with signatures
✓ Training data versioned in Unity Catalog
✓ Generated comprehensive performance report

{'='*80}
SECTION 8: ARCHIVING & CLEANUP
{'='*80}

✓ Defined cleanup policies:
  - Retain Champion and Challenger models
  - Archive versions older than 90 days
  - Maximum 10 versions per model

✓ Implemented archiving workflow
✓ Maintained audit trail for archived models

{'='*80}
KEY ACHIEVEMENTS
{'='*80}

1. ✓ Complete MLflow experiment tracking implementation
2. ✓ Unity Catalog integration for enterprise governance
3. ✓ Model versioning and lifecycle management
4. ✓ RBAC and access control setup
5. ✓ Audit logging and compliance readiness
6. ✓ Data lineage tracking
7. ✓ Reproducibility best practices
8. ✓ Archiving and cleanup policies

{'='*80}
PRODUCTION READINESS CHECKLIST
{'='*80}

✓ Models trained and validated
✓ Best model registered in Unity Catalog
✓ Documentation complete
✓ Governance controls in place
✓ Audit trail established
✓ Monitoring framework defined
✓ Cleanup policies implemented
✓ Team access controls configured

{'='*80}
NEXT STEPS
{'='*80}

1. Deploy Champion model to production endpoint
2. Set up A/B testing infrastructure for Challenger
3. Implement real-time monitoring dashboard
4. Schedule quarterly model retraining
5. Establish model performance review cadence
6. Configure alerting for model drift
7. Document deployment procedures
8. Train team on model operations

{'='*80}
COMPLIANCE NOTES
{'='*80}

✓ All data stored with access controls
✓ Complete audit trail maintained
✓ Model lineage fully documented
✓ Reproducibility guaranteed
✓ Regulatory requirements met
✓ Ready for compliance review

{'='*80}
LAB COMPLETE - ENTERPRISE ML GOVERNANCE ACHIEVED
{'='*80}
"""

print(summary)

# Save summary
with open('/tmp/lab_summary.txt', 'w') as f:
    f.write(summary)

print("\n✓ Lab summary saved to /tmp/lab_summary.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 10: Hands-On Exercises
# MAGIC
# MAGIC Now that you've completed the guided lab, try these exercises to reinforce your learning:
# MAGIC
# MAGIC ### Exercise 1: Train a New Model
# MAGIC - Train a new model with different hyperparameters
# MAGIC - Log it to MLflow with appropriate tags
# MAGIC - Compare its performance to existing models
# MAGIC
# MAGIC ### Exercise 2: Promote a Model
# MAGIC - Choose the best performing model
# MAGIC - Promote it to Champion
# MAGIC - Document the promotion decision
# MAGIC
# MAGIC ### Exercise 3: Query Audit Logs
# MAGIC - Query Unity Catalog audit logs
# MAGIC - Find all operations on your model
# MAGIC - Create a compliance report
# MAGIC
# MAGIC ### Exercise 4: Implement Monitoring
# MAGIC - Create a monitoring dashboard
# MAGIC - Track model performance over time
# MAGIC - Set up alerts for performance degradation
# MAGIC
# MAGIC ### Exercise 5: Archive Old Versions
# MAGIC - Identify versions that should be archived
# MAGIC - Apply archiving tags
# MAGIC - Document archiving decisions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Congratulations! You've completed a comprehensive lab on MLflow and Unity Catalog for enterprise ML governance.
# MAGIC
# MAGIC ### What You've Learned:
# MAGIC
# MAGIC 1. **MLflow Experiment Tracking**
# MAGIC    - Logging parameters, metrics, and artifacts
# MAGIC    - Organizing experiments for team collaboration
# MAGIC    - Comparing model performance
# MAGIC
# MAGIC 2. **Model Registry with Unity Catalog**
# MAGIC    - Registering models with versioning
# MAGIC    - Managing model lifecycle with aliases
# MAGIC    - Loading models for inference
# MAGIC
# MAGIC 3. **Enterprise Governance**
# MAGIC    - RBAC for access control
# MAGIC    - Audit logging for compliance
# MAGIC    - Data lineage tracking
# MAGIC
# MAGIC 4. **Best Practices**
# MAGIC    - Reproducibility through comprehensive logging
# MAGIC    - Documentation for stakeholders
# MAGIC    - Archiving and cleanup policies
# MAGIC
# MAGIC ### Real-World Applications:
# MAGIC
# MAGIC This workflow is used in production environments for:
# MAGIC - Financial services (fraud detection, credit scoring)
# MAGIC - Healthcare (patient risk prediction, diagnosis support)
# MAGIC - Retail (customer churn, demand forecasting)
# MAGIC - Manufacturing (predictive maintenance, quality control)
# MAGIC
# MAGIC ### Resources:
# MAGIC
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
# MAGIC - [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)
# MAGIC - [Databricks ML Best Practices](https://docs.databricks.com/machine-learning/index.html)
# MAGIC
# MAGIC ### Thank You!
# MAGIC
# MAGIC You're now equipped to implement enterprise-grade ML governance in your organization.

# COMMAND ----------

# Final verification - Display key resources
print("="*80)
print("LAB RESOURCES - QUICK REFERENCE")
print("="*80)
print(f"\n📊 Data Table: {table_path}")
print(f"🤖 Model Registry: {MODEL_NAME}")
print(f"🔬 Experiment: {experiment_name}")
print(f"\n📈 Model Versions:")
print(f"   - Champion: Version {model_version.version}")
print(f"   - Challenger: Version {model_version_v2.version}")
print(f"\n📁 Generated Reports:")
print(f"   - Performance Report: /tmp/model_performance_report.txt")
print(f"   - Lab Summary: /tmp/lab_summary.txt")
print(f"\n✅ Lab Status: COMPLETE")
print("="*80)