import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from pycaret.classification import *
from flaml import AutoML
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for outputs
output_dir = "automl_models"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
# (You may want to make this a relative path for portability)
data = pd.read_csv(r"C:\Users\Tayyaba\Downloads\employee_promotion.csv")

# --- EDA ---
print("\n📊 Exploratory Data Analysis (EDA)")
print("=" * 50)

# 1. Dataset Info
print("\nDataset Info:")
print(data.info())

# 2. Missing Data Report
print("\nMissing Data Report:")
missing_data = data.isnull().sum()
print(missing_data)
missing_data.to_csv(os.path.join(output_dir, 'missing_data_report.csv'))

# 3. Column Types
print("\nColumn Types:")
column_types = data.dtypes
print(column_types)
pd.DataFrame(column_types, columns=['Type']).to_csv(os.path.join(output_dir, 'column_types.csv'))

# 4. Numerical Distributions
numerical_cols = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
print("\nGenerating Numerical Distributions...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'numerical_distributions.png'))
plt.close()

# 5. Categorical Distributions
categorical_cols = ['department', 'region', 'education', 'gender', 'recruitment_channel']
print("\nGenerating Categorical Distributions...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(y=data[col].dropna(), order=data[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'categorical_distributions.png'))
plt.close()

# 6. Target Distribution
print("\nTarget Distribution (is_promoted):")
print(data['is_promoted'].value_counts(normalize=True))
plt.figure(figsize=(6, 4))
sns.countplot(x=data['is_promoted'])
plt.title('Distribution of is_promoted')
plt.xlabel('is_promoted')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

# --- Preprocessing ---
print("\n🔧 Preprocessing")
print("=" * 50)

# Remove duplicate rows
before_dupes = data.shape[0]
data = data.drop_duplicates()
after_dupes = data.shape[0]
print(f"   ✅ Removed {before_dupes - after_dupes} duplicate rows")

# Drop unnecessary columns
if 'employee_id' in data.columns:
    data = data.drop(columns=['employee_id'])
    print("   ✅ Dropped 'employee_id' column")

# Handle missing values
missing_before = data.isnull().sum().sum()
if 'previous_year_rating' in data.columns:
    data['previous_year_rating'] = data['previous_year_rating'].fillna(value=0)
if 'education' in data.columns:
    data['education'] = data['education'].fillna(data['education'].mode()[0])
if 'avg_training_score' in data.columns:
    data['avg_training_score'] = data['avg_training_score'].fillna(data['avg_training_score'].mode()[0])
for col in categorical_cols:
    data[col] = data[col].replace('', data[col].mode()[0])
missing_after = data.isnull().sum().sum()
print(f"   ✅ Handled missing values: {missing_before} -> {missing_after}")

# Split features and target
X = data.drop(columns=['is_promoted'])
y = data['is_promoted']

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(f"   ✅ Preprocessing complete. Final shape: {data.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- AutoML ---
print("\n🤖 AutoML Training")
print("=" * 50)

# Initialize dictionaries to store results
pycaret_results = {
    'model_objects': {},
    'model_names': [],
    'metrics': []
}
flaml_results = {
    'model_object': None,
    'model_name': '',
    'metrics': {}
}

# PyCaret AutoML
print("\n🚀 Running PyCaret AutoML...")
try:
    # Setup PyCaret environment
    train_data = X_train.copy()
    train_data['is_promoted'] = y_train
    clf_setup = setup(
        data=train_data,
        target='is_promoted',
        session_id=42,
        normalize=False,  # Already scaled
        feature_selection=True,
        categorical_features=categorical_cols,
        use_gpu=False,
        verbose=False,
        fix_imbalance=False  # No SMOTE
    )

    # Train all available models
    print("Comparing all available models...")
    pycaret_models = compare_models(
        sort='F1',
        n_select=10,  # Select top 10 models for evaluation on test set
        verbose=True
    )

    # Save comparison table
    print("\nPyCaret Model Comparison Table:")
    comparison_table = pull()
    print(comparison_table)
    comparison_table.to_csv(os.path.join(output_dir, 'pycaret_comparison_table.csv'))

    # Process each model (handle both list and single model)
    if isinstance(pycaret_models, list):
        models_to_process = pycaret_models
    else:
        models_to_process = [pycaret_models]
    for i, model in enumerate(models_to_process):
        try:
            tuned_model = tune_model(model, optimize='F1', verbose=False)
            final_model = finalize_model(tuned_model)
            if isinstance(final_model, Pipeline):
                classifier = final_model.steps[-1][1]
                model_name = f"PyCaret_{type(classifier).__name__}"
            else:
                model_name = f"PyCaret_{type(final_model).__name__}"
            model_name = f"{model_name}_{i+1}" if model_name in pycaret_results['model_names'] else model_name
            test_data = X_test.copy()
            predictions = predict_model(final_model, data=test_data)
            y_pred = predictions['prediction_label'].values if 'prediction_label' in predictions else predictions['Label'].values
            y_pred_proba = predictions['prediction_score'].values if 'prediction_score' in predictions else None
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1_Score': f1_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'AUC_ROC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            }
            pycaret_results['model_objects'][model_name] = final_model
            pycaret_results['model_names'].append(model_name)
            pycaret_results['metrics'].append(metrics)
            print(f"  ✓ {model_name}: F1={metrics['F1_Score']:.4f}, Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, AUC-ROC={metrics['AUC_ROC'] if metrics['AUC_ROC'] is not None else 'N/A'}")
            # Save individual model
            save_model(final_model, os.path.join(output_dir, f'pycaret_{model_name.lower()}'))
        except Exception as e:
            print(f"  ✗ Error processing PyCaret model {i+1}: {str(e)}")
    with open(os.path.join(output_dir, 'pycaret_results.pkl'), 'wb') as f:
        pickle.dump(pycaret_results, f)
    pd.DataFrame(pycaret_results['metrics']).to_csv(os.path.join(output_dir, 'pycaret_metrics.csv'), index=False)
    print("   💾 PyCaret results saved as 'pycaret_results.pkl' and 'pycaret_metrics.csv'")
except Exception as e:
    print(f"Error running PyCaret AutoML: {str(e)}")

# FLAML AutoML
print("\n🚀 Running FLAML AutoML...")
try:
    # Check for catboost, remove if not installed
    estimator_list = ['lgbm', 'rf', 'xgboost', 'extra_tree', 'catboost']
    try:
        import catboost
    except ImportError:
        estimator_list.remove('catboost')
        print("catboost not installed, skipping for FLAML.")
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        time_budget=600,  # 10 minutes
        metric="f1",
        log_file_name=os.path.join(output_dir, "flaml_log.txt"),
        seed=42,
        estimator_list=estimator_list,
        eval_method='cv',
        split_ratio=0.8,
        n_splits=5
    )
    y_pred = automl.predict(X_test)
    y_pred_proba = automl.predict_proba(X_test)[:, 1]
    flaml_results['model_name'] = f"FLAML_{automl.best_estimator}"
    flaml_results['model_object'] = automl
    flaml_results['metrics'] = {
        'Model': flaml_results['model_name'],
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'AUC_ROC': roc_auc_score(y_test, y_pred_proba),
        'Best_Config': automl.best_config,
        'Feature_Importance': dict(zip(X_train.columns, automl.feature_importances_)) if hasattr(automl, 'feature_importances_') else {}
    }
    print(f"  ✓ {flaml_results['model_name']}: F1={flaml_results['metrics']['F1_Score']:.4f}, Accuracy={flaml_results['metrics']['Accuracy']:.4f}, Precision={flaml_results['metrics']['Precision']:.4f}, Recall={flaml_results['metrics']['Recall']:.4f}, AUC-ROC={flaml_results['metrics']['AUC_ROC']:.4f}")
    with open(os.path.join(output_dir, 'flaml_model.pkl'), 'wb') as f:
        pickle.dump(automl, f)
    pd.DataFrame([flaml_results['metrics']]).to_csv(os.path.join(output_dir, 'flaml_metrics.csv'), index=False)
    print("   💾 FLAML model and metrics saved.")
except Exception as e:
    print(f"Error running FLAML AutoML: {str(e)}")

# Save preprocessors and feature names
with open(os.path.join(output_dir, 'preprocessors.pkl'), 'wb') as f:
    pickle.dump({
        # 'label_encoders': label_encoders,
        'scaler': scaler,
        'numerical_cols': numerical_cols,
        'feature_columns': list(X_train.columns)
    }, f)
print("   💾 Preprocessors saved as 'preprocessors.pkl'")

print(f"\n✓ EDA, Preprocessing, and AutoML completed! Results saved to {output_dir}/")
print("EDA outputs: missing_data_report.csv, column_types.csv, numerical_distributions.png, categorical_distributions.png, target_distribution.png")
print("AutoML outputs: pycaret_results.pkl, pycaret_metrics.csv, pycaret_comparison_table.csv, flaml_model.pkl, flaml_metrics.csv")

# List all files in output_dir
print("\nFiles saved in the output directory:")
for fname in os.listdir(output_dir):
    print(" -", fname)
