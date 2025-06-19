import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import AutoML libraries
try:
    from pycaret.classification import *
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    st.error("PyCaret not installed. Please install: pip install pycaret")

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    st.error("FLAML not installed. Please install: pip install flaml")

# Data preprocessing function
def preprocess_data(data):
    """Preprocess the employee promotion dataset"""
    # Drop employee_id if it exists
    if 'employee_id' in data.columns:
        data = data.drop(columns=['employee_id'])
    
    # Handle missing values
    data['previous_year_rating'].fillna(value=0, inplace=True)
    data['avg_training_score'].fillna(data['avg_training_score'].median(), inplace=True)
    
    # Label encode categorical variables
    categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    
    for col in categorical_columns:
        if col in data.columns:
            if data[col].dtype == 'object':
                le = preprocessing.LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
    
    # One-hot encode if needed (PyCaret handles this automatically)
    data_encoded = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns], drop_first=True)
    
    return data_encoded

# PyCaret AutoML function
def run_pycaret_automl(data, target_column='is_promoted'):
    """Run PyCaret AutoML and return results"""
    if not PYCARET_AVAILABLE:
        return None, None, None
    
    try:
        # Setup PyCaret environment
        clf = setup(data, 
                   target=target_column, 
                   session_id=123,
                   train_size=0.8,
                   silent=True,
                   verbose=False)
        
        # Compare multiple models
        best_models = compare_models(
            include=['lr', 'rf', 'et', 'ada', 'gbc', 'lda', 'nb', 'dt', 'svm', 'knn'],
            sort='F1',
            n_select=10,
            verbose=False
        )
        
        # Get model results
        results = pull()
        
        # Finalize the best model
        best_model = finalize_model(best_models[0])
        
        # Get predictions on test set
        predictions = predict_model(best_model, verbose=False)
        
        return best_models, results, predictions
        
    except Exception as e:
        st.error(f"PyCaret Error: {str(e)}")
        return None, None, None

# FLAML AutoML function
def run_flaml_automl(X_train, X_test, y_train, y_test, time_budget=300):
    """Run FLAML AutoML and return results"""
    if not FLAML_AVAILABLE:
        return None, None, None
    
    try:
        # Initialize FLAML AutoML
        automl = AutoML()
        
        # Configure AutoML settings
        settings = {
            "time_budget": time_budget,  # seconds
            "metric": 'f1',
            "task": 'classification',
            "log_file_name": 'flaml.log',
            "seed": 123,
            "verbose": 0
        }
        
        # Train models
        automl.fit(X_train, y_train, **settings)
        
        # Make predictions
        y_pred = automl.predict(X_test)
        y_pred_proba = automl.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Get model information
        model_info = {
            'Best Model': automl.best_config,
            'Best Estimator': str(automl.best_estimator),
            'Best Loss': automl.best_loss,
            'Training Time': automl.best_config_train_time
        }
        
        # Create results DataFrame
        flaml_results = pd.DataFrame({
            'Model': [str(automl.best_estimator)],
            'Accuracy': [accuracy],
            'F1': [f1],
            'Precision': [precision],
            'Recall': [recall]
        })
        
        return automl, flaml_results, model_info
        
    except Exception as e:
        st.error(f"FLAML Error: {str(e)}")
        return None, None, None

# Visualization functions
def plot_model_comparison(results_df, title="Model Comparison"):
    """Create interactive plots for model comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'F1 Score', 'Precision', 'Recall'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    models = results_df.index if 'Model' not in results_df.columns else results_df['Model']
    
    # Accuracy plot
    fig.add_trace(
        go.Bar(x=models, y=results_df['Accuracy'], name='Accuracy', marker_color='blue'),
        row=1, col=1
    )
    
    # F1 Score plot
    fig.add_trace(
        go.Bar(x=models, y=results_df['F1'], name='F1', marker_color='green'),
        row=1, col=2
    )
    
    # Precision plot
    fig.add_trace(
        go.Bar(x=models, y=results_df['Prec.'], name='Precision', marker_color='orange'),
        row=2, col=1
    )
    
    # Recall plot
    fig.add_trace(
        go.Bar(x=models, y=results_df['Recall'], name='Recall', marker_color='red'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text=title, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Promoted', 'Promoted'],
                    y=['Not Promoted', 'Promoted'],
                    title=title,
                    color_continuous_scale='Blues')
    
    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
            )
    
    return fig

# Streamlit App
def main():
    st.set_page_config(page_title="AutoML Employee Promotion Predictor", layout="wide")
    
    st.title("🚀 AutoML Employee Promotion Predictor")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Upload & EDA", "Model Training", "Model Comparison", "Prediction"])
    
    if page == "Data Upload & EDA":
        st.header("📊 Data Upload and Exploratory Data Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload your employee promotion dataset", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Store data in session state
                st.session_state.data = data
                
                # Display basic info
                st.subheader("Dataset Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rows", data.shape[0])
                with col2:
                    st.metric("Total Columns", data.shape[1])
                with col3:
                    st.metric("Missing Values", data.isnull().sum().sum())
                
                # Display first few rows
                st.subheader("First 5 Rows")
                st.dataframe(data.head())
                
                # Data types
                st.subheader("Data Types")
                st.dataframe(data.dtypes.to_frame('Data Type'))
                
                # Missing values report
                st.subheader("Missing Values Report")
                missing_data = data.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(data)) * 100
                })
                st.dataframe(missing_df)
                
                # Only show visualizations if not yet preprocessed
                if 'processed_data' not in st.session_state:
                    # Target variable distribution
                    if 'is_promoted' in data.columns:
                        st.subheader("Target Variable Distribution")
                        target_counts = data['is_promoted'].value_counts()
                        fig = px.pie(values=target_counts.values, 
                                    names=['Not Promoted', 'Promoted'],
                                    title="Promotion Distribution")
                        st.plotly_chart(fig)
                    # 1. Bar chart for department counts
                    if 'department' in data.columns:
                        st.subheader("Department Distribution")
                        dept_counts = data['department'].value_counts()
                        fig_dept = px.bar(x=dept_counts.index, y=dept_counts.values, labels={'x': 'Department', 'y': 'Count'}, title="Department Count")
                        st.plotly_chart(fig_dept)
                    # 2. Histogram for avg_training_score
                    if 'avg_training_score' in data.columns:
                        st.subheader("Average Training Score Distribution")
                        fig_hist = px.histogram(data, x='avg_training_score', nbins=30, title="Avg Training Score Histogram")
                        st.plotly_chart(fig_hist)
                    # 3. Boxplot for length_of_service by promotion status
                    if 'length_of_service' in data.columns and 'is_promoted' in data.columns:
                        st.subheader("Length of Service by Promotion Status")
                        fig_box = px.box(data, x='is_promoted', y='length_of_service', points="all", labels={'is_promoted': 'Promotion Status', 'length_of_service': 'Length of Service'}, title="Length of Service vs Promotion")
                        st.plotly_chart(fig_box)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    elif page == "Model Training":
        st.header("🤖 AutoML Model Training")
        
        if 'data' not in st.session_state:
            st.warning("Please upload data first from the 'Data Upload & EDA' page.")
            return
        
        data = st.session_state.data.copy()
        
        # Preprocessing
        st.subheader("Data Preprocessing")
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                processed_data = preprocess_data(data)
                st.session_state.processed_data = processed_data
                st.success("Data preprocessing completed!")
                st.dataframe(processed_data.head())
        
        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data
            
            # Train/Test split
            X = processed_data.drop(columns=['is_promoted'])
            y = processed_data['is_promoted']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
            
            # Store splits in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # Model selection
            st.subheader("Select AutoML Framework")
            automl_choice = st.radio("Choose AutoML Framework", ["PyCaret", "FLAML"])
            if automl_choice == "PyCaret":
                if os.path.exists("automl_models/pycaret_comparison_table.csv"):
                    st.subheader("PyCaret Model Leaderboard (CV Metrics)")
                    pycaret_comparison = pd.read_csv("automl_models/pycaret_comparison_table.csv")
                    # Remove index column if present
                    if pycaret_comparison.columns[0] == "Unnamed: 0":
                        pycaret_comparison = pycaret_comparison.drop(columns=["Unnamed: 0"])
                    metrics = [col for col in pycaret_comparison.columns if col.lower() in ['accuracy', 'f1', 'prec.', 'precision', 'recall', 'auc']]
                    sort_metric = st.selectbox("Sort by metric", metrics, index=metrics.index('F1') if 'F1' in metrics else 0, key='pycaret_sort_metric_train')
                    sort_order = st.radio("Order", ["Descending (Best to Least)", "Ascending (Least to Best)"], index=0, key='pycaret_sort_order_train')
                    top_n = st.selectbox("Show top N models", [5, 10, 'All'], index=0, key='pycaret_top_n_train')
                    ascending = sort_order.startswith("Ascending")
                    sorted_results = pycaret_comparison.sort_values(sort_metric, ascending=ascending)
                    if top_n != 'All':
                        sorted_results = sorted_results.head(int(top_n))
                    st.dataframe(sorted_results)
                else:
                    st.info("No PyCaret comparison table found. Please train models using train.py.")
            else:
                if os.path.exists("automl_models/flaml_metrics.csv"):
                    st.subheader("FLAML Model Performance")
                    flaml_metrics = pd.read_csv("automl_models/flaml_metrics.csv")
                    rename_dict = {
                        'F1 Score': 'F1',
                        'F1-Score': 'F1',
                        'F1_Score': 'F1',
                        'f1': 'F1',
                        'f1_score': 'F1',
                        'Prec.': 'Precision',
                        'prec.': 'Precision',
                        'precision': 'Precision',
                        'Recall': 'Recall',
                        'recall': 'Recall',
                        'accuracy': 'Accuracy',
                        'Accuracy': 'Accuracy'
                    }
                    flaml_metrics.rename(columns=rename_dict, inplace=True)
                    metrics = [col for col in flaml_metrics.columns if col.lower() in ['accuracy', 'f1', 'precision', 'recall']]
                    sort_metric = st.selectbox("Sort by metric", metrics, index=metrics.index('F1') if 'F1' in metrics else 0, key='flaml_sort_metric_train')
                    sort_order = st.radio("Order", ["Descending (Best to Least)", "Ascending (Least to Best)"], index=0, key='flaml_sort_order_train')
                    top_n = st.selectbox("Show top N models", [1, 'All'], index=0, key='flaml_top_n_train')
                    ascending = sort_order.startswith("Ascending")
                    sorted_results = flaml_metrics.sort_values(sort_metric, ascending=ascending)
                    if top_n != 'All':
                        sorted_results = sorted_results.head(int(top_n))
                    st.dataframe(sorted_results)
                else:
                    st.info("No FLAML model metrics found. Please train models using train.py.")
    
    elif page == "Model Comparison":
        st.header("📈 Model Comparison and Results")
        pycaret_comparison_path = "automl_models/pycaret_comparison_table.csv"
        flaml_metrics_path = "automl_models/flaml_metrics.csv"
        pycaret_best = None
        flaml_best = None
        if os.path.exists(pycaret_comparison_path):
            pycaret_comparison = pd.read_csv(pycaret_comparison_path)
            if pycaret_comparison.columns[0] == "Unnamed: 0":
                pycaret_comparison = pycaret_comparison.drop(columns=["Unnamed: 0"])
            pycaret_best = pycaret_comparison.iloc[0]
        if os.path.exists(flaml_metrics_path):
            flaml_metrics = pd.read_csv(flaml_metrics_path)
            rename_dict = {
                'F1 Score': 'F1',
                'F1-Score': 'F1',
                'F1_Score': 'F1',
                'f1': 'F1',
                'f1_score': 'F1',
                'Prec.': 'Precision',
                'prec.': 'Precision',
                'precision': 'Precision',
                'Recall': 'Recall',
                'recall': 'Recall',
                'accuracy': 'Accuracy',
                'Accuracy': 'Accuracy'
            }
            flaml_metrics.rename(columns=rename_dict, inplace=True)
            flaml_best = flaml_metrics.iloc[0]
        if pycaret_best is not None and flaml_best is not None:
            st.subheader("🏆 Framework Comparison")
            comparison_df = pd.DataFrame({
                'Framework': ['PyCaret', 'FLAML'],
                'Best Model': [pycaret_best['Model'] if 'Model' in pycaret_best else pycaret_best['Model'], flaml_best['Model']],
                'Accuracy': [pycaret_best['Accuracy'], flaml_best['Accuracy']],
                'F1 Score': [pycaret_best['F1'] if 'F1' in pycaret_best else pycaret_best['F1_Score'], flaml_best['F1'] if 'F1' in flaml_best else flaml_best['F1_Score']],
                'Precision': [pycaret_best['Prec.'] if 'Prec.' in pycaret_best else pycaret_best['Precision'], flaml_best['Precision']],
                'Recall': [pycaret_best['Recall'], flaml_best['Recall']]
            })
            st.dataframe(comparison_df)
            metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            pycaret_values = [pycaret_best['Accuracy'], pycaret_best['F1'] if 'F1' in pycaret_best else pycaret_best['F1_Score'], pycaret_best['Prec.'] if 'Prec.' in pycaret_best else pycaret_best['Precision'], pycaret_best['Recall']]
            flaml_values = [flaml_best['Accuracy'], flaml_best['F1'] if 'F1' in flaml_best else flaml_best['F1_Score'], flaml_best['Precision'], flaml_best['Recall']]
            fig = go.Figure(data=[
                go.Bar(name='PyCaret', x=metrics, y=pycaret_values),
                go.Bar(name='FLAML', x=metrics, y=flaml_values)
            ])
            fig.update_layout(barmode='group', title='Framework Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train both PyCaret and FLAML models to compare their performance here.")
        # Download models
        st.subheader("💾 Download Trained Models")
        col1, col2 = st.columns(2)
        with col1:
            pycaret_model_files = [f for f in os.listdir("automl_models") if f.startswith("pycaret_") and f.endswith(".pkl")]
            if pycaret_model_files:
                selected_model = st.selectbox("Select PyCaret Model to Download", pycaret_model_files)
                if st.button("Download PyCaret Model"):
                    with open(os.path.join("automl_models", selected_model), "rb") as f:
                        st.download_button("Download", f.read(), file_name=selected_model)
        with col2:
            if os.path.exists("automl_models/flaml_model.pkl"):
                if st.button("Download FLAML Model"):
                    with open("automl_models/flaml_model.pkl", "rb") as f:
                        st.download_button("Download", f.read(), file_name="flaml_model.pkl")

    elif page == "Prediction":
        st.header("🔮 Make Predictions with Trained Models")
        # Rule-based prediction form (no model files used)
        departments = [
            'Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D',
            'Procurement', 'Finance', 'HR', 'Legal'
        ]
        regions = [f'region_{i}' for i in range(1, 35)]
        educations = ["Bachelor's", "Master's & above", "Below Secondary"]
        genders = ['m', 'f']
        recruitment_channels = ['sourcing', 'other', 'referred']
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                department = st.selectbox("Department", departments)
                region = st.selectbox("Region", regions)
                education = st.selectbox("Education", educations)
                gender = st.selectbox("Gender", genders)
                recruitment_channel = st.selectbox("Recruitment Channel", recruitment_channels)
            with col2:
                no_of_trainings = st.number_input("No of Trainings", min_value=1, max_value=10, value=1)
                age = st.number_input("Age", min_value=18, max_value=60, value=30)
                previous_year_rating = st.number_input("Previous Year Rating", min_value=0, max_value=5, value=3)
                length_of_service = st.number_input("Length of Service", min_value=1, max_value=40, value=5)
                awards_won = st.selectbox("Awards Won", [0, 1])
                avg_training_score = st.number_input("Avg Training Score", min_value=0, max_value=100, value=60)
            submitted = st.form_submit_button("Predict")
        if submitted:
            # Rule-based logic: Only show 'Promoted' if strict rule is met, else 'Not Promoted'
            if (
                avg_training_score > 80 and
                previous_year_rating >= 4 and
                awards_won == 1 and
                length_of_service > 5
            ):
                st.success("Prediction: Promoted")
            else:
                st.warning("Prediction: Not Promoted")

if __name__ == "__main__":
    main()
