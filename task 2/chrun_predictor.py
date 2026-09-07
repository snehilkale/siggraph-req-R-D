import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CustomerChurnPredictor:
    """
    A comprehensive customer churn prediction system.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """
        Load and perform initial data inspection.
        """
        print("="*80)
        print("📊 LOADING CUSTOMER CHURN DATASET")
        print("="*80)
        
        try:
            self.df = pd.read_csv(filepath)
            print(f"✅ Dataset loaded successfully!")
            print(f"📈 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
        except FileNotFoundError:
            print("❌ File not found. Please ensure the dataset is in the correct location.")
            print("Download from: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset")
            raise
        
        return self.df
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        """
        print("\n" + "="*80)
        print("🔍 EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic info
        print("\n📊 Dataset Info:")
        print("-"*40)
        self.df.info()
        
        # Statistical summary
        print("\n📊 Statistical Summary:")
        print("-"*40)
        print(self.df.describe())
        
        # Check for missing values
        print("\n🔍 Missing Values:")
        print("-"*40)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Values'] > 0])
        
        # Check target variable distribution
        print("\n🎯 Target Variable Distribution:")
        print("-"*40)
        if 'Churn' in self.df.columns:
            target = 'Churn'
        elif 'churn' in self.df.columns:
            target = 'churn'
        else:
            # Assume last column is target
            target = self.df.columns[-1]
        
        self.target_column = target
        churn_counts = self.df[target].value_counts()
        print(churn_counts)
        print(f"\nChurn Rate: {(churn_counts.get('Yes', churn_counts.get(1, 0)) / len(self.df) * 100):.2f}%")
        
        # Visualize target distribution
        plt.figure(figsize=(8, 5))
        if not pd.api.types.is_numeric_dtype(self.df[target]):
            sns.countplot(data=self.df, x=target)
        else:
            sns.countplot(data=self.df, x=target)
        plt.title('Distribution of Customer Churn', fontsize=14, fontweight='bold')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.savefig('churn_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Identify numerical and categorical columns
        self.numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature columns
        if target in self.numerical_cols:
            self.numerical_cols.remove(target)
        if target in self.categorical_cols:
            self.categorical_cols.remove(target)
        
        print(f"\n📊 Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"📊 Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        
        # Plot correlation with churn
        self.plot_correlations()
        
        return target
    
    def plot_correlations(self):
        """
        Plot correlation matrix and churn correlations.
        """
        # Convert target to numeric for correlation
        df_corr = self.df.copy()
        if not pd.api.types.is_numeric_dtype(df_corr[self.target_column]):
            df_corr[self.target_column] = df_corr[self.target_column].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables for correlation
        for col in self.categorical_cols:
            if col != self.target_column:
                df_corr[col] = LabelEncoder().fit_transform(df_corr[col].astype(str))
        
        # Calculate correlations with churn
        churn_corr = df_corr.corr()[self.target_column].sort_values(ascending=False)
        
        print("\n📊 Top Factors Correlated with Churn:")
        print("-"*40)
        print(churn_corr)
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Plot top correlations bar chart
        plt.figure(figsize=(10, 6))
        top_corr = churn_corr.drop(self.target_column).head(10)
        top_corr.plot(kind='bar')
        plt.title('Top 10 Factors Correlated with Churn', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Correlation with Churn')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_correlations.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """
        Handle missing values, encode categorical variables, and scale numerical attributes.
        """
        print("\n" + "="*80)
        print("🔄 DATA PREPROCESSING")
        print("="*80)
        
        df_processed = self.df.copy()
        
        # 1. Handle missing values
        print("\n📝 Handling missing values...")
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if col in self.numerical_cols:
                    # Fill numerical with median
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    print(f"  - Filled {col} with median: {df_processed[col].median():.2f}")
                else:
                    # Fill categorical with mode
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    print(f"  - Filled {col} with mode: {df_processed[col].mode()[0]}")
        
        # 2. Encode target variable
        print("\n🎯 Encoding target variable...")
        if not pd.api.types.is_numeric_dtype(df_processed[self.target_column]):
            df_processed[self.target_column] = df_processed[self.target_column].map({'Yes': 1, 'No': 0})
            print(f"  - Encoded {self.target_column}: Yes→1, No→0")
        
        # 3. Encode categorical variables
        print("\n📊 Encoding categorical variables...")
        for col in self.categorical_cols:
            if col != self.target_column:
                # Use Label Encoding for simplicity (can be changed to One-Hot)
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                print(f"  - Encoded {col}: {len(le.classes_)} categories")
        
        # 4. Handle TotalCharges if it's object type (common in telecom datasets)
        for col in self.numerical_cols:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
                print(f"  - Converted {col} to numeric")
        
        # 5. Prepare features and target
        self.feature_columns = [col for col in df_processed.columns if col != self.target_column]
        X = df_processed[self.feature_columns]
        y = df_processed[self.target_column]
        
        print(f"\n✅ Preprocessing complete!")
        print(f"📊 Features: {X.shape}")
        print(f"🎯 Target: {y.shape}")
        print(f"📈 Churn rate: {(y.mean() * 100):.2f}%")
        
        return X, y
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Address class imbalance using SMOTE.
        """
        print("\n" + "="*80)
        print("⚖️ HANDLING CLASS IMBALANCE")
        print("="*80)
        
        print(f"\n📊 Original class distribution:")
        print(f"  - Non-churn: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
        print(f"  - Churn: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\n📊 Balanced class distribution (after SMOTE):")
        print(f"  - Non-churn: {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).mean()*100:.2f}%)")
        print(f"  - Churn: {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).mean()*100:.2f}%)")
        
        return X_train_balanced, y_train_balanced
    
    def train_models(self, X_train, y_train):
        """
        Train Random Forest and XGBoost models.
        """
        print("\n" + "="*80)
        print("🤖 TRAINING MODELS")
        print("="*80)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"  ✅ {name} trained successfully!")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate models using multiple metrics.
        """
        print("\n" + "="*80)
        print("📈 MODEL EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            print(f"\n📊 {name} Performance:")
            print("-"*40)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Display results
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            # Classification report
            print("\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("  Confusion Matrix:")
            print(f"    True Negatives: {cm[0][0]}")
            print(f"    False Positives: {cm[0][1]}")
            print(f"    False Negatives: {cm[1][0]}")
            print(f"    True Positives: {cm[1][1]}")
        
        # Plot ROC curves
        self.plot_roc_curves(y_test)
        
        # Plot Precision-Recall curves
        self.plot_precision_recall_curves(y_test)
        
        return self.results
    
    def plot_roc_curves(self, y_test):
        """
        Plot ROC curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = self.results[name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = self.results[name]['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_test):
        """
        Plot Precision-Recall curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = self.results[name]['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=name, linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def extract_feature_importance(self):
        """
        Extract and plot feature importances.
        """
        print("\n" + "="*80)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Store feature importances
                self.feature_importances[name] = pd.DataFrame({
                    'Feature': [self.feature_columns[i] for i in indices],
                    'Importance': importances[indices]
                })
                
                print(f"\n📊 {name} - Top 10 Features:")
                print("-"*40)
                for i in range(min(10, len(indices))):
                    print(f"  {i+1}. {self.feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")
                
                # Plot feature importances
                self.plot_feature_importance(name, top_n=5)
        
        return self.feature_importances
    
    def plot_feature_importance(self, model_name, top_n=5):
        """
        Plot feature importance for a specific model.
        """
        importance_df = self.feature_importances[model_name].head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['Importance'].values, align='center')
        plt.yticks(range(len(importance_df)), importance_df['Feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Drivers of Customer Churn - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def cross_validate_models(self, X, y, cv=5):
        """
        Perform cross-validation for more robust evaluation.
        """
        print("\n" + "="*80)
        print("🔄 CROSS-VALIDATION")
        print("="*80)
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\n📊 {name} - {cv}-Fold Cross-Validation:")
            print("-"*40)
            
            # Calculate cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='roc_auc')
            
            print(f"  ROC-AUC Scores: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"  Mean ROC-AUC: {cv_scores.mean():.4f}")
            print(f"  Std Dev: {cv_scores.std():.4f}")
    
    def create_prediction_function(self, model_name='XGBoost'):
        """
        Create a user-friendly prediction function.
        """
        def predict_churn(customer_data):
            """
            Predict if a customer will churn.
            
            Args:
                customer_data (dict): Dictionary with customer features
            
            Returns:
                str: Prediction result
            """
            # Convert input to DataFrame
            input_df = pd.DataFrame([customer_data])
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
            
            # Ensure all columns are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns
            input_df = input_df[self.feature_columns]
            
            # Scale features if scaler was fitted
            if hasattr(self, 'scaler') and self.scaler is not None:
                input_df = self.scaler.transform(input_df)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            if prediction == 1:
                result = f"🚨 HIGH RISK OF CHURN (Probability: {probability[1]*100:.2f}%)"
            else:
                result = f"✅ LOW RISK OF CHURN (Probability: {probability[0]*100:.2f}%)"
            
            return result
        
        return predict_churn
    
    def compare_models(self):
        """
        Create comparison visualization of model performances.
        """
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON")
        print("="*80)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T[metrics]
        print("\n📊 Performance Comparison:")
        print("-"*40)
        print(comparison_df)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, name in enumerate(model_names):
            values = [self.results[name][metric] for metric in metrics]
            offset = width * i
            ax.bar(x + offset, values, width, label=name)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Identify best model based on ROC-AUC
        best_model_name = comparison_df['roc_auc'].idxmax()
        best_score = comparison_df['roc_auc'].max()
        self.best_model = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC-AUC Score: {best_score:.4f}")

def main():
    """
    Main function to run the churn prediction pipeline.
    """
    print("\n" + "="*80)
    print("🎯 CUSTOMER CHURN PREDICTOR")
    print("="*80)
    
    # Initialize predictor
    predictor = CustomerChurnPredictor(random_state=42)
    
    # Load data
    try:
        # Try common dataset filenames
        for filename in ['customer_churn.csv', 'churn.csv', 'Customer-Churn.csv', 'customer_churn_dataset.csv']:
            try:
                df = predictor.load_data(filename)
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("No dataset found. Please place the dataset in the project folder.")
    except FileNotFoundError:
        print("\n❌ Dataset not found!")
        print("Please download from: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset")
        print("And save as 'customer_churn.csv' in the project folder.")
        return
    
    # Exploratory Data Analysis
    target = predictor.explore_data()
    
    # Preprocess data
    X, y = predictor.preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Testing set: {X_test.shape}")
    
    # Scale numerical features
    print("\n📏 Scaling numerical features...")
    numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
    X_train[numerical_features] = predictor.scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = predictor.scaler.transform(X_test[numerical_features])
    print("✅ Features scaled!")
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = predictor.handle_class_imbalance(X_train, y_train)
    
    # Train models
    predictor.train_models(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    predictor.evaluate_models(X_test, y_test)
    
    # Extract feature importance
    predictor.extract_feature_importance()
    
    # Cross-validation
    predictor.cross_validate_models(X, y, cv=5)
    
    # Compare models
    predictor.compare_models()
    
    # Create prediction function
    predict_churn = predictor.create_prediction_function(model_name='XGBoost')
    
    # Test prediction function with sample data
    print("\n" + "="*80)
    print("🧪 TESTING PREDICTION FUNCTION")
    print("="*80)
    
    # Create sample customer data
    sample_customers = [
        {
            'tenure': 1,
            'MonthlyCharges': 85.5,
            'TotalCharges': 85.5,
            'Contract': 'Month-to-month',
            'PaymentMethod': 'Electronic check',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'TechSupport': 'No'
        },
        {
            'tenure': 60,
            'MonthlyCharges': 45.0,
            'TotalCharges': 2700.0,
            'Contract': 'Two year',
            'PaymentMethod': 'Credit card (automatic)',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'TechSupport': 'Yes'
        }
    ]
    
    for i, customer in enumerate(sample_customers, 1):
        print(f"\n📊 Sample Customer {i}:")
        for key, value in customer.items():
            print(f"  {key}: {value}")
        
        try:
            result = predict_churn(customer)
            print(f"  Prediction: {result}")
        except Exception as e:
            print(f"  ⚠️ Could not predict (missing features): {e}")
    
    print("\n" + "="*80)
    print("✅ CHURN PREDICTION COMPLETE!")
    print("="*80)
    
    # Summary of findings
    print("\n📋 SUMMARY:")
    print("-"*40)
    if predictor.feature_importances:
        best_features = predictor.feature_importances['XGBoost'].head(5)
        print("Top 5 Drivers of Customer Churn:")
        for i, row in best_features.iterrows():
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
    
    if predictor.best_model:
        print(f"\nBest Model: {predictor.best_model}")
        print(f"ROC-AUC Score: {predictor.results[predictor.best_model]['roc_auc']:.4f}")

if __name__ == "__main__":
    main()