import streamlit as st
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from io import BytesIO
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import subprocess
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# MISTRAL MODEL INTEGRATION
# ==========================================

class MistralHelper:
    def __init__(self, model_name=None):
        """
        If model_name is None, set the default path to your Mistral model file
        """
        if model_name is None:
            # 🔥🔥 PUT YOUR MODEL PATH HERE
            self.model_name = r"C:\Users\Navya sree\Downloads\Streamlit data cleaner\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        else:
            self.model_name = model_name

        self.available = self.check_availability()

    
    def check_availability(self):
        """Check if Mistral model is available"""
        try:
            result = subprocess.run(
                ['mistral', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_cleaning_suggestions(self, df):
        """
        Get intelligent cleaning suggestions from Mistral
        
        Args:
            df: pandas DataFrame
        
        Returns:
            str: Suggestions from Mistral model
        """
        if not self.available:
            return None
        
        try:
            # Prepare dataset summary for Mistral
            summary = self._prepare_dataset_summary(df)
            
            prompt = f"""You are a data cleaning expert. Analyze this dataset and provide specific recommendations:

{summary}

Please provide:
1. Which columns need the most attention
2. Recommended imputation strategies for each column with missing values
3. Potential outlier columns that need investigation
4. Any data quality concerns
5. Suggested preprocessing steps

Keep your response concise and actionable."""

            # Call Mistral model
            result = subprocess.run(
                ['mistral', 'run', self.model_name, '--prompt', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_column_analysis(self, df, column_name):
        """
        Get detailed analysis for a specific column
        
        Args:
            df: pandas DataFrame
            column_name: Name of column to analyze
        
        Returns:
            str: Analysis from Mistral
        """
        if not self.available:
            return None
        
        try:
            col_data = df[column_name]
            
            prompt = f"""Analyze this data column and suggest the best cleaning approach:

Column: {column_name}
Data Type: {col_data.dtype}
Total Values: {len(col_data)}
Missing Values: {col_data.isna().sum()} ({col_data.isna().sum()/len(col_data)*100:.2f}%)
Unique Values: {col_data.nunique()}

Statistics:
{col_data.describe().to_string() if col_data.dtype in ['int64', 'float64'] else 'Categorical data'}

What is the best approach to handle this column? Consider:
1. Missing value imputation method
2. Outlier detection and handling
3. Any transformations needed

Provide a brief, specific recommendation."""

            result = subprocess.run(
                ['mistral', 'run', self.model_name, '--prompt', prompt],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception as e:
            return None
    
    def explain_cleaning_results(self, report):
        """
        Get Mistral's interpretation of cleaning results
        
        Args:
            report: Cleaning report dictionary
        
        Returns:
            str: Interpretation from Mistral
        """
        if not self.available:
            return None
        
        try:
            prompt = f"""Interpret these data cleaning results and provide insights:

Original Dataset: {report['original_shape']['rows']} rows × {report['original_shape']['columns']} columns
Cleaned Dataset: {report['final_shape']['rows']} rows × {report['final_shape']['columns']} columns

Missing Values Handled: {len(report['missing_values'])} columns
Outliers Handled: {len(report['outliers'])} columns
Duplicates Removed: {report['duplicates']['removed']}

Data Quality:
- Completeness: {report['data_quality']['completeness']}
- Consistency: {report['data_quality']['consistency']}

Provide:
1. Overall assessment of the data quality improvement
2. Any concerns or warnings
3. Recommendations for next steps (feature engineering, modeling, etc.)

Keep it concise and actionable."""

            result = subprocess.run(
                ['mistral', 'run', self.model_name, '--prompt', prompt],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception as e:
            return None
    
    def _prepare_dataset_summary(self, df):
        """Prepare a concise dataset summary for Mistral"""
        summary = f"""
Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns

Columns and Types:
{df.dtypes.to_string()}

Missing Values:
{df.isna().sum()[df.isna().sum() > 0].to_string() if df.isna().sum().sum() > 0 else 'No missing values'}

Numeric Columns Statistics:
{df.describe().to_string() if len(df.select_dtypes(include=['int64', 'float64']).columns) > 0 else 'No numeric columns'}

Duplicates: {df.duplicated().sum()} rows
"""
        return summary

# Page configuration
st.set_page_config(
    page_title="Smart Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'model_selection_report' not in st.session_state:
    st.session_state.model_selection_report = None


class SmartDataCleaner:
    """
    Advanced data cleaning with automatic model selection
    """
    
    def __init__(self):
        self.report = {
            'original_shape': None,
            'model_selection': {},
            'missing_values': {},
            'outliers': {},
            'duplicates': {},
            'final_shape': None,
            'data_quality': {}
        }
        self.label_encoders = {}
    
    def detect_missing_type(self, df, column):
        """Detect if missing data is MCAR, MAR, or MNAR"""
        missing_mask = df[column].isna()
        
        if missing_mask.sum() == 0:
            return "No Missing"
        
        # Check correlation with other columns
        correlations = []
        for col in df.columns:
            if col != column and df[col].dtype in ['int64', 'float64']:
                try:
                    corr = np.corrcoef(missing_mask.astype(int), 
                                      df[col].fillna(df[col].mean()))[0, 1]
                    if abs(corr) > 0.3:
                        correlations.append((col, corr))
                except:
                    pass
        
        if len(correlations) > 0:
            return "MAR"
        elif missing_mask.sum() / len(df) < 0.05:
            return "MCAR"
        else:
            return "MNAR"
    
    def evaluate_imputation_method(self, df, column, method_name, imputer):
        """
        Evaluate imputation method using cross-validation on non-missing data
        Returns RMSE score (lower is better)
        """
        try:
            # Get complete cases for this column
            complete_data = df[df[column].notna()].copy()
            
            if len(complete_data) < 50:
                return None, "Insufficient data"
            
            # Take a sample if dataset is too large
            if len(complete_data) > 1000:
                complete_data = complete_data.sample(n=1000, random_state=42)
            
            X = complete_data.drop(columns=[column])
            y = complete_data[column]
            
            # Handle categorical columns in X
            X_encoded = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
            
            # Fill any remaining NaN in X
            X_encoded = X_encoded.fillna(X_encoded.mean())
            
            # Choose appropriate model
            if df[column].dtype in ['int64', 'float64']:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                scoring = 'neg_mean_squared_error'
            else:
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                scoring = 'accuracy'
            
            # Cross-validation
            scores = cross_val_score(model, X_encoded, y, cv=3, scoring=scoring)
            score = -scores.mean() if scoring == 'neg_mean_squared_error' else scores.mean()
            
            return score, None
            
        except Exception as e:
            return None, str(e)
    
    def select_best_imputation_method(self, df, column, missing_type):
        """
        Automatically select the best imputation method for a column
        """
        methods_to_test = []
        
        if df[column].dtype in ['int64', 'float64']:
            # Numeric column
            if missing_type == 'MCAR':
                methods_to_test = [
                    ('Mean', SimpleImputer(strategy='mean')),
                    ('Median', SimpleImputer(strategy='median')),
                    ('KNN', KNNImputer(n_neighbors=5))
                ]
            elif missing_type == 'MAR':
                methods_to_test = [
                    ('KNN (k=3)', KNNImputer(n_neighbors=3)),
                    ('KNN (k=5)', KNNImputer(n_neighbors=5)),
                    ('KNN (k=7)', KNNImputer(n_neighbors=7)),
                    ('MICE', IterativeImputer(max_iter=10, random_state=42)),
                    ('Random Forest', 'RF')
                ]
            else:  # MNAR
                methods_to_test = [
                    ('MICE', IterativeImputer(max_iter=10, random_state=42)),
                    ('Random Forest', 'RF'),
                    ('KNN (k=5)', KNNImputer(n_neighbors=5))
                ]
        else:
            # Categorical column
            methods_to_test = [
                ('Mode', SimpleImputer(strategy='most_frequent')),
                ('Constant ("Missing")', SimpleImputer(strategy='constant', fill_value='Missing'))
            ]
        
        # Evaluate each method
        results = {}
        for method_name, imputer in methods_to_test:
            if method_name == 'Random Forest':
                # Special handling for RF
                score, error = self.evaluate_rf_imputation(df, column)
            else:
                score, error = self.evaluate_imputation_method(df, column, method_name, imputer)
            
            if score is not None:
                results[method_name] = score
        
        # Select best method (lowest score for regression, highest for classification)
        if results:
            if df[column].dtype in ['int64', 'float64']:
                best_method = min(results, key=results.get)
            else:
                best_method = max(results, key=results.get)
            
            return best_method, results
        else:
            # Fallback to simple method
            if df[column].dtype in ['int64', 'float64']:
                return 'Median', {'Median': 'fallback'}
            else:
                return 'Mode', {'Mode': 'fallback'}
    
    def evaluate_rf_imputation(self, df, column):
        """Evaluate Random Forest imputation"""
        try:
            complete_data = df[df[column].notna()].copy()
            
            if len(complete_data) < 50:
                return None, "Insufficient data"
            
            if len(complete_data) > 1000:
                complete_data = complete_data.sample(n=1000, random_state=42)
            
            X = complete_data.drop(columns=[column])
            y = complete_data[column]
            
            # Encode categorical variables
            X_encoded = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
            
            X_encoded = X_encoded.fillna(X_encoded.mean())
            
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            scores = cross_val_score(model, X_encoded, y, cv=3, scoring='neg_mean_squared_error')
            score = -scores.mean()
            
            return score, None
            
        except Exception as e:
            return None, str(e)
    
    def apply_imputation(self, df, column, method_name):
        """Apply the selected imputation method"""
        df_copy = df.copy()
        
        if method_name == 'Mean':
            df_copy[column].fillna(df[column].mean(), inplace=True)
        elif method_name == 'Median':
            df_copy[column].fillna(df[column].median(), inplace=True)
        elif method_name == 'Mode':
            df_copy[column].fillna(df[column].mode()[0] if len(df[column].mode()) > 0 else 'Missing', inplace=True)
        elif 'KNN' in method_name:
            k = int(method_name.split('=')[1].strip(')')) if '=' in method_name else 5
            imputer = KNNImputer(n_neighbors=k)
            df_copy[[column]] = imputer.fit_transform(df_copy[[column]])
        elif method_name == 'MICE':
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df_copy[[column]] = imputer.fit_transform(df_copy[[column]])
        elif method_name == 'Random Forest':
            df_copy = self.apply_rf_imputation(df_copy, column)
        elif 'Constant' in method_name:
            df_copy[column].fillna('Missing', inplace=True)
        
        return df_copy
    
    def apply_rf_imputation(self, df, column):
        """Apply Random Forest imputation"""
        try:
            # Separate data with and without missing values
            train_data = df[df[column].notna()].copy()
            predict_data = df[df[column].isna()].copy()
            
            if len(predict_data) == 0:
                return df
            
            # Prepare features
            X_train = train_data.drop(columns=[column])
            y_train = train_data[column]
            X_predict = predict_data.drop(columns=[column])
            
            # Encode categorical variables
            for col in X_train.columns:
                if X_train[col].dtype == 'object':
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(df[col].astype(str).unique())
                    
                    X_train[col] = self.label_encoders[col].transform(X_train[col].astype(str))
                    X_predict[col] = self.label_encoders[col].transform(X_predict[col].astype(str))
            
            # Fill any remaining NaN
            X_train = X_train.fillna(X_train.mean())
            X_predict = X_predict.fillna(X_train.mean())
            
            # Train and predict
            model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_predict)
            
            # Fill in predictions
            df_copy = df.copy()
            df_copy.loc[df_copy[column].isna(), column] = predictions
            
            return df_copy
            
        except Exception as e:
            # Fallback to median
            df_copy = df.copy()
            df_copy[column].fillna(df[column].median(), inplace=True)
            return df_copy
    
    def smart_impute_missing_values(self, df, progress_callback=None):
        """
        Smart missing value imputation with automatic model selection
        """
        missing_report = {}
        model_selection_report = {}
        df_cleaned = df.copy()
        
        cols_with_missing = [col for col in df.columns if df[col].isna().sum() > 0]
        total_cols = len(cols_with_missing)
        
        for idx, col in enumerate(cols_with_missing):
            if progress_callback:
                progress_callback((idx + 1) / total_cols, f"Processing {col}...")
            
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            # Remove column if >50% missing
            if missing_pct > 50:
                df_cleaned = df_cleaned.drop(columns=[col])
                missing_report[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_pct),
                    'type': 'N/A',
                    'action': 'Column removed (>50% missing)',
                    'selected_method': 'Removal'
                }
                continue
            
            # Detect missing type
            missing_type = self.detect_missing_type(df, col)
            
            # Select best method
            best_method, method_scores = self.select_best_imputation_method(df_cleaned, col, missing_type)
            
            # Apply imputation
            df_cleaned = self.apply_imputation(df_cleaned, col, best_method)
            
            missing_report[col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct),
                'type': missing_type,
                'action': f'{best_method} imputation (auto-selected)',
                'selected_method': best_method
            }
            
            model_selection_report[col] = {
                'missing_type': missing_type,
                'methods_evaluated': list(method_scores.keys()),
                'best_method': best_method,
                'scores': {k: float(v) if isinstance(v, (int, float)) else str(v) 
                          for k, v in method_scores.items()}
            }
        
        self.report['missing_values'] = missing_report
        self.report['model_selection'] = model_selection_report
        
        return df_cleaned
    
    def detect_and_handle_outliers(self, df):
        """Detect and handle outliers using IQR method"""
        outlier_report = {}
        df_cleaned = df.copy()
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                # Winsorization (capping)
                df_cleaned[col] = np.where(
                    df_cleaned[col] > upper_bound, upper_bound,
                    np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
                )
                
                outlier_report[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(df)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': 'IQR Method',
                    'action': f'Winsorized (capped at [{lower_bound:.2f}, {upper_bound:.2f}])'
                }
        
        self.report['outliers'] = outlier_report
        return df_cleaned
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        original_count = len(df)
        df_cleaned = df.drop_duplicates(keep='first')
        duplicates_removed = original_count - len(df_cleaned)
        
        self.report['duplicates'] = {
            'found': int(duplicates_removed),
            'removed': int(duplicates_removed),
            'kept_first': True
        }
        
        return df_cleaned
    
    def calculate_data_quality(self, df):
        """Calculate final data quality metrics"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        # Count remaining outliers
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_count = 0
        for col in numeric_cols:
            try:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_count += (z_scores > 3).sum()
            except:
                pass
        
        outlier_rate = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        
        self.report['data_quality'] = {
            'completeness': f"{completeness:.2f}%",
            'consistency': f"{100 - outlier_rate:.2f}%",
            'outlier_rate': f"{outlier_rate:.2f}%"
        }
    
    def clean(self, df, progress_callback=None):
        """Main cleaning pipeline with progress tracking"""
        self.report['original_shape'] = {'rows': int(df.shape[0]), 'columns': int(df.shape[1])}
        
        # Step 1: Handle missing values with smart model selection
        if progress_callback:
            progress_callback(0.1, "Analyzing missing values...")
        df = self.smart_impute_missing_values(df, progress_callback)
        
        # Step 2: Handle outliers
        if progress_callback:
            progress_callback(0.7, "Detecting and handling outliers...")
        df = self.detect_and_handle_outliers(df)
        
        # Step 3: Remove duplicates
        if progress_callback:
            progress_callback(0.9, "Removing duplicates...")
        df = self.remove_duplicates(df)
        
        self.report['final_shape'] = {'rows': int(df.shape[0]), 'columns': int(df.shape[1])}
        
        # Calculate final quality metrics
        if progress_callback:
            progress_callback(1.0, "Calculating quality metrics...")
        self.calculate_data_quality(df)
        
        return df, self.report


def load_file(uploaded_file):
    """Load different file formats"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'xml':
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            data = []
            for child in root:
                data.append({elem.tag: elem.text for elem in child})
            df = pd.DataFrame(data)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def convert_df_to_format(df, file_format, original_filename):
    """Convert dataframe to specified format"""
    output_filename = f"cleaned_{original_filename.rsplit('.', 1)[0]}.{file_format}"
    
    if file_format == 'csv':
        return df.to_csv(index=False).encode('utf-8'), output_filename
    elif file_format == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8'), output_filename
    elif file_format == 'xlsx':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue(), output_filename
    elif file_format == 'xml':
        xml_data = df.to_xml(index=False)
        return xml_data.encode('utf-8'), output_filename


# ==========================================
# STREAMLIT APP
# ==========================================

# Header
st.markdown('<div class="main-header">🧹 Smart Data Cleaner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automatic Model Selection for Intelligent Data Cleaning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses **intelligent model selection** to automatically choose the best cleaning method for your data.
    
    **Features:**
    - 🤖 Auto-selects best imputation method
    - 📊 Evaluates multiple algorithms
    - 🎯 Handles MCAR, MAR, MNAR
    - 🔍 IQR-based outlier detection
    - 🔄 Duplicate removal
    
    **Supported Formats:**
    - CSV, JSON, XML, Excel
    
    **Methods Evaluated:**
    - Mean/Median/Mode
    - KNN (k=3, 5, 7)
    - MICE
    - Random Forest
    """)
    
    st.header("📊 How It Works")
    st.markdown("""
    1. **Upload** your dataset
    2. **Analyze** missing patterns
    3. **Evaluate** multiple methods
    4. **Select** best performer
    5. **Clean** and download
    """)

# Main content
uploaded_file = st.file_uploader(
    "📁 Upload Your Dataset",
    type=['csv', 'json', 'xml', 'xlsx', 'xls'],
    help="Drag and drop or click to browse"
)

if uploaded_file:
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_file(uploaded_file)
    
    if df is not None:
        st.session_state.original_df = df
        
        # Display original data info
        st.subheader("📊 Original Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Rows", df.shape[0])
        with col2:
            st.metric("📋 Columns", df.shape[1])
        with col3:
            st.metric("❌ Missing Values", df.isna().sum().sum())
        with col4:
            missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("📉 Missing %", f"{missing_pct:.2f}%")
        
        # Show data preview
        with st.expander("🔍 View Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show data info
        with st.expander("📈 Data Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.write("**Missing Value Summary:**")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing': df.isna().sum().values,
                    'Percentage': (df.isna().sum() / len(df) * 100).values
                })
                missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
        
        # Clean button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clean_button = st.button("🚀 Start Smart Cleaning", type="primary", use_container_width=True)
        
        if clean_button:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            with st.spinner("Cleaning in progress..."):
                cleaner = SmartDataCleaner()
                cleaned_df, report = cleaner.clean(df, progress_callback=update_progress)
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = report
                st.session_state.model_selection_report = report.get('model_selection', {})
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Data cleaning completed successfully!")
            st.balloons()
        
        # Display results if available
        if st.session_state.cleaning_report:
            report = st.session_state.cleaning_report
            cleaned_df = st.session_state.cleaned_df
            
            st.markdown("---")
            st.subheader("📊 Cleaning Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rows_removed = report['original_shape']['rows'] - report['final_shape']['rows']
                st.metric(
                    "Rows Removed",
                    rows_removed,
                    delta=f"-{rows_removed}",
                    delta_color="inverse"
                )
            
            with col2:
                missing_total = sum([v['count'] for v in report['missing_values'].values() if 'count' in v])
                st.metric("Missing Values Fixed", missing_total)
            
            with col3:
                outlier_total = sum([v['count'] for v in report['outliers'].values()])
                st.metric("Outliers Handled", outlier_total)
            
            with col4:
                st.metric("Duplicates Removed", report['duplicates']['removed'])
            
            # Model Selection Report
            if st.session_state.model_selection_report:
                st.markdown("---")
                st.subheader("🤖 Smart Model Selection Report")
                st.markdown("""
                <div class="info-box">
                <b>How it works:</b> For each column with missing values, multiple imputation methods were evaluated 
                and the best performing method was automatically selected based on cross-validation scores.
                </div>
                """, unsafe_allow_html=True)
                
                for col, info in st.session_state.model_selection_report.items():
                    with st.expander(f"📊 {col} - Selected Method: **{info['best_method']}**", expanded=False):
                        st.write(f"**Missing Type Detected:** {info['missing_type']}")
                        st.write(f"**Methods Evaluated:** {', '.join(info['methods_evaluated'])}")
                        st.write(f"**Best Method:** {info['best_method']}")
                        
                        if info['scores']:
                            st.write("**Performance Scores:**")
                            scores_df = pd.DataFrame([
                                {'Method': k, 'Score': v}
                                for k, v in info['scores'].items()
                            ])
                            st.dataframe(scores_df, use_container_width=True)
            
            # Detailed Report
            st.markdown("---")
            st.subheader("📋 Detailed Cleaning Report")
            
            # Dataset Dimensions
            with st.expander("📐 Dataset Dimensions", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Original Size",
                        f"{report['original_shape']['rows']} × {report['original_shape']['columns']}"
                    )
                with col2:
                    st.metric(
                        "Cleaned Size",
                        f"{report['final_shape']['rows']} × {report['final_shape']['columns']}"
                    )
            
            # Missing Values
            if report['missing_values']:
                with st.expander("🔍 Missing Values Treatment", expanded=True):
                    for col, info in report['missing_values'].items():
                        st.markdown(f"**{col}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"Count: **{info['count']}** ({info['percentage']:.2f}%)")
                        with col2:
                            st.write(f"Type: **{info['type']}**")
                        with col3:
                            st.write(f"Method: **{info['selected_method']}**")
                        st.write(f"Action: {info['action']}")
                        st.markdown("---")
            
            # Outliers
            if report['outliers']:
                with st.expander("⚠️ Outlier Detection & Treatment", expanded=True):
                    total_outliers = sum([v['count'] for v in report['outliers'].values()])
                    st.write(f"**Total Outliers Detected:** {total_outliers}")
                    st.write(f"**Detection Method:** IQR (Interquartile Range)")
                    st.markdown("---")
                    
                    for col, info in report['outliers'].items():
                        st.markdown(f"**{col}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Count: **{info['count']}** ({info['percentage']:.2f}%)")
                            st.write(f"Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
                        with col2:
                            st.write(f"Method: **{info['method']}**")
                            st.write(f"Action: {info['action']}")
                        st.markdown("---")
            
            # Duplicates
            with st.expander("🔄 Duplicate Removal", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duplicates Found", report['duplicates']['found'])
                with col2:
                    st.metric("Duplicates Removed", report['duplicates']['removed'])
                st.write("**Strategy:** Kept first occurrence, removed subsequent duplicates")
            
            # Data Quality
            with st.expander("✅ Final Data Quality Metrics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    completeness = float(report['data_quality']['completeness'].strip('%'))
                    st.metric(
                        "Completeness",
                        report['data_quality']['completeness'],
                        delta=f"{completeness - 100:.2f}%" if completeness < 100 else "Perfect!"
                    )
                with col2:
                    st.metric("Consistency", report['data_quality']['consistency'])
                with col3:
                    st.metric("Outlier Rate", report['data_quality']['outlier_rate'])
            
            # Display cleaned data
            st.markdown("---")
            st.subheader("🎯 Cleaned Dataset Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Final Rows", cleaned_df.shape[0])
            with col2:
                st.metric("📋 Final Columns", cleaned_df.shape[1])
            with col3:
                st.metric("✨ Missing Values", cleaned_df.isna().sum().sum())
            with col4:
                quality_score = float(report['data_quality']['completeness'].strip('%'))
                st.metric("🏆 Quality Score", f"{quality_score:.1f}%")
            
            st.dataframe(cleaned_df.head(20), use_container_width=True)
            
            # Statistical comparison
            with st.expander("📊 Statistical Comparison: Before vs After"):
                st.write("**Numerical Columns Summary**")
                
                numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_cols) > 0:
                    comparison_data = []
                    
                    for col in numeric_cols:
                        if col in df.columns:
                            comparison_data.append({
                                'Column': col,
                                'Before Mean': f"{df[col].mean():.2f}",
                                'After Mean': f"{cleaned_df[col].mean():.2f}",
                                'Before Std': f"{df[col].std():.2f}",
                                'After Std': f"{cleaned_df[col].std():.2f}",
                                'Before Missing': int(df[col].isna().sum()),
                                'After Missing': int(cleaned_df[col].isna().sum())
                            })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Download section
            st.markdown("---")
            st.subheader("⬇️ Download Cleaned Data")
            
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Main download button (same format as input)
            file_data, filename = convert_df_to_format(
                cleaned_df, 
                file_extension, 
                uploaded_file.name
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=file_data,
                    file_name=filename,
                    mime='application/octet-stream',
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                # Download report as JSON
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="📄 Download Report (JSON)",
                    data=report_json,
                    file_name=f"cleaning_report_{uploaded_file.name.split('.')[0]}.json",
                    mime='application/json',
                    use_container_width=True
                )
            
            # Additional format downloads
            st.markdown("**Download in other formats:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data, csv_name = convert_df_to_format(cleaned_df, 'csv', uploaded_file.name)
                st.download_button(
                    "💾 CSV",
                    csv_data,
                    csv_name,
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data, json_name = convert_df_to_format(cleaned_df, 'json', uploaded_file.name)
                st.download_button(
                    "💾 JSON",
                    json_data,
                    json_name,
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                xlsx_data, xlsx_name = convert_df_to_format(cleaned_df, 'xlsx', uploaded_file.name)
                st.download_button(
                    "💾 Excel",
                    xlsx_data,
                    xlsx_name,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Success message
            st.markdown("---")
            st.markdown("""
            <div class="success-box">
            <h3>✅ Cleaning Complete!</h3>
            <p><b>Your data has been successfully cleaned using intelligent model selection.</b></p>
            <p>Each column was analyzed and the best imputation method was automatically selected based on:</p>
            <ul>
                <li>Missing data type (MCAR, MAR, MNAR)</li>
                <li>Cross-validation performance scores</li>
                <li>Data characteristics and relationships</li>
            </ul>
            <p>Download your cleaned data above and check the detailed report for full transparency.</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Landing page when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h2>👋 Welcome to Smart Data Cleaner!</h2>
        <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
            Upload your dataset to get started with intelligent automated cleaning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Smart Selection
        Automatically evaluates and selects the best imputation method for each column
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Multiple Methods
        Tests Mean, Median, KNN, MICE, and Random Forest to find optimal approach
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Full Transparency
        Detailed reports showing which methods were evaluated and why one was chosen
        """)
    
    st.markdown("---")
    
    # Example workflow
    st.markdown("""
    ### 🚀 How It Works
    
    1. **Upload** - Drop your CSV, JSON, XML, or Excel file
    2. **Analyze** - The system detects missing patterns (MCAR, MAR, MNAR)
    3. **Evaluate** - Multiple imputation methods are tested automatically
    4. **Select** - Best performing method is chosen for each column
    5. **Clean** - Outliers removed, duplicates eliminated
    6. **Download** - Get your cleaned data in the original format
    
    ### 📋 What Gets Cleaned?
    
    - **Missing Values**: Intelligently imputed using the best method
    - **Outliers**: Detected using IQR and handled via Winsorization
    - **Duplicates**: Complete row duplicates removed
    - **Data Quality**: Comprehensive quality metrics calculated
    
    ### 🎯 Supported Methods
    
    | Method | Best For | Speed |
    |--------|----------|-------|
    | Mean/Median | MCAR data | ⚡⚡⚡ Fast |
    | KNN | MAR data | ⚡⚡ Medium |
    | MICE | MNAR data | ⚡ Slower |
    | Random Forest | Complex relationships | ⚡ Slower |
    
    """)
    
    # Example data
    st.markdown("---")
    st.markdown("### 📝 Want to Try with Sample Data?")
    
    if st.button("🎲 Generate Sample Messy Dataset"):
        # Generate sample data
        np.random.seed(42)
        n_rows = 200
        
        sample_data = {
            'customer_id': range(1, n_rows + 1),
            'age': np.random.randint(18, 80, n_rows),
            'income': np.random.normal(50000, 20000, n_rows),
            'credit_score': np.random.randint(300, 850, n_rows),
            'transaction_count': np.random.poisson(10, n_rows),
            'satisfaction': np.random.randint(1, 11, n_rows)
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Add missing values
        mcar_indices = np.random.choice(sample_df.index, size=int(0.1 * n_rows), replace=False)
        sample_df.loc[mcar_indices, 'age'] = np.nan
        
        mar_indices = np.random.choice(sample_df.index, size=int(0.15 * n_rows), replace=False)
        sample_df.loc[mar_indices, 'income'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(sample_df.index, 10, replace=False)
        sample_df.loc[outlier_indices, 'income'] = sample_df.loc[outlier_indices, 'income'] * 10
        
        # Add duplicates
        duplicate_rows = sample_df.sample(n=10).copy()
        sample_df = pd.concat([sample_df, duplicate_rows], ignore_index=True)
        
        st.success("✅ Sample dataset generated!")
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Download sample
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Dataset",
            data=sample_csv,
            file_name="sample_messy_data.csv",
            mime="text/csv"
        )
        
        st.info("💡 Download this sample dataset and upload it above to see the smart cleaning in action!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Made with ❤️ using Streamlit | Smart Data Cleaner v2.0</p>
    <p>Automatic model selection powered by scikit-learn and Random Forest</p>
</div>
""", unsafe_allow_html=True)