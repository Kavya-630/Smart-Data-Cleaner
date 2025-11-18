# 🧹 Smart Data Cleaner with Automatic Model Selection

A powerful Streamlit web application that intelligently cleans your datasets by automatically selecting the best imputation method for each column with missing values.

## 🌟 Key Features

### 🤖 **Intelligent Model Selection**
- Automatically evaluates multiple imputation methods (Mean, Median, KNN, MICE, Random Forest)
- Cross-validates each method to find the best performer
- Selects optimal method based on missing data type (MCAR, MAR, MNAR)

### 📊 **Comprehensive Data Cleaning**
- **Missing Values**: Smart imputation with automatic method selection
- **Outliers**: IQR-based detection and Winsorization
- **Duplicates**: Complete row duplicate removal
- **Multi-Format Support**: CSV, JSON, XML, Excel

### 📈 **Detailed Reporting**
- Model selection report showing which methods were evaluated
- Performance scores for each method
- Complete transparency on cleaning actions
- Before/After statistical comparison

---

## 🚀 Quick Start

### **Step 1: Installation**

```bash
# Clone or download the files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Run the Application**

```bash
streamlit run streamlit_data_cleaner.py
```

### **Step 3: Access the App**

- The app will automatically open in your browser at `http://localhost:8501`
- If not, manually navigate to the URL shown in terminal

---

## 📁 File Structure

```
smart-data-cleaner/
├── streamlit_data_cleaner.py    # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 💻 Usage Guide

### **1. Upload Your Dataset**
- Click the file uploader
- Select your file (CSV, JSON, XML, or Excel)
- Supported formats: `.csv`, `.json`, `.xml`, `.xlsx`, `.xls`

### **2. Review Original Data**
- View dataset overview with key metrics
- Check data preview and missing value summary
- Review data types and statistics

### **3. Start Smart Cleaning**
- Click the "🚀 Start Smart Cleaning" button
- Watch the progress bar as methods are evaluated
- See which method is selected for each column

### **4. Review Results**
- **Model Selection Report**: See which methods were tested and why one was chosen
- **Cleaning Summary**: View rows removed, missing values fixed, outliers handled
- **Detailed Report**: Column-by-column breakdown of actions taken
- **Quality Metrics**: Completeness, consistency, and outlier rates

### **5. Download Cleaned Data**
- Download in original format or convert to CSV/JSON/Excel
- Download detailed JSON report for documentation

---

## 🎯 How Model Selection Works

### **For Each Column with Missing Values:**

1. **Detect Missing Type**
   - **MCAR** (Missing Completely At Random): Random pattern, no correlation
   - **MAR** (Missing At Random): Depends on other observed variables
   - **MNAR** (Missing Not At Random): Depends on the missing value itself

2. **Evaluate Methods**
   - **MCAR**: Mean, Median, KNN
   - **MAR**: KNN (k=3,5,7), MICE, Random Forest
   - **MNAR**: MICE, Random Forest, KNN

3. **Select Best Method**
   - Cross-validation on complete cases
   - Choose method with best performance score
   - Apply to missing values

4. **Apply & Report**
   - Impute missing values using selected method
   - Document decision and performance scores
   - Provide full transparency

---

## 📊 Supported Imputation Methods

| Method | Description | Best For | Pros | Cons |
|--------|-------------|----------|------|------|
| **Mean/Median** | Simple average/middle value | MCAR, quick fixes | Fast, simple | Ignores relationships |
| **Mode** | Most frequent value | Categorical data | Simple | May not capture patterns |
| **KNN (k=3,5,7)** | K-Nearest Neighbors | MAR data | Considers relationships | Slower, needs tuning |
| **MICE** | Multiple Imputation by Chained Equations | MNAR, complex patterns | Sophisticated | Slowest |
| **Random Forest** | Tree-based prediction | Complex relationships | Accurate, robust | Computational cost |

---

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Disk Space**: 500MB for dependencies

### **Recommended Dataset Sizes:**
- **Small** (<10MB): All methods work well
- **Medium** (10-100MB): KNN and MICE may be slower
- **Large** (>100MB): Consider using Mean/Median for speed

---

## 📋 Example Workflow

### **Sample Dataset: Customer Data**

**Original:**
```
- 1000 rows × 15 columns
- 145 missing values across 4 columns
- 23 outliers in 2 columns
- 12 duplicate rows
```

**Cleaning Process:**
1. **Age column**: MCAR detected → Tested Mean, Median, KNN → **KNN (k=5) selected** (best score)
2. **Income column**: MAR detected → Tested KNN, MICE, RF → **Random Forest selected** (best score)
3. **Email column**: >50% missing → **Column removed**
4. **Phone column**: MCAR detected → Tested Mode → **Mode selected**
5. **Outliers**: Salary and transaction amount → **Winsorized**
6. **Duplicates**: 12 found → **Removed**

**Result:**
```
- 988 rows × 14 columns
- 0 missing values
- 98.5% completeness
- 99.2% consistency
```

---

## 🎨 Features in Detail

### **1. Automatic Missing Type Detection**
```python
# The system automatically detects:
- MCAR: Random missing pattern, no correlation with other features
- MAR: Missing depends on observed values (e.g., older people skip online surveys)
- MNAR: Missing depends on the value itself (e.g., high earners don't report income)
```

### **2. Smart Outlier Handling**
```python
# IQR Method:
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

# Winsorization (Capping):
- Values below lower bound → Set to lower bound
- Values above upper bound → Set to upper bound
```

### **3. Comprehensive Reporting**
- Original vs Cleaned dimensions
- Column-by-column action log
- Performance scores for evaluated methods
- Statistical before/after comparison
- Data quality metrics

---

## 🐛 Troubleshooting

### **Issue: "No module named 'sklearn'"**
```bash
pip install --upgrade scikit-learn
```

### **Issue: "Cannot read Excel file"**
```bash
pip install openpyxl
```

### **Issue: "Memory Error"**
- Your dataset may be too large
- Try with a smaller sample first
- Close other applications to free RAM

### **Issue: "Slow Performance"**
- Normal for large datasets with MICE/RF methods
- Consider using faster methods (Mean/Median/KNN)
- Reduce dataset size for testing

### **Issue: XML parsing error**
```bash
pip install lxml
```

---

## 📊 Performance Tips

1. **For Large Datasets** (>100MB):
   - Use Mean/Median/Mode for speed
   - Process in batches
   - Consider sampling for initial testing

2. **For Many Missing Values**:
   - Columns with >50% missing are auto-removed
   - Focus cleaning on important columns first

3. **For Complex Relationships**:
   - Random Forest works best but is slower
   - MICE is good for multiple imputation
   - KNN balances speed and accuracy

---

## 🔐 Data Privacy

- **All processing happens locally** on your machine
- No data is sent to external servers
- Files are processed in-memory only
- Nothing is stored permanently by the app

---

## 📞 Common Questions

**Q: How long does cleaning take?**
A: Depends on dataset size and methods used. Small datasets (<10MB) typically take 10-30 seconds. Large datasets may take several minutes if using MICE or Random Forest.

**Q: Can I choose the imputation method manually?**
A: Currently, the app automatically selects the best method. This ensures optimal results based on cross-validation scores.

**Q: What if my data format isn't supported?**
A: Currently supports CSV, JSON, XML, and Excel. For other formats, convert to one of these first.

**Q: Will it work with time-series data?**
A: The current version is designed for tabular data. Time-series specific methods may be added in future versions.

**Q: Can I see the code for method selection?**
A: Yes! The entire process is transparent. Check the Model Selection Report for details on what was evaluated and why.

---

## 🎓 Understanding the Reports

### **Model Selection Report**
```
Column: age
├── Missing Type: MAR
├── Methods Evaluated: KNN (k=3), KNN (k=5), KNN (k=7), MICE
├── Performance Scores:
│   ├── KNN (k=3): 12.45
│   ├── KNN (k=5): 11.89  ← Best (lowest error)
│   ├── KNN (k=7): 12.12
│   └── MICE: 13.05
└── Selected: KNN (k=5)
```

### **Cleaning Report**
- Shows exactly what action was taken for each column
- Includes count and percentage of affected values
- Documents method used and reasoning

### **Quality Metrics**
- **Completeness**: % of non-missing values
- **Consistency**: % of values within expected ranges
- **Outlier Rate**: % of remaining outliers after cleaning

---

## 🚀 Advanced Usage

### **Processing Multiple Files**
```bash
# Process files one at a time through the web interface
# Or create a batch processing script using the underlying classes
```

### **Customizing Thresholds**
```python
# In the code, you can adjust:
- Missing value threshold for column removal (default: 50%)
- IQR multiplier for outliers (default: 1.5)
- KNN neighbors (default: 3, 5, 7)
- Cross-validation folds (default: 3)
```

---

## 📚 References

- [Scikit-learn Imputation Documentation](https://scikit-learn.org/stable/modules/impute.html)
- [Missing Data Theory](https://en.wikipedia.org/wiki/Missing_data)
- [IQR Method for Outliers](https://en.wikipedia.org/wiki/Interquartile_range)
- [MICE Algorithm Paper](https://www.jstatsoft.org/article/view/v045i03)

---

## 📝 License

MIT License - Free to use and modify

---

## 🤝 Contributing

Suggestions for improvements:
- Additional imputation methods
- Time-series support
- Custom threshold settings
- Batch processing capability
- API endpoint

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the Common Questions
3. Examine the example workflow


for Demo - https://smart-data-cleaner-2nzhhq9phihvxxs5wfffso.streamlit.app/
---

**Built with ❤️ using Streamlit, scikit-learn, and pandas**

**Version:** 2.0.0  

**Last Updated:** 2025-11-18
