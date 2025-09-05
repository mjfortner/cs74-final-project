# CS74 Final Project: Text Classification and Clustering

This project implements machine learning models for text classification and clustering on product review data. The main analysis is conducted in a Jupyter notebook that explores various classification approaches and clustering techniques.

## Project Overview

This project focuses on:
- **Binary Classification**: Predicting whether reviews are positive/negative using different rating cutoffs (1, 2, 3, 4)
- **Multiclass Classification**: Predicting exact star ratings (1-5) for product reviews
- **Clustering**: Grouping reviews into clusters based on text content and metadata

## Dataset

The project uses product review data with the following features:
- `reviewText`: The actual review text
- `overall`: Star rating (1-5)
- `category`: Product category
- Additional metadata like `verified`, `vote`, `unixReviewTime`

## Models Implemented

### Classification Models
- **Logistic Regression**: With different solvers and regularization parameters
- **Linear SVM**: Support Vector Machine with various kernels
- **Naive Bayes**: Multinomial Naive Bayes for text classification
- **Random Forest**: For multiclass classification

### Clustering
- **K-Means**: With dimensionality reduction using TruncatedSVD
- **Silhouette Analysis**: For optimal cluster number selection

## Features

- TF-IDF vectorization for text preprocessing
- Cross-validation with stratified sampling
- Hyperparameter tuning using GridSearchCV
- Comprehensive evaluation metrics (accuracy, F1-score, ROC curves)
- Confusion matrix visualization
- Multiclass ROC curve analysis

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook combinednb.ipynb
   ```

## File Structure

```
├── combinednb.ipynb          # Main analysis notebook
├── Training.csv              # Training dataset
├── Test.csv                  # Test dataset
├── Test.xlsx                 # Alternative test data format
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Results

The notebook generates several output files:
- `binary_predictions_cutoff_1.csv` - Binary predictions for cutoff=1
- `binary_predictions_cutoff_2.csv` - Binary predictions for cutoff=2
- `binary_predictions_cutoff_3.csv` - Binary predictions for cutoff=3
- `binary_predictions_cutoff_4.csv` - Binary predictions for cutoff=4
- `multiclass_predictions_1_to_5.csv` - Multiclass predictions

## Key Findings

- **Binary Classification**: Best performance achieved with Logistic Regression and Linear SVM
- **Multiclass Classification**: Challenging task with moderate accuracy due to class imbalance
- **Clustering**: Optimal number of clusters found to be 3 based on silhouette analysis

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm
- jupyter

## Author

Max Fortner - CS74 Final Project

## License

This project is for educational purposes as part of CS74 coursework.
