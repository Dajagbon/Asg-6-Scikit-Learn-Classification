# Breast Cancer Classification Using Machine Learning

## Project Overview

This project aims to utilize machine learning techniques to classify breast cancer using the built-in breast cancer dataset from Scikit Learn. The primary objectives of this project are to:

1. Clean and preprocess the dataset.
2. Build and train multiple machine learning classification models.
3. Evaluate the performance of each model.
4. Visualize the results to gain insights into model performance.

## Dataset

The dataset used in this project is the built-in breast cancer dataset from Scikit Learn. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe the characteristics of the cell nuclei present in the image.

### Columns in the Dataset

- `mean radius`
- `mean texture`
- `mean perimeter`
- `mean area`
- `mean smoothness`
- ... (additional features)
- `target`: Binary classification (0 for malignant, 1 for benign)

## Data Cleaning and Preprocessing

The data cleaning and preprocessing steps are performed in the `Scikit Learn Classification.py` script. The main steps include:

1. **Load Dataset**: Load the built-in breast cancer dataset from Scikit Learn.
2. **Split Dataset**: Split the dataset into training and testing sets using the standard data division process.

### Excerpt from `Scikit Learn Classification.py`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets using the standard data division process
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
