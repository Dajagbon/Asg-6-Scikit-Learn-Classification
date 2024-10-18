# Breast Cancer Classification Using Machine Learning

## Project Overview

The logistic regression model performs best among the models in my script. It has the highest accuracy, precision, and recall percentage. The logistic regression has an accuracy score of 96%, which is excellent, although it’s not industry standards. High accuracy means the model correctly predicts a high percentage of the instances. The precision score is also great because high precision means fewer false positives. Finally, the recall score is 99%, which means fewer false negatives. 

This project aims to utilize machine learning techniques to classify breast cancer using the built-in breast cancer dataset from Scikit Learn. The primary objectives of this project are to:

1. Clean and preprocess the dataset.
2. Build and train multiple machine learning classification models.
3. Evaluate the performance of each model.
4. Visualize the results to gain insights into model performance.

## Dataset

The dataset used in this project is the Breast Cancer dataset from Scikit-learn. It includes features computed from digitized images of fine needle aspirate (FNA) of breast mass and a binary target variable indicating the presence or absence of breast cancer.

### Columns in the Dataset

- `mean radius`
- `mean texture`
- `mean perimeter`
- `mean area`
- `mean smoothness`
- ... (additional features)
- `target`: Binary classification (0 for malignant, 1 for benign)

## Class Design and Implementation

The project does not implement custom classes but makes extensive use of scikit-learn classes for machine learning models, preprocessing, and evaluation metrics.

## 1. Data Loading and Preprocessing 

load_breast_cancer(): Function to load the Breast Cancer dataset.

train_test_split: Function to split the data into training and testing sets.

StandardScaler: Used to normalize the dataset.

## 2. Model Initialization
The following scikit-learn classes are used to initialize the models:

LogisticRegression(max_iter=10000)

SVC()

RandomForestClassifier()

KNeighborsClassifier()

GradientBoostingClassifier()

## 3. Model Training
Each model is trained using the fit(X_train, y_train) method provided by scikit-learn.

## 4. Predictions
Predictions are made on the test set using the predict(X_test) method.

## 5. Evaluation Metrics
The following evaluation metrics are computed for each model:

accuracy_score: Measures the ratio of correctly predicted instances to the total instances.

precision_score: Measures the ratio of correctly predicted positive observations to the total predicted positives.

recall_score: Measures the ratio of correctly predicted positive observations to all observations in the actual class.

classification_report: Provides a detailed report of precision, recall, and F1-score for each class.

confusion_matrix: Summarizes the prediction results by showing the number of true positives, true negatives, false positives, and false negatives.

## 6. Visualization
The project uses seaborn and matplotlib for visualizations:

sns.heatmap: Used to plot the confusion matrix for each model.

## Limitations
Class Imbalance: The dataset may have an imbalance in classes which can affect model performance.

Hyperparameter Tuning: The models are used with default parameters, and no hyperparameter tuning is performed. Hyperparameter optimization could improve the models' performance.

Feature Engineering: The project uses raw features without additional feature engineering which might be required for better performance.

## How to Run
Ensure all necessary libraries are installed: pandas, numpy, scikit-learn, seaborn, matplotlib.

Load and run the script provided in this ReadMe.

Analyze the output metrics and confusion matrices to determine the best-performing model.
