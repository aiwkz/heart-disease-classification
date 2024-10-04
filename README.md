## Predicting Heart Disease using Machine Learning

### Overview

This project explores the use of various machine learning algorithms to predict whether or not a patient has heart disease based on clinical attributes. The dataset used is from the Cleveland Heart Disease data available via the UCI Machine Learning Repository, which is also hosted on Kaggle.

The main objective of this project is to apply data science and machine learning techniques to build a model capable of predicting heart disease with high accuracy.

---

### Table of Contents

1. [Problem Definition](#problem-definition)
2. [Data](#data)
3. [Features](#features)
4. [Project Workflow](#project-workflow)
5. [Models Used](#models-used)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Acknowledgements](#acknowledgements)

---

### Problem Definition

Given clinical parameters about a patient, the goal is to predict whether or not they have heart disease.

---

### Data

The dataset used in this project is the **Cleveland Heart Disease Dataset** obtained from the UCI Machine Learning Repository. This version is available on Kaggle [here](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

-   **Number of observations**: 303
-   **Number of features**: 14 clinical features (e.g., age, chest pain type, blood pressure)

---

### Features

The key features in the dataset include:

1. **age**: Age of the patient
2. **sex**: Sex of the patient (1 = male, 0 = female)
3. **cp**: Chest pain type (0 to 3)
4. **trestbps**: Resting blood pressure
5. **chol**: Serum cholesterol in mg/dl
6. **thalach**: Maximum heart rate achieved
7. **target**: Whether the patient has heart disease (1 = yes, 0 = no)

---

### Project Workflow

The project follows a standard machine learning workflow:

1. **Problem Definition**: Formulate the prediction task.
2. **Data Exploration**: Perform exploratory data analysis (EDA) to understand the dataset and relationships between variables.
3. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale features as necessary.
4. **Modeling**: Build different machine learning models to predict heart disease:
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest Classifier
5. **Hyperparameter Tuning**: Optimize model performance using techniques like RandomizedSearchCV and GridSearchCV.
6. **Evaluation**: Evaluate model performance using metrics like accuracy, precision, recall, and F1 score.

---

### Models Used

-   **Logistic Regression**: A baseline linear model for binary classification.
-   **K-Nearest Neighbors (KNN)**: A distance-based algorithm.
-   **Random Forest Classifier**: An ensemble method that builds multiple decision trees.

---

### Evaluation Metrics

For evaluating the models, the following metrics were used:

-   **Accuracy**: The overall correctness of predictions.
-   **Precision**: How many of the predicted positive cases were actually positive.
-   **Recall**: How many of the actual positive cases were correctly predicted.
-   **F1 Score**: The harmonic mean of precision and recall.
-   **ROC Curve**: A graphical representation of model performance across various thresholds.
-   **Confusion Matrix**: A matrix showing true positive, false positive, true negative, and false negative counts.

---

### Results

After comparing the models, **Logistic Regression** with hyperparameter tuning provided the best performance with an accuracy score of X%. The following evaluation metrics were used:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | X%    |
| Precision | Y%    |
| Recall    | Z%    |
| F1 Score  | W%    |

---

### Future Work

There are several avenues for future work in this project:

-   Implement more advanced models, such as **XGBoost** or **Gradient Boosting**.
-   Explore feature engineering techniques to improve model performance.
-   Experiment with deep learning models using **TensorFlow** for binary classification.
-   Model deployment to a web service or application for real-time prediction.

---

### Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/aiwkz/heart-disease-classification.git
    cd heart-disease-classification
    ```

2. **Install the required libraries**:
   You can use the `requirements.txt` file to install the necessary Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Launch Jupyter Notebook**:
   Run the following command to start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. **Run the Notebook**:
   Open the `end-to-end-heart-disease-classification.ipynb` notebook in Jupyter and run the cells to reproduce the analysis.

---

### Usage

To use the project, you can interact with the data, models, and visualizations directly from the Jupyter Notebook. The notebook is well-documented, and explanations for each code block are provided.

---

### Acknowledgements

-   Dataset from the **Cleveland Heart Disease** dataset via the UCI Machine Learning Repository and Kaggle.
-   Scikit-learn for machine learning tools and libraries.
-   Matplotlib and Seaborn for visualization support.
