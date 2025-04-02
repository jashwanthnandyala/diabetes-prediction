This project implements a Diabetes Prediction Model using Support Vector Machine (SVM) on the PIMA Diabetes Dataset.
It preprocesses data, trains an SVM classifier, and evaluates its accuracy. The model also includes a predictive system 
that takes user input and determines the likelihood of diabetes.Additionally, a Flask web application is integrated for
easy user interaction.

Features:

Data Preprocessing: Standardization of features using StandardScaler.
Train-Test Split: Splitting dataset for model evaluation.
Model Training: Uses SVM with a linear kernel.
Accuracy Evaluation: Assesses model performance on training and test data.
Predictive System: Accepts user input and provides a diabetes prediction.
Data Visualization: Correlation matrix visualization using Seaborn.

Tech Stack:

Python
Flask (Web Framework)
NumPy & Pandas (Data Handling)
Scikit-Learn (Machine Learning Model)
Seaborn & Matplotlib (Data Visualization)
Joblib (Model Serialization)

Future Enhancements:

Hyperparameter Tuning to optimize performance.
Deep Learning Integration for improved predictions.
Enhanced Web UI for a better user experience.
Database Integration to store user predictions.
