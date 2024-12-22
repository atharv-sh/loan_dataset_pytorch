# Financial Loan Status Prediction

This Jupyter Notebook demonstrates a deep learning model for predicting financial loan status. It utilizes a neural network trained on a dataset of loan applications, employing various data preprocessing and analysis techniques.

## Data

The model is trained on the `financial_loan.csv` dataset.  The dataset contains various features related to loan applications, including loan amount, interest rate, employment details, and borrower's financial history.  

## Preprocessing

The following preprocessing steps are performed:

1. **Handling Missing Values:** Missing values in the 'emp_title' column are filled with the mode (most frequent value).

2. **Exploratory Data Analysis (EDA):** Visualizations using `seaborn` and `matplotlib` are used to understand the distribution of various features like loan amount, loan status, grade, employment length, and home ownership.  A correlation heatmap is generated to visualize relationships between different features.

3. **Label Encoding:** Categorical features are converted into numerical representations using Label Encoding.

4. **Data Splitting and Scaling:** The data is split into training and testing sets. Numerical features are scaled using `StandardScaler` to ensure they have zero mean and unit variance.

5. **Data Conversion:** Data is converted into PyTorch tensors for model training.

## Model

A neural network with three fully connected layers is implemented using PyTorch. The model uses a sigmoid activation function in the output layer to predict the probability of loan status.

**Architecture:**

* Input Layer:  Input size determined by the number of features in the dataset.
* Hidden Layer 1: 64 neurons with ReLU activation.
* Hidden Layer 2: 32 neurons with ReLU activation
* Output Layer: 1 neuron with sigmoid activation (for binary classification).

**Training:**

* Loss Function: Binary Cross-Entropy Loss (`BCELoss`).
* Optimizer: Adam optimizer.
* Training is performed for a specified number of epochs.

**Evaluation:**

The model's performance is evaluated on the test set by calculating accuracy.

## Prediction

The notebook includes an example of how to use the trained model to make predictions on new data.  New data is preprocessed similarly to the training data and predictions are made by feeding the data through the loaded model.

## Requirements

Ensure that the necessary libraries are installed.
