# LAB-Introduction-to-ML-Pipeline

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)

# Boston Housing Price Prediction

## Introduction

In this project, we will walk through a complete Machine Learning workflow using the Boston Housing dataset. We will cover data collection, preprocessing, feature engineering, model training, evaluation, and deployment. The goal is to build a linear regression model to predict the median value of owner-occupied homes (MEDV) in Boston suburbs. We will also create a FastAPI application to serve our model and make predictions based on input features.

## Setup Instructions

To run the notebook and reproduce the results, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/codeskills-ai/LAB-Introduction-to-ML-Pipeline.git
    cd LAB-Introduction-to-ML-Pipeline
    ```

2. **Create a virtual environment and onstall the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

## Steps in the Workflow

### Step 1: Importing Libraries
In this step, you will import all the necessary libraries for data manipulation, machine learning, and API development. Suggested libraries include Pandas, Scikit-learn, Pickle, Numpy, and Matplotlib.

### Step 2: Data Collection
Download the Boston Housing dataset from the specified URL and load it into a Pandas DataFrame. Use appropriate column names for the dataset.

### Step 3: Basic EDA (Exploratory Data Analysis)
Conduct basic EDA to understand the data distribution and relationships between features.

#### Histograms
Visualize the distribution of key features using histograms.

#### Scatter Plots
Visualize relationships between features and the target variable using scatter plots.

### Step 4: Data Preprocessing
Handle any missing values in the dataset by filling them with the mean of their respective columns.

### Step 5: Feature Engineering
Create a new feature called 'PRICE_PER_ROOM' to capture the relationship between the price of the house and the number of rooms. Separate the features (X) from the target variable (y). Normalize the training data using MinMaxScaler.

### Step 6: Splitting the Data
Split the data into training and testing sets using an 80-20 split to evaluate the model's performance on unseen data.

### Step 7: Model Training
Train a Linear Regression model on the training data using Scikit-learn's LinearRegression.

### Step 8: Model Evaluation and Cross-validation
Evaluate the trained model on the test set using MAE, MSE, and R² score. Use 5-fold cross-validation to evaluate the model's performance.

### Step 9: Serialization
Serialize the trained model and scaler using Pickle and save them to files named `linear_regression_model.pkl` and `scaler.pkl`, respectively.

### Step 10: Creating an API using FastAPI
Create a FastAPI application to serve the model and make predictions based on input features. Implement a `/predict` endpoint that accepts a POST request with input features and returns the predicted house price.

### Step 11: Test the API
Test the API within the Jupyter notebook using the `requests` library by sending different sets of input features and observing the model's responses. This approach allows you to test the importance and influence of each feature on the predicted house prices, helping to understand how different input variables affect the model's output.

### Additional Steps
- You are free to experiment with different techniques and approaches throughout this project.
- Experiment with the input values and try to explore which are the most relevant features for the model.
- Experiment with Different Models: Compare the performance of different machine learning models like Decision Tree and Random Forest.
