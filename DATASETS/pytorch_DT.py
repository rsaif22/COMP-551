import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define a simple label encoder function
def encode_column(column):
    if column.dtype == object:
        column = column.astype('category').cat.codes
    return column

def build_test_DT(datafile, output_column_name, num_classes, num_features, seed=None, exclude_features=[]):
    # Set the random seed
    np.random.seed(seed)

    # Read the CSV file with headers
    df = pd.read_csv(datafile)

    # Remove unimportant features
    df = df.drop(columns=exclude_features)

    # Encode categorical columns
    df = df.apply(encode_column)

    # Extract features and target variable
    feature_columns = df.columns.difference([output_column_name])
    X = df[feature_columns].to_numpy()
    y = df[output_column_name]

    # Convert the target labels to integers
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=seed)

    # Build a decision tree and train it
    clf = tree.DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    # Test the model on the test set
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc