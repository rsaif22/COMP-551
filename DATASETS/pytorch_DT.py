import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

# Define a simple label encoder function
def encode_column(column):
    if column.dtype == object:
        column = column.astype('category').cat.codes
    return column
#TODO: add unimportant features
def build_test_DT(datafile, output_column_name, num_classes, num_features):
    # Read the CSV file with headers
    df = pd.read_csv(datafile)

    # Encode categorical columns
    df = df.apply(encode_column)

    # Extract features and target variable
    feature_columns = df.columns.difference([output_column_name])
    X = df[feature_columns].to_numpy()
    y = df[output_column_name]

    # Convert the species labels to integers
    species_to_int = {species: idx for idx, species in enumerate(np.unique(y))}
    y_int = np.array([species_to_int[eval] for eval in y])

    # Optionally, convert the integer labels to one-hot encoding
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]
    
    num_classes = len(species_to_int)
    y_one_hot = one_hot_encode(y_int, num_classes)

    # Build a decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y_one_hot)
    acc = accuracy_score(y_one_hot, clf.predict(X))
    return acc