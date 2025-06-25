"""
Preprocessing and visualization of the Iris dataset using scikit-learn.
Author: [َََAbolfazl Karimi]
GitHub: [https://github.com/abolfazlkarimi83]
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data            # Features
Y = iris.target          # Labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, random_state=42
)

# Visualize the training data (Feature 0 vs Feature 3)
plt.figure(figsize=(6, 4))
plt.scatter(X_train[:, 0], X_train[:, 3], s=15, c=y_train, cmap='viridis')
plt.title("Iris Training Data: Feature 0 vs Feature 3")
plt.xlabel("Feature 0")
plt.ylabel("Feature 3")
plt.colorbar(label='Class')
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply Min-Max Scaling to the features (range 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the shapes of the processed data
print("Shape of X_train_scaled:", np.shape(X_train_scaled))
print("Shape of y_train:", np.shape(y_train))
