import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('../diabetes_binary_health_indicators_BRFSS2015.csv')

data['Diabetes_binary'] = data['Diabetes_binary'].astype(int)

X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, *, value=None
    ):
        """
        A node in the decision tree.

        Parameters:
        - feature_index: Index of the feature to split on
        - threshold: Threshold value to split at
        - left: Left child node
        - right: Right child node
        - value: Class label if it's a leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifierFromScratch:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Decision Tree Classifier.

        Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum number of samples required to split an internal node
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        y = y.astype(int)  # Ensure y is integer
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or num_labels == 1
            or num_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feature_idx, threshold = self._best_split(X, y, num_samples, num_features)

        # Grow the children that result from the split
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(
            feature_index=feature_idx, threshold=threshold, left=left, right=right
        )

    def _best_split(self, X, y, num_samples, num_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in range(num_features):
            feature_values = X[:, feature_idx]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                gain = self._information_gain(y, feature_values, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, feature_values, threshold):
        parent_entropy = self._entropy(y)

        # Generate split
        left_indices = feature_values <= threshold
        right_indices = feature_values > threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        # Compute weighted average entropy of children
        e_left = self._entropy(y[left_indices])
        e_right = self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain
        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        y = y.astype(int) 
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        y = y.astype(int)
        counter = np.bincount(y)
        most_common = np.argmax(counter)
        return most_common

    def _predict(self, inputs, node):
        if node.value is not None:
            return node.value
        feature_value = inputs[node.feature_index]
        if feature_value <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

clf = DecisionTreeClassifierFromScratch(max_depth=10)
clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Chart: Distribution of Actual Labels
actual_counts = y_test.value_counts(normalize=True) * 100
plt.figure(figsize=(6, 4))
actual_counts.plot(kind='bar', color=['blue', 'green'])
plt.title("Distribution of Actual Diabetes Binary Labels")
plt.xlabel("Class (0 = No Diabetes, 1 = Prediabetes/Diabetes)")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.show()

# Chart: Distribution of Predicted Labels
predicted_counts = pd.Series(y_pred).value_counts(normalize=True) * 100
plt.figure(figsize=(6, 4))
predicted_counts.plot(kind='bar', color=['orange', 'purple'])
plt.title("Distribution of Predicted Diabetes Binary Labels")
plt.xlabel("Class (0 = No Diabetes, 1 = Prediabetes/Diabetes)")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.show()

# Pie Chart: Comparison of Actual and Predicted Labels
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
actual_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue', 'lightgreen'],
    title="Actual Labels",
)
plt.ylabel('')

plt.subplot(1, 2, 2)
predicted_counts.plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['orange', 'purple'],
    title="Predicted Labels",
)
plt.ylabel('')

plt.tight_layout()
plt.show()

