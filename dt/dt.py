import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv('../diabetes_binary_health_indicators_BRFSS2015.csv')

X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

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
actual_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], title="Actual Labels")
plt.ylabel('')

plt.subplot(1, 2, 2)
predicted_counts.plot(kind='pie', autopct='%1.1f%%', colors=['orange', 'purple'], title="Predicted Labels")
plt.ylabel('')

plt.tight_layout()
plt.show()

