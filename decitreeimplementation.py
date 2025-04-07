# 🔧 Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Step 2: Load and prepare the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Optional: Map target integers to actual class names for readability
target_names = dict(enumerate(iris.target_names))
y_named = y.map(target_names)

X.head()

# 📊 Step 3: Exploratory Data Analysis
sns.pairplot(pd.concat([X, y_named.rename("species")], axis=1), hue="species")
plt.suptitle("Feature Distribution by Species", y=1.02)
plt.show()

# 🔀 Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Step 5: Build the Decision Tree model
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# 📈 Step 6: Visualize the Decision Tree
plt.figure(figsize=(15,10))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()

# 📋 Step 7: Evaluate the Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

#✅You’ve Successfully:
#Loaded and visualized the Iris dataset
#Built and trained a Decision Tree model using scikit-learn
#Visualized the tree structure
#Evaluated the model’s performance
