import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

"""
Survivor: 0 = No; 1 = Yes
Class: 1 = First class; 2 = Second class; 3 = Third class
Gender: 0 = Man; 1 = Woman
Age: Age in years
SiblingsEspouses: Number of siblings or spouses aboard the Titanic, for the passenger in question.
ParentsChildren: Number of parents or children aboard the Titanic, for the passenger in question.
"""

# Read csv file
df = pd.read_csv('DataSet_Titanic.csv')

# Save the predictor attributes (all labels except 'Survivor') in variable X
X = df.drop('Survivor', axis=1)

# Save the label to be predicted ('Survivor') in y
y = df.Survivor

# Create a tree object
tree = DecisionTreeClassifier(max_depth=2, random_state=42)

# Fit the machine
tree.fit(X, y)

# Make predictions on the dataset
pred_y = tree.predict(X)

# Compare with the actual labels
print('Accuracy: ', accuracy_score(pred_y, y))

# Create a confusion matrix (not normalized)
# cm = confusion_matrix(y, pred_y)

# Create a normalized confusion matrix
cm = confusion_matrix(y, pred_y, normalize='true')

# Create a plot for the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)

# Create a plot for the normalized confusion matrix
disp.plot(cmap=plt.cm.Blues, values_format='.2f')
plt.show()

# Visualize a tree graphically
# Export the decision tree to a .dot file.
dot_data = export_graphviz(tree, out_file=None,
                     feature_names=X.columns.values,
                     class_names=['No', 'Yes'],
                     filled=True, rounded=True,
                     special_characters=True)

# Visualize the decision tree
graph = graphviz.Source(dot_data)
graph.view()

# Plot the importance of each variable in the obtained prediction on a bar graph
# Create the variables x (importances) and y (columns)
importances = tree.feature_importances_
columns = X.columns

# Create the graph
sns.barplot(x=columns, y=importances)
plt.title('Importance of each attribute')
plt.show()
