#!bin/usr/python
# Shelby Luttrell
#hw4 

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, preprocessing

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# loading training data
df = pd.read_csv('iris_data.csv', header=None, names=names)
df.head()

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4]) 	# end index is exclusive
y = np.array(df['species']) 	

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
prediction = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, prediction))

# creating odd list of K for KNN
listofk = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, listofk))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
misClassError = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[misClassError.index(min(misClassError))]
knn_accuracy = cv_scores[optimal_k]

print ("The optimal number of neighbors is ",optimal_k," with an accuracy of ",knn_accuracy)

# plot misclassification error vs k
plt.plot(neighbors, misClassError)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

#features_train, labels_train, features_test, labels_test = makeTerrainData()
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

yhat_test = clf.predict(X_test)

# Compute accuracy based on test samples
dt_acc = accuracy_score(y_test, yhat_test)
print(dt_acc)

#print out the accuracy for each of the methods used
methods = ('Decision Tree','KNN' )
y_pos = np.arange(len(methods))
performance = [dt_acc,knn_accuracy]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, methods)
plt.ylabel('Score')
plt.xlabel('DT/KNN')
plt.title('Compare Decision Tree and kNN Average Accuracy')
plt.show()