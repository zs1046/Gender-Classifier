from sklearn import tree #import decision tree from sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import numpy as np

# [height, weight, shoe size]
# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Classifier
# using the default value for the hyperparameter
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_KNN = KNeighborsClassifier()
clf_Gaussian = GaussianNB()


#Training the model
clf_tree.fit(X,Y)
clf_svm.fit(X,Y)
clf_KNN.fit(X,Y)
clf_Gaussian.fit(X,Y)


# Testing using the same data
treePrediction = clf_tree.predict(X)
accuracy_tree = accuracy_score(Y, treePrediction) * 100
print("DecisionTree accuracy was {}".format(accuracy_tree))

svmPrediction = clf_svm.predict(X)
accuracy_svm = accuracy_score(Y, svmPrediction) * 100
print("DecisionTree accuracy was {}".format(accuracy_svm))

knnPrediction = clf_KNN.predict(X)
accuracy_KNN = accuracy_score(Y, knnPrediction) * 100
print("DecisionTree accuracy was {}".format(accuracy_KNN))

gaussianPrediction = clf_Gaussian.predict(X)
accuracy_Gaussian = accuracy_score(Y, gaussianPrediction) * 100
print("DecisionTree accuracy was {}".format(accuracy_Gaussian))

#Print the best classifier
index = np.argmax([accuracy_svm, accuracy_tree, accuracy_KNN])
classifiers = {0: 'SVM', 1: 'KNN', 2: "Decision Tree", 3: "Naive Bayes"}
print ('Best gender classifier is {}'.format(classifiers[index]))