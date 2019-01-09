from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np

clf_tree = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1 Gaussian Naive Bayes Classifier
# 2 Support Vector Machines Classifier
# 3 K Neighbours Classifier

clf_nb = naive_bayes.GaussianNB()
clf_svm = svm.SVC(gamma='scale')
clf_Kneighbor = neighbors.KNeighborsClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_nb = clf_nb.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)
clf_Kneighbor = clf_Kneighbor.fit(X, Y)

pred_tree = clf_tree.predict([[190, 70, 43]])
pred_nb = clf_nb.predict([[190, 70, 43]])
pred_svm = clf_svm.predict([[190, 70, 43]])
pred_Kneighbor = clf_Kneighbor.predict([[190, 70, 43]])

print("Decision Tree Classifier's result is {} ".format(pred_tree))
print("Gaussian Naive Bayes Classifier's result is {} ".format(pred_nb))
print("Support Vector Machines Classifier's result is {} ".format(pred_svm))
print("K Neighbours Classifier's result is {} ".format(pred_Kneighbor))
print()

# CHALLENGE compare their results and print the best one!

pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print("Decision Tree Classifier's accuracy is {} ".format(acc_tree))

pred_nb = clf_nb.predict(X)
acc_nb = accuracy_score(Y, pred_nb) * 100
print("Gaussian Naive Bayes Classifier's accuracy is {} ".format(acc_nb))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print("Support Vector Machines Classifier's accuracy is {} ".format(acc_svm))

pred_Kneighbor = clf_Kneighbor.predict(X)
acc_Kneighbor = accuracy_score(Y, pred_Kneighbor) * 100
print("K Neighbours Classifier's accuracy is {} ".format(acc_Kneighbor))

print()

index = np.argmax([acc_tree, acc_nb, acc_svm, acc_Kneighbor])
classifiers = {0: 'Decision Tree Classifier', 1: 'Gaussian Naive Bayes Classifier', 2: 'Support Vector Machines Classifier', 3: 'K Neighbours Classifier'}
print('Best gender classifier is {}'.format(classifiers[index]))
