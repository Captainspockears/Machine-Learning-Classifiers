from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
import numpy as np

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1 Gaussian Naive Bayes Classifier
# 2 Support Vector Machines Classifier
# 3 K Neighbours Classifier

clf1 = naive_bayes.GaussianNB()
clf2 = svm.SVC(gamma='scale')
clf3 = neighbors.KNeighborsClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])

# CHALLENGE compare their results and print the best one!

print(prediction)
print(prediction1)
print(prediction2)
print(prediction3)

y_pred = [None]*100000
y_true = [None]*100000
for i in range(0, 100000):
    y_pred[i] = clf.predict([[190, 70, 43]])
    y_true[i] = 'male'


accuracy = metrics.accuracy_score(y_true, y_pred)

print("Accuracy of Decision Tree Classifier: ", accuracy)

y_pred1 = [None]*100000
y_true1 = [None]*100000
for i in range(0, 100000):
    y_pred1[i] = clf1.predict([[190, 70, 43]])
    y_true1[i] = 'male'


accuracy1 = metrics.accuracy_score(y_true1, y_pred1)

print("Accuracy of Gaussian Naive Bayes Classifier: ", accuracy1)

y_pred2 = [None]*100000
y_true2 = [None]*100000
for i in range(0, 100000):
    y_pred2[i] = clf2.predict([[190, 70, 43]])
    y_true2[i] = 'male'


accuracy2 = metrics.accuracy_score(y_true2, y_pred2)

print("Accuracy of Support Vector Machines Classifier: ", accuracy2)

y_pred3 = [None]*100000
y_true3 = [None]*100000
for i in range(0, 100000):
    y_pred3[i] = clf3.predict([[190, 70, 43]])
    y_true3[i] = 'male'


accuracy3 = metrics.accuracy_score(y_true3, y_pred3)

print("Accuracy of K Neighbours Classifier: ", accuracy3)

