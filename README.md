# Machine-Learning-Classifiers
I built a gender classifier using 4 classifiers in the sklearn library.

The classifiers are:
1. Decision Tree Classifier
2. Gaussian Naive Bayes Classifier
3. Support Vector Machines Classifier
4. K Neighbours Classifier

I compared the accuracy of the various classifers and found the classifier with the highest accuracy.

I learnt this from the following sources:
1. https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU - the code
2. https://scikit-learn.org/stable/tutorial/machine_learning_map/ - deciding which classifiers I can use
3. https://github.com/Naresh1318/GenderClassifier/blob/master/Run_Code.py - the winner of the challenge's code

Problems I faced:
1. I don't know when to use each classifier
2. I dont understand the math behind the classifiers
3. I'm not sure that my method of finding the accuracy is right. By repeating the execution a million times, Im testing the precision, not    the accuracy. If someone could find a way to test the accuracy, that would be great

Solutions:

3. Use the following code to find the accuracy.
   pred = clf.predict(X)
   acc = accuracy_score(Y, pred) * 100
   Here X is an array containing the dataset in the format [height, weight, shoe_size]
   clf is the classifier object
   predict(test_case_array) - returns an array of the predictions for each test case
   accuracy_score(real_value_array, predicted_value_array) - returns the accuracy score from 0 to 1
   
   
