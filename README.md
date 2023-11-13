# titanic-dataset-decision-tree-and-kMeans
My version of a Decision Tree and K means Clustering using Python

The code follows the following instructions:

1. Load the given titanic_train.csv file.
2. Remove all irrelevant columns.
3. Create new columns for
a. AgeGroup (NK if not given, Baby if less than 2, Child if less than 14, Youth if less than 24, Adult if less than or equal to 65 and Senior otherwise)
b. Relatives (based on SibSp + ParCh - None of 0, Few if less than 4, Many otherwise)
c. Fare (Free if 0, Low if less than 50, average if less than 100, high if greater than 100)
4. ScikitLearn’s decision tree may not accept categorical data, if so, apply one-hot encoding to convert the attributes to binary
5. Split data into train and test set
6. Fit model using Decision Trees in SciKitLearn Package for the train set.
7. Predict the survival for the test set.
8. Print accuracy and confusion matrix and plot the decision tree.
9. Perform clustering.
10. Plot elbow graph and find the best k.

Use the jupyter notebook to run the code
