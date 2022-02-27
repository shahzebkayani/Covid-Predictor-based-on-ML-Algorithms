
# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import plot_confusion_matrix

print('Importing the dataset')
# Importing the dataset
dataset = pd.read_csv('coroml.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('Filling missing data')
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, :-1])
X[:, :-1] = imputer.transform(X[:, :-1])

print('Splitting the dataset into 80% train and 20% test')
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


print('Applying Feature Scaling on the dataset')
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





"""
print('Applying Logistic regression and training the model')
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""

# Predicting the Test set results
y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.8)

print('Making Confusion Matrix')
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(classifier, X_test, y_test)  

print('Finding accuracy of the model')
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    
    sum_of_all_elements = confusion_matrix.sum()
    print(sum_of_all_elements)
    return diagonal_sum / sum_of_all_elements*100

d=accuracy(cm)


print('Generating new random data and testing the model')
Xnew, _ = make_blobs(n_samples=1, centers=9, n_features=16, random_state=2)
# make a prediction
ynew = classifier.predict(Xnew)


# show the inputs and predicted probabilities
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
if(ynew==0):
        print('No, This Person does not seem to have COVID-19')
elif(ynew==1):
        print('There are chances that this person has COVID, immediate isolation and testing advised')
print('Percentage accuracy of the proposed model')
print(d)
