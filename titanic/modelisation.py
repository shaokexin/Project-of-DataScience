from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def logmodel_prediction(X, y):
# Divide X and y into two parts
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
   # Check the performance of the logistic regression model
# 1. training model,  
# 2. According to the model, use X_test as input to generate the variable y_pred.
   logreg = LogisticRegression(max_iter=1000)
   logreg.fit(X_train, y_train.values.reshape(-1))
   y_pred = logreg.predict(X_test)
# print('Predictive accuracy on top of the test data set: {:.2f}'.format(logreg.score(X_test, y_test))
   print('Train/Test split results:')
   print("Accuracy %2.3f" % accuracy_score(y_test, y_pred)) 
   
   print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def  randomforestmodel_prediction(X,y):
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)


    random_forest.score(X_train, y_train)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


    rf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
