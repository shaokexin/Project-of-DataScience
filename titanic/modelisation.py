from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #set background of seaborn is white.
sns.set(style="whitegrid", color_codes=True)



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
    
   logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
   fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
   plt.figure()
   plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
   plt.plot([0, 1], [0, 1],'r--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver operating characteristic')
   plt.legend(loc="lower right")
   plt.savefig('Log_ROC')
   plt.show()
   return logreg.predict
    

    
    


def logmodel_prediction_test(X, y,Xtest):
# Divide X and y into two parts
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
   # Check the performance of the logistic regression model
# 1. training model,  
# 2. According to the model, use X_test as input to generate the variable y_pred.
   logreg = LogisticRegression(max_iter=1000)
   logreg.fit(X_train, y_train.values.reshape(-1))
   y_pred = logreg.predict(X_test)
# print('Predictive accuracy on top of the test data set: {:.2f}'.format(logreg.score(X_test, y_test))
   print('predict Survived for titanic.test data:') 
   print(logreg.predict(Xtest))   
    



    
    
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
    return
