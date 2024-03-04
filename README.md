# smsclassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score  

data = pd.read_csv("/Users/nareshreddy/Downloads/spam.csv",encoding='Latin-1')
data.head()

data = data.drop(['Unnamed: 2' ,'Unnamed: 3' , 'Unnamed: 4' ],axis=1)
data = data.rename(columns = {'v1':'label','v2':'message'})

data.describe()

data.groupby('label').describe()

data['length']=data['message'].apply(len)
data.head()

data.describe()

%matplotlib inline
data.hist(by='label',column='length',bins=30,figsize=[15,5])

data['label_num']=data.label.map({'ham':0,'spam':1})
data.head()

x = data.message
y = data.label_num

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=11)

vect = CountVectorizer()
X_train = vect.fit_transform(x_train)
X_test = vect.transform(x_test)

svc = SVC(kernel = 'linear')
mnb = MultinomialNB(alpha=0.2)
gnb = GaussianNB()
lr = LogisticRegression(solver='liblinear',penalty='l1')
rfc = RandomForestClassifier(n_estimators=100,random_state=11)
abc = AdaBoostClassifier(n_estimators=100,random_state=11)

def training(clf,x_train,Y_train):
    clf.fit(x_train,Y_train)
def predict(clf,X_test):
    return clf.predict(X_test)

classifier = {'SVM':svc , 'MultinomialNB':mnb , 'GaussianNB':gnb , 'logistic':lr , 'RandomForest':rfc , 'Adaboost':abc}

score = []
for n, c in classifier.items():
    training(c, X_train.toarray(), y_train)
    pred = predict(c, X_test.toarray())
    score.append((n, accuracy_score(y_test, pred, normalize=True)))
score_df = pd.DataFrame(score, columns=['Classifier', 'Accuracy'])
score_df['Accuracy(%)'] = score_df['Accuracy'] * 100
print(score_df)
