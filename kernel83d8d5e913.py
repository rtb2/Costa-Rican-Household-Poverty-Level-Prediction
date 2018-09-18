# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing data set
test_set = pd.read_csv('../input/test.csv')
train_set = pd.read_csv('../input/train.csv')

#Data Cleaning

#removing unnecessary columns
train_set = train_set.drop('Id', axis=1)
train_set.head()

#Dealing with Missing Values in v2a1
def median_imputation(a,b):
    train_set.loc[(train_set[a] == 1) & (train_set[b] == 1), 
              ['v2a1']] = train_set.loc[(train_set[a] == 1) & 
              (train_set[b] == 1),['v2a1']].fillna(train_set.loc[(train_set[a] == 1) & 
               (train_set[b] == 1), ["v2a1"]].median())


for i in ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6']:
    median_imputation(i,'area1')
    median_imputation(i,'area2')
    
train_set.columns[train_set.isna().any()]

#Dealing with Missing Values in v18q1 :number of tablets household owns
train_set['v18q1'].isna().sum()
              
from sklearn.preprocessing import Imputer

imputer_v18q1 = Imputer(strategy='most_frequent')
train_set.loc[(train_set["area1"]==1),
              ['v18q1']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area1"]==1),
              ['v18q1']])
train_set.loc[(train_set["area2"]==1),
              ['v18q1']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area2"]==1),
              ['v18q1']])

#Dealing with Missing Values in rez_esc :Years behind in school

train_set['rez_esc'].describe()

#using Median Imputation
imputer_rez_esc = Imputer(strategy='median')
train_set.loc[(train_set["area1"]==1),
              ['rez_esc']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area1"]==1),
              ['rez_esc']])
train_set.loc[(train_set["area2"]==1),
              ['rez_esc']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area2"]==1),
              ['rez_esc']])
              
train_set.columns[train_set.isna().any()]

#Dealing with Missing Values in meaneduc

train_set['meaneduc'].describe()
train_set['meaneduc'].isna().sum()

train_set.loc[train_set['meaneduc'].isna(), ['idhogar','meaneduc']]

imputer_meaneduc = Imputer(strategy='median')
train_set.loc[(train_set["area1"]==1),
              ['meaneduc']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area1"]==1),
              ['meaneduc']])
train_set.loc[(train_set["area2"]==1),
              ['meaneduc']] = imputer_v18q1.fit_transform(train_set.loc[(train_set["area2"]==1),
              ['meaneduc']])
              
#Removing squared values as the original values are available
to_drop  = ['SQBescolari','SQBage','SQBhogar_total','SQBedjefe', 'SQBhogar_nin','SQBovercrowding',
            'SQBdependency','SQBmeaned','agesq']
train_set = train_set.drop(to_drop, axis=1)

#Dealing with object datatypes
train_set.select_dtypes(include=['object']).columns

train_set = train_set.drop('idhogar', axis=1)

#Dealing with dependency: dependency = (hogar_nin + hogar_mayor)/hogar_adul

sum1 = train_set["hogar_nin"] + train_set["hogar_mayor"]

train_set["dependency"] = sum1/train_set["hogar_adul"]

#Dealing with 'edjefa', 'edjefe'
#dropping edjefa and edjefe as the escolari(years of schooling) and parentesco1(household head)
#variaables are already present

train_set = train_set.drop(['edjefa', 'edjefe'], axis=1)

#test set cleaning
#removing unnecessary columns
test_set = test_set.drop('Id', axis=1)
test_set.head()

#Dealing with Missing Values in v2a1
def median_imputation(a,b):
    test_set.loc[(test_set[a] == 1) & (test_set[b] == 1), 
              ['v2a1']] = test_set.loc[(test_set[a] == 1) & 
              (test_set[b] == 1),['v2a1']].fillna(test_set.loc[(test_set[a] == 1) & 
               (test_set[b] == 1), ["v2a1"]].median())


for i in ['lugar1','lugar2','lugar3','lugar4','lugar5','lugar6']:
    median_imputation(i,'area1')
    median_imputation(i,'area2')
    
test_set.columns[test_set.isna().any()]

#Dealing with Missing Values in v18q1 :number of tablets household owns
test_set['v18q1'].isna().sum()
              
from sklearn.preprocessing import Imputer

imputer_v18q1 = Imputer(strategy='most_frequent')
test_set.loc[(test_set["area1"]==1),
              ['v18q1']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area1"]==1),
              ['v18q1']])
test_set.loc[(test_set["area2"]==1),
              ['v18q1']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area2"]==1),
              ['v18q1']])

#Dealing with Missing Values in rez_esc :Years behind in school

test_set['rez_esc'].describe()

#using Median Imputation
imputer_rez_esc = Imputer(strategy='median')
test_set.loc[(test_set["area1"]==1),
              ['rez_esc']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area1"]==1),
              ['rez_esc']])
test_set.loc[(test_set["area2"]==1),
              ['rez_esc']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area2"]==1),
              ['rez_esc']])
              
test_set.columns[test_set.isna().any()]

#Dealing with Missing Values in meaneduc

test_set['meaneduc'].describe()
test_set['meaneduc'].isna().sum()

test_set.loc[test_set['meaneduc'].isna(), ['idhogar','meaneduc']]

imputer_meaneduc = Imputer(strategy='median')
test_set.loc[(test_set["area1"]==1),
              ['meaneduc']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area1"]==1),
              ['meaneduc']])
test_set.loc[(test_set["area2"]==1),
              ['meaneduc']] = imputer_v18q1.fit_transform(test_set.loc[(test_set["area2"]==1),
              ['meaneduc']])
              
#Removing squared values as the original values are available
to_drop  = ['SQBescolari','SQBage','SQBhogar_total','SQBedjefe', 'SQBhogar_nin','SQBovercrowding',
            'SQBdependency','SQBmeaned','agesq']
test_set = test_set.drop(to_drop, axis=1)

#Dealing with object datatypes
test_set.select_dtypes(include=['object']).columns

test_set = test_set.drop('idhogar', axis=1)

#Dealing with dependency: dependency = (hogar_nin + hogar_mayor)/hogar_adul

sum1 = test_set["hogar_nin"] + test_set["hogar_mayor"]

test_set["dependency"] = sum1/test_set["hogar_adul"]

#Dealing with 'edjefa', 'edjefe'
#dropping edjefa and edjefe as the escolari(years of schooling) and parentesco1(household head)
#variaables are already present

test_set = test_set.drop(['edjefa', 'edjefe'], axis=1)


train_set.loc[np.isinf(train_set["dependency"]),"dependency"]
test_set.loc[np.isinf(test_set["dependency"]),"dependency"]

train_set[["dependency"]] = train_set[["dependency"]].replace(np.inf,0)
test_set[["dependency"]] = test_set[["dependency"]].replace(np.inf,0)

#seperating dependent and independent variables
X_train = train_set.iloc[:,:-1].values
y_train = train_set.iloc[:,[-1]].values
X_test = test_set.iloc[:,:].values  

#One hot encoding the dependent variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Now let's make ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Initializing the ANN
classifier = Sequential()

#Adding inpot layer and first hidden layer
classifier.add(Dense(units=67, kernel_initializer= 'uniform', activation= 'relu', input_dim = 129))

#Adding Second Hidden Layer
classifier.add(Dense(units=67, kernel_initializer= 'uniform', activation= 'relu'))

#Adding Output Layer
classifier.add(Dense(units=4, kernel_initializer= 'uniform', activation= 'softmax'))

#Compiling ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"]) 

#Fitting the ANN to the Traing Set
classifier.fit(X_train, y_train, batch_size=25, epochs=250)

#Making Predictions and evaluating models

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Converting probabilities to Booleans
y_pred = (y_pred>0.5)

#Reversing Onehot encoding
y_pred = pd.DataFrame(y_pred).idxmax(axis=1)
#As the poverty levels are reported as 1,2,3,4 adding 1 to prediction will achive that
y_pred = y_pred+1
y_pred = pd.DataFrame(y_pred)
#Converting y_pred to desired sample submission format
test_set1 = pd.read_csv('../input/test.csv')
y_pred.loc[:,'Id'] = pd.Series(test_set1["Id"], index=y_pred.index)

y_pred.columns = ["Target", "Id"]

y_pred = y_pred[["Id", "Target"]]

y_pred.to_csv("sample_submission.csv", index=False)