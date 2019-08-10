# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:05:48 2019

@author: lenovo
"""
#importing lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Dataset importing(All.feat)
data2 = pd.read_csv('ofo.csv')
X2 = data2.iloc[:,0:11]  #independent columns
y2 = data2.iloc[:,11]  #target column i.e count

#Feature Ranking
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X2,y2)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
print("\n\n\nThe Effect of each independent feature will be as shown")
feat_importances = pd.Series(model.feature_importances_, index=X2.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#Dataset importing(eff.feat)
data2 = pd.read_csv('ofo.csv')
X2 = data2.iloc[:,[0,2,4,6,7,8,9,10]].values  #independent columns
y2 = data2.iloc[:,11].values

#spliting into test and train
from sklearn.cross_validation import train_test_split
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

#Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20,random_state = 0)
regressor.fit(X2, y2)
y2_pred= regressor.predict(X2_test)

#df to csv
prediction = pd.DataFrame(X2_test, columns=['season','hour','h','j','k','f','d','s']).to_csv('result.csv')
Rdata=pd.read_csv('result.csv')

# visualisation
plt.scatter(Rdata['hour'],y2_pred)
plt.xlabel('HOUR')
plt.ylabel('count')
plt.title('HOUR')
plt.show()


#Dataset importing(eff.feat)
data1 = pd.read_csv('mobike.csv')
X1= data1.iloc[:,[0,2,4,6,7,8,9,10]].values  #independent columns
y1 = data1.iloc[:,11].values

#spliting into test and train
from sklearn.cross_validation import train_test_split
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=0)

#Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20,random_state = 0)
regressor.fit(X1, y1)
y1_pred= regressor.predict(X1_test)

#df to csv
prediction = pd.DataFrame(X1_test, columns=['season','hour','h','j','k','f','d','s']).to_csv('result1.csv')
Rdata1=pd.read_csv('result1.csv')

# visualisation
plt.scatter(Rdata1['hour'],y1_pred,color='red')
plt.xlabel('HOUR')
plt.ylabel('count')
plt.title('HOUR')
plt.show()

plt.scatter(Rdata['season'],y1_pred)
plt.xlabel('season')
plt.ylabel('count')
plt.title('SEASON')
plt.show()

plt.scatter(Rdata1['season'],y2_pred)
plt.xlabel('season')
plt.ylabel('count')
plt.title('SEASON')
plt.show()

plt.bar(Rdata1['hour'],y2_pred)
plt.bar(Rdata['hour'],y1_pred,color='green')

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.bar(Rdata1['hour'],y2_pred)
ax2.bar(Rdata['hour'],y1_pred,color='green')

