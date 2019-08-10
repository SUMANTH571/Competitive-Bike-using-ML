import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# FEATURE RANKING 1


data1 = pd.read_csv('mobike.csv')
X1 = data1.iloc[:,0:11]  #independent columns
y1 = data1.iloc[:,11]   #target column i.e count
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X1,y1)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
print("\n\n\nThe Effect of each independent feature will be as shown")
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#  FEATURE RANKING 2  

data2 = pd.read_csv('ofo.csv')
X2 = data2.iloc[:,0:11]  #independent columns
y2 = data2.iloc[:,11]   #target column i.e count

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X2,y2)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
print("\n\n\nThe Effect of each independent feature will be as shown")
feat_importances = pd.Series(model.feature_importances_, index=X2.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# RANDOM FOREST REGRESSION

#first

data1 = pd.read_csv('mobike.csv')
X1 = data1.iloc[:,[2,4,6,7,8,9,10]].values  #independent columns
y1 = data1.iloc[:,11].values
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state = 0)
regressor.fit(X1, y1)
y1_pred = regressor.predict(X1[[5555]])
print('Predicted value is')
print(y1_pred)
print('Actual value is')
print(y1[[5555]])

#second

data2 = pd.read_csv('ofo.csv')
X2 = data2.iloc[:,[2,4,6,7,8,9,10]].values  #independent columns
y2 = data2.iloc[:,11].values

from sklearn.cross_validation import train_test_split
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state = 0)
regressor.fit(X2, y2)
y2_pred = regressor.predict(X2[[5555]])
print('Predicted value is')
print(y2_pred)
print('Actual value is')
print(y2[[4545]])






















"""
ax = plt.gca()

data1.plot(kind='line',x='season',y='y2',ax=ax)
data2.plot(kind='line',x='season',y='y1', color='red', ax=ax)

plt.show()
"""