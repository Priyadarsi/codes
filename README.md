#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#read files in kaggle
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

#check the data
train.head()
train.describe()
train.info()

#dataframe operations
#check the shape of dataframe
df.shape() 

#sum of number of null values in each column
df.isna().sum()

#count the number of each type of value
df[‘Diagnosis’].value_counts()

#check the datatype of each column
df.dtypes()

#check the correlation
df.iloc[:,1:12].corr()

#visualize the correlation
plt.figure(figsize=(10,10)
sns.heatmap(df.iloc[:,1:12].corr())

#to get heatmap with annotation and / or with percentages
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt=’.0%’)

#info on unique values
df['col2'].unique()
df['col2'].nunique()
df['col2'].value_counts()
#select columns by datatype
train.select_dtypes('object').columns

#drop columns
df.drop('new',axis=1,inplace=True)

#select row
df.loc['A']

#select row based on position
df.iloc[2]

#select a value
df.loc[['A','B'],['W','Y']]

#conditional statement
df>0
df[df>0]
df[df['W']>0]
df[df['W']>0]['Y']
df[(df['W']>0) & (df['Y'] > 1)]

# Reset to default 0,1...n index
df.reset_index()

#set index, not inplace until mentioned
df.set_index('States',inplace=True)

#multi index
# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
df = pd.DataFrame(np.random.randn(6,2),index=hier_index,columns=['A','B'])

#fetch values
df.loc['G1']
df.loc['G1'].loc[1]
df.index.names = ['Group','Num']

df.xs('G1')
df.xs('G1',1)
df.xs(1,level='Num')

#data visualization

#histplot
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

#distplot
sns.distplot(tips['total_bill'],kde=False,bins=30)

#jointplot

sns.jointplot(x='fare',y='age',data=titanic)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
#pairplot
sns.pairplot(tips,hue='sex',palette='coolwarm')
#barplot

sns.barplot(x='sex',y='total_bill',data=tips)

#countplot

sns.countplot(x='sex',data=tips)

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data = train,palette='rainbow')

sns.countplot(x='SibSp',data=train)

#boxplot

sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')
sns.boxplot(data=tips,palette='rainbow',orient='h')
sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")

#violinplot

sns.violinplot(x="day", y="total_bill", data=tips,palette='rainbow')
sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',palette='Set1')
sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')

#stripplot

sns.stripplot(x="day", y="total_bill", data=tips)
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True)
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1')
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True)

#swarmplot

sns.swarmplot(x="day", y="total_bill", data=tips)
sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True)

#heatmap and clutermap

sns.heatmap(tips.corr())
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
sns.heatmap(pvflights)

sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)

#clustermap
sns.clustermap(pvflights)
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)

#regression Plots

sns.lmplot(x='total_bill',y='tip',data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm')
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm')



#convert object to integer

sex1 = pd.get_dummies(test['Sex'],drop_first=True)
embark1 = pd.get_dummies(test['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace =True) 
train= pd.concat([train,sex,embark],axis=1)

------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)


labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
-------------------------------------------------------------
#Example of conversion for heterogeneous data 
train.select_dtypes('object').columns

df = train[['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType',
       'SaleCondition']]

from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)

train = train.drop(['Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType',
       'SaleCondition'], axis=1)

train= pd.concat([train,df],axis=1)


#convert float to integer

train['item_cnt_day'] = train['item_cnt_day'].apply(np.int64)

#handle null values

#check for null values 
train.isnull().values.any()

#fill null values with mean of column

df['A'].fillna(value=df['A'].mean())

#fill null values with zero

df['DataFrame Column'] = df['DataFrame Column'].fillna(0)

#check for null values on heatmap

sns.heatmap(train.isnull(),yticklabels=False,cbar=True,cmap='viridis')

#drop all rows with null values in a dataset

train = train.dropna(inplace = True)


#regression algorithms

#linear_regression

from sklearn.linear_model import LinearRegression
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Variance:',metrics.explained_variance_score(y_test, predictions))

#xgboost_regression

#X = train.drop('SalePrice',axis=1)
#y = train['SalePrice']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from catboost import CatBoostRegressor
# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=10,learning_rate=0.5,depth=2)
# Fit model
model.fit(X_train,y_train)
# Get predictions
preds = model.predict(X_test)

plt.scatter(y_test,preds)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, preds))
print('MSE:', metrics.mean_squared_error(y_test, preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
print('Variance:',metrics.explained_variance_score(y_test, preds))

#decision tree regression

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train,y_train)
regr_2.fit(X_train,y_train)

# Predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.scatter(y_test,y_1)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

print('MAE:', metrics.mean_absolute_error(y_test, preds))
print('MSE:', metrics.mean_squared_error(y_test, preds))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
print('Variance:',metrics.explained_variance_score(y_test, preds))



#classification algorithms

#logistic_regression

from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(A_train, b_train)
predictions = logmodel.predict(A_test)
from sklearn.metrics import classification_report
print(classification_report(b_test,predictions))

#knnclassifier

#standardise variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#predictions
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#choosing a k value
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


#plot the loop
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#use the new k value
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

#support vector machines
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#k means cluster

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
 kmeans.cluster_centers_


#decisiontree

X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state = 101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#random forest

X = train.drop('Survived',axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state = 101)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#xgboost

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
pd.crosstab(y_test,y_predict)

#catboost

from catboost import CatBoostClassifier
from catboost import Pool
X = train.drop('Survived',axis=1) 
y = train['Survived']
cat_features = [0, 1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30,random_state = 101)
train_pool = Pool(data=X_train,label=y_train,cat_features=cat_features)
validation_pool = Pool(data=X_test,label=y_test,cat_features=cat_features)
#keras

#import libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

#perform a train test split

C = train.drop('S',axis=1).values
d = train['S'].values

#normalise the data
from sklearn.model_selection import train_test_split

C_train, C_test, d_train, d_test = train_test_split(C,d,test_size=0.25, random_state = 101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(C_train)

C_train = scaler.transform(C_train)
C_test = scaler.transform(C_test)

#create the model

model = Sequential()
model.add(Dense(units=9,activation='relu'))
model.add(Dense(units=5,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = 'adam')
model.fit(C_train, d_train, epochs=600)

#create predictions

predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


#submissions file

X_test = test
predictions = dtree.predict(X_test)
submission = pd.DataFrame()
submission['PassengerId']= test['PassengerId']
submission['Survived'] = predictions
submission.head()
submission.to_csv('randomforest_submission.csv',index=False)
