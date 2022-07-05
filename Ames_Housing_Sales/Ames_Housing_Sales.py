import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df=pd.read_csv('E:/py/Data/machine learing/Ames_Housing_Sales.csv')
print(df.sample(1).iloc[0])
print(df.isnull().sum().all())
df=df.fillna('mean')
#print(df.info())
plt.figure(figsize=(10,5))
plt.hist(df['SalePrice'])
plt.title('Histogram of Sale Price', size = 16)
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.show()

df = df[df['SalePrice'] <= 450000]
featuers=df.drop(['SalePrice'],axis=1)
corlation=featuers.corrwith(df['SalePrice'])
plt.figure(figsize=(10,10))
corlation.plot(kind='barh',title='featear corrlation with sale price5')
plt.show()

plt.hist(df['SalePrice'])
plt.title('Histogram of Sale Price', size = 16)
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.show()

high_corlations=corlation[corlation >.2].index
highest_corlations=corlation[corlation>.5].index
print(df[high_corlations].info())

sns.regplot(x=df.GrLivArea[:100],y=df['SalePrice'][:100],data=df, ci=None,color='red').set_title('sale price vs GrLivArea')
plt.show()
df = df[df['GrLivArea'] <= 2500]

sns.regplot(x=df.GarageYrBlt[:100],y=df['SalePrice'][:100],data=df, ci=None,color='red').set_title('sale price vs GarageYrBlt')
plt.show()

sns.regplot(x=df.GarageArea[:200],y=df['SalePrice'][:200],data=df, ci=None,color='red').set_title('sale price vs GarageArea')
plt.show()

sns.regplot(x=df.FullBath[:200],y=df['SalePrice'][:200],data=df, ci=None,color='red').set_title('sale price vs FullBath')
plt.show()

sns.regplot(x=df.HalfBath,y=df['SalePrice'],data=df, ci=None,color='red').set_title('sale price vs HalfBath')
plt.show()
df = df[df['HalfBath'] <= 2]


#print(df.columns)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]


#droping low features corrlations with sales price
X=X[high_corlations]
#print(X.info())



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,shuffle=True)

ln=LinearRegression()
ln.fit(X_train,y_train)
pred2=ln.predict(X_test)
print(ln.score(X_test, y_test))

error1=[]
for k in range(2,15):
    forest = RandomForestRegressor(min_samples_split=k, n_estimators =400, random_state = 1)
    forest.fit(X_train,y_train)
    y_pred1= forest.predict(X_test)
    error1.append(np.mean(y_pred1))
plt.figure('test K')
plt.plot(range(2,15),error1,label="min_samples_split")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()

error1=[]
for k in range(102,501,100):
    forest = RandomForestRegressor(min_samples_split=2, n_estimators = k, random_state = 1)
    forest.fit(X_train,y_train)
    y_pred1= forest.predict(X_test)
    error1.append(np.mean(y_pred1))
plt.figure('test K')
plt.plot(range(102,501,100),error1,label="N of trees")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()

forest=RandomForestRegressor(n_estimators=400,min_samples_split=2,random_state =42)
forest.fit(X_train,y_train)
forest_pred=forest.predict(X_test)
print('forest score : ',forest.score(X_test,y_test),'\n')


mse=mean_squared_error(y_test,forest_pred)
print('mse :',mse)

plt.figure(figsize=(12,9))

# Generate a scatterplot of predicted values versus actual values.
plt.scatter(forest_pred, y_test, s=10, color='skyblue', alpha = 0.5)

# Plot a line.
plt.plot([np.min(forest_pred), np.max(forest_pred)],
         [np.min(forest_pred), np.max(forest_pred)],
         color = 'black')

# plt.plot([0, np.max(y_test)],
#          [0, np.max(y_test)],
#          color = 'red')

# Tweak title and axis labels.
plt.xlabel("Predicted Values: $\hat{y}$", fontsize = 20)
plt.ylabel("Actual Values: $y$", fontsize = 20)
plt.title('Actual Values vs. Predicted Values (forest)', fontsize = 24)
plt.show()

#Histogram of residuals shows a near normal distribution, with slight right skewed
resid = y_test - forest_pred
plt.hist(resid, bins=30)
plt.title('Histogram of residuals', size=15)
plt.show()












