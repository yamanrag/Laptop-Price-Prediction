#!/usr/bin/env python
# coding: utf-8

# # Imoprting Libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('laptop_data.csv')


# In[3]:


df.head(10)


# In[4]:


df.shape


# df.info()  #This provide information about all the data 

# In[5]:


df.duplicated().sum() #Checking duplicate rows which can ruin our analysis


# In[6]:


df.isnull().sum()


# In[7]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# # Pre Processing

# In[8]:


df.drop(columns=['Unnamed: 0'],inplace=True) #To remove the column


# In[9]:


df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')


# In[10]:


df['Ram']=df['Ram'].astype('int32') #convert object to int
df['Weight']=df['Weight'].astype('float32') #convert object to float


# In[11]:


df.info()


# # Data Analysis

# In[12]:


import seaborn as sns
sns.distplot(df['Price'])


# In[13]:


df['Company'].value_counts().plot(kind='bar') #Getting Laptops arrording to company or brand


# In[14]:


from matplotlib import pyplot as plt
sns.barplot(x=df['Company'],y=df['Price']) # for checking Price according to brand
plt.xticks(rotation='vertical')
plt.show()


# In[15]:


df['TypeName'].value_counts().plot(kind='bar')


# In[16]:


from matplotlib import pyplot as plt
sns.barplot(x=df['TypeName'],y=df['Price']) # for checking Price according to brand
plt.xticks(rotation='vertical')
plt.show()


# In[17]:


sns.distplot(df['Inches'])


# In[18]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[19]:


df['ScreenResolution'].value_counts()


# # Feature Engineering

# In[20]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0) #How many laptops are touchscreen


# In[21]:


df.sample(5)


# In[22]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[23]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[24]:


df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0) #How many laptops are IPS Panel


# In[25]:


df['Ips'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[27]:


new= df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[28]:


df['X_res']=new[0] #Creating 2 new columns
df['Y_res']=new[1]


# In[29]:


df.head()


# In[30]:


df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])  #extraction of x resolution


# In[31]:


df.head()


# In[32]:


df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')


# In[33]:


df.info()


# In[34]:


df.corr()['Price']#Checking the correlation


# In[35]:


df['ppi']=(((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float') #Creating New Column name=ppi


# In[36]:


df.corr(numeric_only=True)['Price']


# In[37]:


df.drop(columns=['ScreenResolution'],inplace=True) #Now removing screen resolution column 


# In[38]:


df.head()


# In[39]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True) #Now removing unnecessary column 


# In[40]:


df.head()


# In[41]:


df['Cpu'].value_counts()


# In[42]:


df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3])) #Extracting first three word from CPU


# In[43]:


df.head()


# # Creating New function

# In[44]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3' :
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[45]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[46]:


df.head()


# In[47]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[48]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[49]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[50]:


df.head()


# In[51]:


df['Ram'].value_counts().plot(kind='bar')


# In[52]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[53]:


df['Memory'].value_counts()


# In[54]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


#for checking HDD,SSD,Hybrid,Flash_Storage value


# In[55]:


df.head()


# In[56]:


df.drop(columns=['Memory'],inplace=True)


# In[57]:


df.head()


# In[58]:


df.corr()['Price']


# In[59]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[60]:


df.head()


# In[61]:


df['Gpu'].value_counts()


# In[62]:


df['Gpu brand']=df['Gpu'].apply(lambda x:x.split()[0])


# In[63]:


df.head()


# In[64]:


df['Gpu brand'].value_counts()


# In[65]:


df=df[df['Gpu brand']!='ARM']


# In[66]:


df['Gpu brand'].value_counts()


# In[67]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)


# In[68]:


df.drop(columns=['Gpu'],inplace=True)


# In[69]:


df.head()


# In[70]:


df['OpSys'].value_counts()


# In[71]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# # Creating New Function

# In[72]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[73]:


df['os']=df['OpSys'].apply(cat_os)


# In[74]:


df.head()


# In[75]:


df.drop(columns=['OpSys'],inplace=True)


# In[76]:


sns.barplot(x=df['os'],y=df['Price'])


# In[77]:


sns.distplot(df['Weight'])


# In[78]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[79]:


df.corr()['Price']


# In[80]:


sns.heatmap(df.corr()) #for checking the correlation with other columns


# In[81]:


sns.distplot(np.log(df['Price']))


# In[82]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# In[83]:


X


# In[84]:


y  #prices are decreased due to log


# In[85]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[86]:


X_train


# In[87]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[88]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# # Checking All the Models

# # Linear regression

# In[89]:


step1=ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output= False ,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[90]:


np.exp(0.21)


# # Rigid Regression

# In[91]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Lasso Regression

# In[92]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # KNN

# In[93]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Decision Tree

# In[94]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # SVM

# In[95]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Random Forest

# In[96]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Extra Trees

# In[97]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              bootstrap=True,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # AdaBoost

# In[98]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Gradient Boost

# In[99]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # XgBoost

# In[100]:


from xgboost import XGBRegressor
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Voting Regressor

# In[101]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
et = ExtraTreesRegressor(n_estimators=100,random_state=3,bootstrap=True,max_samples=0.5,max_features=0.75,max_depth=10)

step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Stacking

# In[102]:


from sklearn.ensemble import VotingRegressor,StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Exporting the Model

# In[103]:


import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[104]:


df


# In[105]:


X_train

