import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Regression Problem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchmetrics import MeanSquaredError

from torchviz import make_dot

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# BUSINESS GOAL
# Predict the price of a house in London.

# Data Validation

df = pd.read_csv('london_houses.csv')
print(df.head(5))

# Check Missing Values
print(df.isnull().sum().sort_values())

# Check the data types
print(df.info())

# Check unique values for object columns
for col in df.select_dtypes(include='object').columns:
    print(f'{col}: {df[col].unique()}')

# Balcony column we need to map 
df['Balcony'] = df['Balcony'].map({'High-level Balcony': 'Yes', 'No Balcony': 'No', 'Low-level Balcony': 'Yes'})

# Change Column type from object to category
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')  

print(df.dtypes)

# Check for duplicates
print(df.duplicated().sum())
# --------------//----------------


# Exploratory Data Analysis

# Check the distribution of the target variable
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df['Price (£)'], color='blue', kde=True, ax=axes[0]).set(title='The Distribution of Target Variable - Price')
sns.histplot(df['Price (£)'],log_scale=True,color='gray', ax=axes[1]).set(title='The Distribution of Target Variable - Price (Log Scale)')
plt.savefig('images/Analyze_target_variable.png', format='png')
#plt.show()

# --------------//----------------

# Numeric Features - Bedrooms, Bathrooms, Square Meters, Building Age, Floors
numeric = df[['Bedrooms', 'Bathrooms', 'Square Meters', 'Building Age', 'Floors', 'Price (£)']]
sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm').set(title='Correlation Matrix between Numerical Features')
plt.savefig('images/Correlation_between_numeric_features.png', format='png')
#plt.show()


# Numeric Features - Bedrooms, Bathrooms, Square Meters, Building Age, Floors
fig = plt.figure(figsize=(20, 22), constrained_layout=True)
spec = fig.add_gridspec(5, 2)

# Bedrooms
ax00 = fig.add_subplot(spec[0, 0])
sns.histplot(df['Bedrooms'], color='blue', kde=True, ax=ax00).set(title='Distribution of Houses by the Bedrooms')

ax01 = fig.add_subplot(spec[0, 1])
sns.boxplot(x='Bedrooms', data=df, color='#7E1037', ax=ax01).set(title='Bedrooms Distribution')

# Bathrooms
ax10 = fig.add_subplot(spec[1, 0])
sns.histplot(df['Bathrooms'], color='blue', kde=True, ax=ax10).set(title='Distribution of Houses by the Bathrooms')

ax11 = fig.add_subplot(spec[1, 1])
sns.boxplot(x='Bathrooms', data=df, color='#7E1037', ax=ax11).set(title='Bathrooms Distribution')

# Square Meters
ax20 = fig.add_subplot(spec[2, 0])
sns.histplot(df['Square Meters'], color='blue', kde=True, ax=ax20).set(title='Distribution of Houses by the Square Meters')

ax21 = fig.add_subplot(spec[2, 1])
sns.boxplot(x='Square Meters', data=df, color='#7E1037', ax=ax21).set(title='Square Meters Distribution')

# Building Age
ax30 = fig.add_subplot(spec[3, 0])
sns.histplot(df['Building Age'], color='blue', kde=True, ax=ax30).set(title='Distribution of Houses by the Building Age')

ax31 = fig.add_subplot(spec[3, 1])
sns.boxplot(x='Building Age', data=df, color='#7E1037', ax=ax31).set(title='Building Age Distribution')

# Floors
ax40 = fig.add_subplot(spec[4, 0])
sns.histplot(df['Floors'], color='blue', kde=True, ax=ax40).set(title='Distribution of Houses by the Floors')

ax41 = fig.add_subplot(spec[4, 1])
sns.boxplot(x='Floors', data=df, color='#7E1037', ax=ax41).set(title='Floors Distribution')

plt.savefig('images/Analyze_numeric_features.png', format='png')
#plt.show()

# --------------//----------------

# Numeric Features VS Target Variable
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
spec = fig.add_gridspec(2, 3)

# Bedrooms vs Price
ax00 = fig.add_subplot(spec[0, 0])
sns.scatterplot(x=df['Bedrooms'], y=df['Price (£)'], ax=ax00).set(title='Price vs Bedrooms')

# Bathrooms vs Price
ax01 = fig.add_subplot(spec[0, 1])
sns.scatterplot(x='Bathrooms', y='Price (£)', data=df, ax=ax01).set(title='Price vs Bathrooms')

# Square Meters vs Price
ax02 = fig.add_subplot(spec[0, 2])
sns.scatterplot(x='Square Meters', y='Price (£)', data=df, ax=ax02).set(title='Price vs Square Meters')

# Building Age vs Price
ax10 = fig.add_subplot(spec[1, 0])
sns.scatterplot(x='Building Age', y='Price (£)', data=df, ax=ax10).set(title='Price vs Building Age')

# Floors vs Price
ax11 = fig.add_subplot(spec[1, 1])
sns.scatterplot(x='Floors', y='Price (£)', data=df, ax=ax11).set(title='Price vs Floors')

plt.savefig('images/Analyze_numeric_features_vs_target.png', format='png')
#plt.show()

# --------------//----------------

# Categorical Features (Yes/No) - Garden, Garage, Balcony 
fix, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.countplot(x='Garden', data=df, color='gray', ax=axes[0]).set(title='Count of Houses by Garden')
sns.countplot(x='Garage', data=df, color='gray', ax=axes[1]).set(title='Count of Houses by Garage')
sns.countplot(x='Balcony', data=df, color='gray', ax=axes[2]).set(title='Count of Houses by Balcony')
plt.savefig('images/Analyze_categorical_features_yes_no.png', format='png')
#plt.show()

# --------------//----------------

# Categorical Features (Rest) - Property Type, Heating Type, Interior Style, View, Materials, Building Status
fig, axes = plt.subplots(3, 2, figsize=(26, 14))
spec = fig.add_gridspec(3, 2)

# Property Type
ax00 = fig.add_subplot(spec[0, 0])
sns.countplot(x='Property Type', data=df, color='gray', ax=ax00).set(title='Count of Houses by Property Type')

# Heating Type
ax01 = fig.add_subplot(spec[0, 1])
sns.countplot(x='Heating Type', data=df, color='gray', ax=ax01).set(title='Count of Houses by Heating Type')

# Interior Style
ax10 = fig.add_subplot(spec[1, 0])
sns.countplot(x='Interior Style', data=df, color='gray', ax=ax10).set(title='Count of Houses by Interior Style')

# View  
ax11 = fig.add_subplot(spec[1, 1])
sns.countplot(x='View', data=df, color='gray', ax=ax11).set(title='Count of Houses by View')

# Materials
ax20 = fig.add_subplot(spec[2, 0])
sns.countplot(x='Materials', data=df, color='gray', ax=ax20).set(title='Count of Houses by Materials')

# Building Status
ax21 = fig.add_subplot(spec[2, 1])
sns.countplot(x='Building Status', data=df, color='gray', ax=ax21).set(title='Count of Houses by Building Status')

plt.savefig('images/Analyze_categorical_features_rest.png', format='png')
#plt.show()

# --------------//----------------

# Catetorical Features vs Target Variable
fig, axes = plt.subplots(3, 2, figsize=(26, 14))
spec = fig.add_gridspec(3, 2)

# Property Type
ax00 = fig.add_subplot(spec[0, 0])
sns.boxplot(x='Property Type', y='Price (£)', data=df, color='gray', ax=ax00).set(title='Distribution of Price by Property Type')

# Heating Type
ax01 = fig.add_subplot(spec[0, 1])
sns.boxplot(x='Heating Type', y='Price (£)', data=df, color='gray', ax=ax01).set(title='Distribution of Price by Heating Type')

# Interior Style
ax10 = fig.add_subplot(spec[1, 0])
sns.boxplot(x='Interior Style', y='Price (£)', data=df, color='gray', ax=ax10).set(title='Distribution of Price by Interior Style')

# View
ax11 = fig.add_subplot(spec[1, 1])
sns.boxplot(x='View', y='Price (£)', data=df, color='gray', ax=ax11).set(title='Distribution of Price by View')

# Materials
ax20 = fig.add_subplot(spec[2, 0])
sns.boxplot(x='Materials', y='Price (£)', data=df, color='gray', ax=ax20).set(title='Distribution of Price by Materials')

# Building Status
ax21 = fig.add_subplot(spec[2, 1])
sns.boxplot(x='Building Status', y='Price (£)', data=df, color='gray', ax=ax21).set(title='Distribution of Price by Building Status')

plt.savefig('images/Analyze_categorical_features_vs_target.png', format='png')
#plt.show()


# Model Fitting and Evaluation
# Data Preprocessing
df['Price (£)'] = np.log(df['Price (£)'])

# Label Encoding
encoder = LabelEncoder()
for col in df.select_dtypes(include='category').columns:
    df[col] = encoder.fit_transform(df[col])

# Standard Scaling
scaler = StandardScaler()
X = df.drop(columns='Price (£)')
y = df['Price (£)']

columns_to_scale = ['Bedrooms', 'Bathrooms', 'Square Meters', 'Building Age', 'Floors']
for col in columns_to_scale:
    X[col] = scaler.fit_transform(X[[col]])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print('Linear Regression r2_score: ', r2_score(y_test,y_pred))
print('Linear Regression Root Mean Squared Error: ',np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred))))

# --------------//----------------

# Feature Importance
resultdict = {}
for i in range(len(X.columns)):
    resultdict[X.columns[i]] = linear_model.coef_[i]
    
plt.bar(resultdict.keys(),resultdict.values(), alpha=0.5,color='gray')
plt.xticks(rotation='vertical')
plt.title('Feature Importance in Linear Regression Model')
plt.savefig('images/Feature_importance_linear_regression.png', format='png')
#plt.show()

# --------------//----------------

# Dont considere the features Bathrooms and Heating Type
X = X.drop(columns=['Bathrooms', 'Heating Type'])
print(X.columns)
y = df['Price (£)']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
print('Linear Regression r2_score: ', r2_score(y_test,y_pred))
print('Linear Regression Root Mean Squared Error: ',np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred))))

# --------------//----------------

# Random Forest Regressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print('Random Forest Regressor r2_score: ', r2_score(y_test,y_pred))
print('Random Forest Regressor Root Mean Squared Error: ',np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred))))

# --------------//----------------

# Deep Learning Model

# Run on GPU if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

train_dataset = TensorDataset(torch.tensor(X_train.to_numpy()).float(), torch.tensor(y_train.to_numpy()).float())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import NN_Model
net = NN_Model.NN_Model(input_size=X_train.shape[1], hidden_size=64, output_size=1)

# Visualize the model
y = net(X_train)
dot = make_dot(y.mean(), params=dict(net.named_parameters()), show_attrs=True, show_saved=True)
dot.render('images/Neural Network Architecture', format='png')

learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# Training loop
for epoch in range(1000):
    training_loss = 0.0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(features)
        loss = criterion(outputs, labels)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} - Training Loss: {training_loss/len(train_loader)}')



# Evaluation

test_dataset = TensorDataset(torch.tensor(X_test.to_numpy()).float(), torch.tensor(y_test.to_numpy()).float())
test_loader = DataLoader(test_dataset, batch_size=32)

mse = MeanSquaredError()

net.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = net(features)
        mse.update(outputs, labels)

test_mse = mse.compute()
print(f'Test MSE: {test_mse:.4f}')

