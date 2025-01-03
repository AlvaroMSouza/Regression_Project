# Predicting House Prices in London

Houses in London is a dataset that contains various characteristics and price information about houses in London. Our goal is to develop and compare 3 models and see each of them is better at predicting the prices of the London houses. In order to do that it was necessary to follow the steps below. 

Models:
* Linear Regression
* Random Forest
* Deep Learning Neural Network

More information about the dataset is available [here](https://www.kaggle.com/datasets/oktayrdeki/houses-in-london/data)

## Data Validation
The dataset consists of 1000 entries and 17 columns. I am going to examine, validate and clean each column in the dataset, so in order to do that I am going to execute the following steps:

* Check Missing Values.
* Analyze the dataset and check the types of each column to see if it is necessary to change them.
* Check Duplicates.

### Checking NaN
This dataset has no missing values. 

### Checking Data Types
After analyzing the types of each column, we can immediatly see that there are a lot of columns with the type *object* that need to be changed for the correct type, *category*. Besides that after close inspecting on the column **Balcony**, I noticed that it was necessary to transform the values in order to have only Yes or No values, instead of different categories of the Balcony. I transform *High-Level Balcony*, *Low-Level Balcony* into 'Yes' value and *No Balcony* into 'No' value.

### Checking Duplicates
This dataset has no duplicates. 


## Exploratory Analysis
I have explored the target variable and features for each house, and relationships between features and target variables.

### Target Variable - Price
Since we need to predict the price, the price variable would be our target variable. From the histogram it was possible to understand that there is a longer right tail. Therefore, I applied a log transformation of the price variable in order to have a distribution close to normal distribution.

### Numeric Features - Bedrooms, Bathrooms, Square Meters, Building Age, Floors
In order to understand the numeric features and analyse each one of them, I decided to use histograms and boxplots to visualize the distribution of each feature and their outliers. At the end I also decided to do a heatmap to analyse the correlation between the numeric features and the target variable.

By the histograms from the figure below we can conclude that the distribution of **Bedrooms** for the Houses is basically the same throught the different numbers of bedrooms. We can see this pattern on the rest of the numeric features where every feature has a uniform distribution. 

From the heatmap, we can conclude that there is strong linear positive relationship between the target variable **Price** and the feature **Square Meters**. The rest of the variables don't present any correlation neither positive or negative.

### Relationship between Numeric Features VS Target
Taking into consideration the heatmap presented and now the scatterplot, we can see a strong linear relationship between **Price** and **Square Meters**. When it comes to the relationship between **Price** and the other features, besides **Building Age**, we can see clusters which is normal taking into consideration the uniform distribution. At last, the numeric feature **Building Age** and the target varible **Price** don't have any relationship.


### Categorical Features - Address, Neighborhood, Garden, Garage, Property  Type, Heating Type, Balcony, Interior Style, View, Materials, Building Status
From the chart called *Analyze_categorical_features_yes_no.png* and *Analyze_categorical_features_rest.png* it is possible to observe that all categorical features have uniform distribution, besides **Balcony** where the most frequent category is having a house with a balcony.

### Relationship between Categorical Features VS Target
I investigated the relationship between the categorical features and the target variable and what I conclude was since most of the features have a uniform distribution their isn't a big difference between each category from a feature and the target variable. 

## Model Fitting and Evaluation
Predicting the price is a regression problem in machine learning. I am choosing the Linear Regression model, Random Forest regression model and Neural Network in order to see the diferences between this models.

For the evaluation, I am choosing R squared and RMSE (Root Mean Squared Error) to evaluate the model. R squared measures how well the model fits dependent variables (i.e. features). RMSE measures how much your predicted results deviate from the actual number.

Since there are a lot of features that have a uniform distribution I decided to first use all features to train and test the *Linear Regression* for further analysis on the **Feature Importance**. Then, after analyzing I will choose the most important features and train again the *Linear Regression* model and the others that I mention before and analyze the results.

### Prepare Data for Modelling
To enable modelling, we chose  as features Neighborhood, Bedrooms, Bathrooms, Square Meters, Building Age, Garden, Garage, Floors, Property Type, Heating Type, Balcony, Interior Style, View, Materials, Building Status and Balcony. For target variable I will use price. I also have made the following changes:
* Normalize the numeric features
* Convert the categorical variables into numeric features
* Split the data into a training set and a test set

### Linear Regression
I obtain a Linear Regression Model with R squared of 0.71 and RMSE of 548754.956. After that I decided to analyze the Feature Importance and what I found was that the Features: **Square Meters**, **Property Type**, **Garage** where the most relevant features, while **Bathrooms** and **Heating Type** didn't contribute for training. After taking those features from the dataset and retrain the model I conclude that the change was almost non-existent, so it **is not necessary**. 

### Random Forest 
Random Forest Regressor outperformed the Linear Regression by having a R squared of 0.895 and RMSE of 281639.987. **Random Forest is a better solution** compared with the Linear Regression.

### Deep Learning Neural Network
I built a neural network with a simple architecture with four fully connected linear layers, interspersed with ReLu activation functions. The result I obtain with this model was really good since the training loss and validation loss both decrease and stabilize at a specific point as we can see by the image below.

![Compare Loss per poch](https://github.com/user-attachments/assets/3f2684c0-7cfc-410f-90e3-5db15166e556)

