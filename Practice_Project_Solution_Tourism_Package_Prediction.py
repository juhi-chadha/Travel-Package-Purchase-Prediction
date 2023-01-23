#!/usr/bin/env python
# coding: utf-8

# # **Classification and Hypothesis Testing Practice Project: Travel Package Purchase Prediction**
# 
# ---------------
# 
# ## **Context**
# 
# You are a Data Scientist for a tourism company named "Visit with us". The Policy Maker of the company wants to enable and establish a viable business model to expand the customer base. A viable business model is a central concept that helps you to understand the existing ways of doing the business and how to change the ways for the benefit of the tourism sector.
# 
# One of the ways to expand the customer base is to introduce a new offering of packages. Currently, there are 5 types of packages the company is offering - Basic, Standard, Deluxe, Super Deluxe, King. Looking at the data of the last year, we observed that 18% of the customers purchased the packages. However, it was difficult to identify the potential customers because customers were contacted at random without looking at the available information. 
# 
# The company is now planning to launch a new product i.e. Wellness Tourism Package. Wellness Tourism is defined as Travel that allows the traveler to maintain, enhance or kick-start a healthy lifestyle, and support or increase one's sense of well-being. This time company wants to harness the available data of existing and potential customers to target the right customers. 
# 
# You as a Data Scientist at "**Visit with us**" travel company has to analyze the customers' data and information to provide recommendations to the Policy Maker and build a model to predict the potential customer who is going to purchase the newly introduced travel package. The model will be built to make predictions before a customer is contacted.
#  
# 
# ## **Objective**
# 
# To build a model to predict which customer is potentially going to purchase the newly introduced travel package.
# 
# 
# ## **Data Description**
# 
# - CustomerID: Unique customer ID
# - ProdTaken: Whether the customer has purchased a package or not (0: No, 1: Yes)
# - Age: Age of customer
# - TypeofContact: How customer was contacted (Company Invited or Self Inquiry)
# - CityTier: City tier depends on the development of a city, population, facilities, and living standards. The categories are ordered i.e. Tier 1 > Tier 2 > Tier 3. It's the city the customer lives in. 
# - DurationOfPitch: Duration of the pitch by a salesperson to the customer
# - Occupation: Occupation of customer
# - Gender: Gender of customer
# - NumberOfPersonVisiting: Total number of persons planning to take the trip with the customer
# - NumberOfFollowups: Total number of follow-ups has been done by the salesperson after the sales pitch
# - ProductPitched: Product pitched by the salesperson
# - PreferredPropertyStar: Preferred hotel property rating by customer
# - MaritalStatus: Marital status of customer
# - NumberOfTrips: Average number of trips in a year by customer
# - Passport: The customer has a passport or not (0: No, 1: Yes)
# - PitchSatisfactionScore: Sales pitch satisfaction score
# - OwnCar: Whether the customers own a car or not (0: No, 1: Yes)
# - NumberOfChildrenVisiting: Total number of children with age less than 5 planning to take the trip with the customer
# - Designation: Designation of the customer in the current organization
# - MonthlyIncome: Gross monthly income of the customer
# 
# 

# ## **Importing the libraries required**

# In[3]:


# Importing the basic libraries we will require for the project

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# Libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the Machine Learning models we require from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Importing the other functions we may require from Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# To get diferent metric scores
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,plot_confusion_matrix,precision_recall_curve,roc_curve,make_scorer

# Code to ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')


# ## **Loading the dataset**

# In[4]:


# Loading the dataset - sheet_name parameter is used if there are Basicple tabs in the excel file.
data=pd.read_excel("Tourism.xlsx",sheet_name='Tourism')


# ## **Overview of the dataset**

# ### **View the first and last 5 rows of the dataset**
# 
# Let's **view the first few rows and last few rows** of the dataset in order to understand its structure a little better.
# 
# We will use the head() and tail() methods from Pandas to do this.

# In[8]:


data.head()

df = pd.DataFrame(data)
df.style.apply()


# In[ ]:


data.tail()


# ### **Understand the shape of the dataset**

# In[ ]:


data.shape


# * The dataset has 4888 rows and 20 columns. 

# ### **Check the data types of the columns for the dataset**

# In[ ]:


data.info()


# - We can see that 8 columns have less than 4,888 non-null values i.e. columns have missing values.

# ### **Check the percentage of missing values in each column**

# In[ ]:


pd.DataFrame(data={'% of Missing Values':round(data.isna().sum()/data.isna().count()*100,2)}).sort_values(by='% of Missing Values',ascending=False)


# - `DurationOfPitch` column has 5.14% missing values out of the total observations.
# - The `MonthlyIncome` column has 4.77% missing values out of the total observations.
# - The `Age` column has 4.62% missing values out of the total observations.
# - `TypeofContact` column has 0.51% missing values out of the total observations.
# - The `NumberOfFollowups` column has 0.92% missing values out of the total observations.
# - `PreferredPropertyStar` column has 0.53% missing values out of the total observations.
# - `NumberOfTrips` column has 2.86% missing values out of the total observations.
# - `NumberOfChildrenVisiting` column has 1.35% missing values out of the total observations.
# - We will impute these values after we split the data into train and test sets.

# ### **Check the number of unique values in each column**

# In[ ]:


data.nunique()


# - We can drop the column - CustomerID as it is unique for each customer and will not add value to the model.
# - Most of the variables are categorical except - Age, duration of pitch, monthly income, and number of trips of customers.

# **Dropping the unique values column**

# In[ ]:


# Dropping CustomerID column
data.drop(columns='CustomerID',inplace=True)


# ### **Question 1: Check the summary statistics of the dataset and write your observations (2 Marks)**
# 
# 

# **Let's check the statistical summary of the data.**

# In[1]:


data.describe().T

df = pd


# **Write your Answer here :**
# 
# - Mean and median of age column are very close to each other i.e. approx 37 and 36 respectively.
# - Duration of pitch has some outliers at the right end as the 75th percentile value is 20 and the max value is 127. We need to explore this further.
# - It seems like monthly income has some outliers at both ends. We need to explore this further.
# - The number of trips also has some outliers as the 75th percentile value is 4 and the max value is 22.
# - We can see that the target variable - ProdTaken is imbalanced as most of the values are 0. 

# ### **Check the count of each unique category in each of the categorical variables.** 

# In[ ]:


# Making a list of all catrgorical variables 
cat_col=['TypeofContact', 'CityTier','Occupation', 'Gender', 'NumberOfPersonVisiting',
       'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
       'MaritalStatus', 'Passport', 'PitchSatisfactionScore',
       'OwnCar', 'NumberOfChildrenVisiting', 'Designation']

# Printing number of count of each unique value in each column
for column in cat_col:
    print(data[column].value_counts())
    print('-'*50)


# - The Free lancer category in the occupation column has just 2 entries out of 4,888 observations.
# - We can see that Gender has 3 unique values which include - 'Fe Male' and 'Female'. This must be a data input error, we should replace 'Fe Male' with 'Female'.
# - NumberOfPersonVisiting equal to 5 has a count equal to 3 only.
# - The majority of the customers are married.
# - The majority of the customers own a car.

# In[ ]:


# Replacing 'Fe Male' with 'Female'
data.Gender=data.Gender.replace('Fe Male', 'Female')


# In[ ]:


# Converting the data type of each categorical variable to 'category'
for column in cat_col:
    data[column]=data[column].astype('category')


# In[ ]:


data.info()


# In[ ]:


df = data.copy()


# ## **Exploratory Data Analysis**

# ### **Question 2: Univariate Analysis**

# Let's explore these variables in some more depth by observing their distributions.

# We will first define a **hist_box() function** that provides both a boxplot and a histogram in the same visual, with which we can perform univariate analysis on the columns of this dataset.

# In[ ]:


# Defining the hist_box() function
def hist_box(data,col):
  f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (0.15, 0.85)}, figsize=(12,6))
  # Adding a graph in each part
  sns.boxplot(data[col], ax=ax_box, showmeans=True)
  sns.distplot(data[col], ax=ax_hist)
  plt.show()


# #### **Question 2.1:  Plot the histogram and box plot for the variable `Age` using the hist_box function provided and write your insights. (1 Mark)**

# In[ ]:


hist_box(df, "Age")


# **Write your Answer here :**
# 
# - Age distribution looks approximately normally distributed.
# - The boxplot for the age column confirms that there are no outliers for this variable
# - Age can be an important variable while targeting customers for the tourism package. We will further explore this in bivariate analysis.

# #### **Question 2.2:  Plot the histogram and box plot for the variable `Duration of Pitch` using the hist_box function provided and write your insights. (1 Mark)**

# In[ ]:


hist_box(df, 'DurationOfPitch')


# **Write your Answer here :**
# 
# - The distribution for the duration of pitch is right-skewed.
# - The duration of the pitch for most of the customers is less than 20 minutes.
# - There are some observations that can be considered as outliers as they are very far from the upper whisker in the boxplot. Let's check how many such extreme values are there.

# In[ ]:


df[df['DurationOfPitch']>40]


# - We can see that there are just two observations which can be considered as outliers.

# **Lets plot the histogram and box plot for the variable `Monthly Income` using the hist_box function**
# 

# In[ ]:


hist_box(df, 'MonthlyIncome')


# - The distribution for monthly income shows that most of the values lie between 20,000 to 40,000.
# - Income is one of the important factors to consider while approaching a customer with a certain package. We can explore this further in bivariate analysis. 
# - There are some observations on the left and some observations on the right of the boxplot which can be considered as outliers. Let's check how many such extreme values are there. 

# In[ ]:


df[(df.MonthlyIncome>40000) | (df.MonthlyIncome<12000)]


# - There are just four such observations which can be considered as outliers.

# **Lets plot the histogram and box plot for the variable `Number of Trips` using the hist_box function**
# 

# In[ ]:


hist_box(df,'NumberOfTrips')


# - The distribution for the number of trips is right-skewed 
# - Boxplot shows that the number of trips has some outliers at the right end. Let's check how many such extreme values are there. 

# In[ ]:


df.NumberOfTrips.value_counts(normalize=True)


# - We can see that most of the customers i.e. 52% have taken 2 or 3 trips.
# - As expected, with the increase in the number of trips the percentage of customers is decreasing.
# - The percentage of categories 19 or above is very less. We can consider these values as outliers.
# - We can see that there are just four observations with a number of trips 19 or greater

# **Removing these outliers form duration of pitch, monthly income, and number of trips.**

# In[ ]:


# Dropping observaions with duration of pitch greater than 40. There are just 2 such observations
df.drop(index=df[df.DurationOfPitch>37].index,inplace=True)

# Dropping observation with monthly income less than 12000 or greater than 40000. There are just 4 such observations
df.drop(index=df[(df.MonthlyIncome>40000) | (df.MonthlyIncome<12000)].index,inplace=True)

# Dropping observations with number of trips greater than 8. There are just 4 such observations
df.drop(index=df[df.NumberOfTrips>10].index,inplace=True)


# #### **Let's understand the distribution of the categorical variables**

# **Number of Person Visiting**

# In[ ]:


sns.countplot(df['NumberOfPersonVisiting'])
plt.show()


# In[ ]:


df['NumberOfPersonVisiting'].value_counts(normalize=True)


# - Most customers have 3 persons who are visiting with them. This can be because most people like to travel with family.
# - As mentioned earlier, there are just 3 observations where the number of persons visiting with the customers are 5 i.e. 0.1%.

# **Occupation**

# In[ ]:


sns.countplot(df['Occupation'])
plt.show()


# In[ ]:


df['Occupation'].value_counts(normalize=True)


# - The majority of customers i.e. 91% are either salaried or owns a small business. 
# - As mentioned earlier, the freelancer category has only 2 observations.

# **City Tier**

# In[ ]:


sns.countplot(df['CityTier'])
plt.show()


# In[ ]:


df['CityTier'].value_counts(normalize=True)


# - Most of the customers i.e. approx 65% are from tier 1 cities. This can be because of better living standards and exposure as compared to tier 2 and tier 3 cities.
# - Surprisingly, tier 3 cities have a much higher count than tier 2 cities. This can be because the company has less marketing in tier 2 cities.

# **Gender**

# In[ ]:


sns.countplot(df['Gender'])
plt.show()


# In[ ]:


df['Gender'].value_counts(normalize=True)


# - Male customers are more than the number of female customers
# - There are approx 60% male customers as compared to 40% female customers
# - This might be because males do the booking/inquiry when traveling with females which imply that males are the direct customers of the company.

# **Number of Follow ups**

# In[ ]:


sns.countplot(df['NumberOfFollowups'])
plt.show()


# In[ ]:


df['NumberOfFollowups'].value_counts(normalize=True)


# - We can see that company usually follow-ups with 3 or 4 times with their customers
# - We can explore this further and observe which number of follow-ups have more customers who buy the product.

# **Product Pitched**

# In[ ]:


sns.countplot(df['ProductPitched'])
plt.show()


# In[ ]:


df['ProductPitched'].value_counts(normalize=True)


# - The company pitches Deluxe or Basic packages to their customers more than the other packages. 
# - This might be because the company makes more profit from Deluxe or Basic packages or these packages are less expensive, so preferred by the majority of the customers.

# **Type of Contact**

# In[ ]:


sns.countplot(df['TypeofContact'])
plt.show()


# In[ ]:


df['TypeofContact'].value_counts(normalize=True)


# - There are approx 70% of customers who reached out to the company first i.e. self-inquiry. 
# - This shows the positive outreach of the company as most of the inquires are initiated from the customer's end.

# **Designation**

# In[ ]:


sns.countplot(df['Designation'])
plt.show()


# In[ ]:


df['Designation'].value_counts(normalize=True)


# - Approx 73% of the customers are at the executive or manager level.
# - We can see that the higher the position, the lesser number of observations which makes sense as executives/managers are more common than AVP/VP. 

# **Product Taken**

# In[ ]:


sns.countplot(df['ProdTaken'])
plt.show()


# In[ ]:


df['ProdTaken'].value_counts(normalize=True)


# - This plot shows the distribution of both classes in the target variable is `imbalanced`.
# - We only have approx 19% of customers who have purchased the product.

# ### **Question 3: Bivariate Analysis**

# #### **Question 3.1: Find and visualize the correlation matrix using a heatmap and write your observations from the plot. (2 Marks)**
# 
# 

# In[ ]:


cols_list = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(12, 7))
sns.heatmap(data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# **Write your Answer here :**
# 
# - The Number of trips and age have a weak positive correlation, which makes sense as age increases number of trips is expected to increase slightly.
# - Age and monthly income are positively correlated.
# - ProdTaken has a weak negative correlation with age which agrees with our earlier observation that as age increases the probability for purchasing a package decreases.  
# - No other variables have a high correlation among them.

# We will define a **stacked barplot()** function to help analyse how the target variable varies across predictor categories.

# In[ ]:


# Defining the stacked_barplot() function
def stacked_barplot(data,predictor,target,figsize=(10,6)):
  (pd.crosstab(data[predictor],data[target],normalize='index')*100).plot(kind='bar',figsize=figsize,stacked=True)
  plt.legend(loc="lower right")
  plt.ylabel('Percentage Cancellations %')


# #### **Question 3.2: Plot the stacked barplot for the variable `Marital Status` against the target variable `ProdTaken` using the stacked_barplot  function provided and write your insights. (1 Mark)**

# In[ ]:


stacked_barplot(data, "MaritalStatus", "ProdTaken" )


# **Write your Answer here :**
# 
# - The married people are the most common customer for the company but this graph shows that the conversion rate is higher for single and unmarried customers as compared to the married customers.
# - The company can target single and unmarried customers more and can modify packages as per these customers.
#  

# #### **Question 3.3: Plot the stacked barplot for the variable `ProductPitched` against the target variable `ProdTaken` using the stacked_barplot  function provided and write your insights. (1 Mark)**

# In[ ]:


stacked_barplot(df, "ProductPitched", "ProdTaken" )


# **Write your Answer here :**
# 
# - The conversion rate of customers is higher if the product pitched is Basic. This might be because the basic package is less expensive. 
# - We saw earlier that company pitches the deluxe package more than the standard package, but the standard package shows a higher conversion rate than the deluxe package. The company can pitch standard packages more often.

# **Let's plot the stacked barplot for the variable `Passport` against the target variable `ProdTaken` using the stacked_barplot function.**

# In[ ]:


stacked_barplot(data, "Passport", "ProdTaken" )


# - The conversion rate for customers with a passport is higher as compared to the customers without a passport.
# - The company should customize more international packages to attract more such customers.
# 
# 
#  
# 
# 

# **Let's plot the stacked barplot for the variable `Designation` against the target variable `ProdTaken` using the stacked_barplot function.**

# In[ ]:


stacked_barplot(data, "Designation", "ProdTaken" )


# - The conversion rate of executives is higher than other designations.
# - Customers at VP and AVP positions have the least conversion rate.

# ## **Data Preparation for Modeling**
# 

# **Separating the independent variables (X) and the dependent variable (Y)**

# In[ ]:


# Separating target variable and other variables
X=data.drop(columns='ProdTaken')
Y=data['ProdTaken']


#  **As we aim to predict customers who are more likely to buy the product, we should drop the columns `DurationOfPitch', 'NumberOfFollowups', 'ProductPitched', 'PitchSatisfactionScore'` as these columns would not be available at the time of prediction for new data.**

# In[ ]:


# Dropping columns
X.drop(columns=['DurationOfPitch','NumberOfFollowups','ProductPitched','PitchSatisfactionScore'],inplace=True)


# **Splitting the data into a 70% train and 30% test set**
# 
# Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use the stratified sampling technique to ensure that relative class frequencies are approximately preserved in each train and validation fold.

# In[ ]:


# Splitting the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=1,stratify=Y)


# **As we saw earlier, our data has missing values. We will impute missing values using median for continuous variables and mode for categorical variables. We will use `SimpleImputer` to do this.**
# 
# **The `SimpleImputer` provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median, or most frequent) of each column in which the missing values are located.**

# In[ ]:


si1=SimpleImputer(strategy='median')

median_imputed_col=['Age','MonthlyIncome','NumberOfTrips']

# Fit and transform the train data
X_train[median_imputed_col]=si1.fit_transform(X_train[median_imputed_col])

#Transform the test data i.e. replace missing values with the median calculated using training data
X_test[median_imputed_col]=si1.transform(X_test[median_imputed_col])


# In[ ]:


si2=SimpleImputer(strategy='most_frequent')

mode_imputed_col=['TypeofContact','PreferredPropertyStar','NumberOfChildrenVisiting']

# Fit and transform the train data
X_train[mode_imputed_col]=si2.fit_transform(X_train[mode_imputed_col])

# Transform the test data i.e. replace missing values with the mode calculated using training data
X_test[mode_imputed_col]=si2.transform(X_test[mode_imputed_col])


# In[ ]:


# Checking that no column has missing values in train or test sets
print(X_train.isna().sum())
print('-'*30)
print(X_test.isna().sum())


# **Let's create dummy variables for string type variables and convert other column types back to float.**

# In[ ]:


#converting data types of columns to float
for column in ['NumberOfPersonVisiting', 'Passport', 'OwnCar']:
    X_train[column]=X_train[column].astype('float')
    X_test[column]=X_test[column].astype('float')


# In[ ]:


#List of columns to create a dummy variables
col_dummy=['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'CityTier']


# In[ ]:


#Encoding categorical varaibles
X_train=pd.get_dummies(X_train, columns=col_dummy, drop_first=True)
X_test=pd.get_dummies(X_test, columns=col_dummy, drop_first=True)


# ## **Model evaluation criterion:**
# 
# #### **The model can make wrong predictions as:**
# 1. Predicting a customer will buy the product and the customer doesn't buy - Loss of resources
# 2. Predicting a customer will not buy the product and the customer buys - Loss of opportunity
# 
# #### **Which case is more important?** 
# * Predicting that customer will not buy the product but he buys i.e. losing on a potential source of income for the company because that customer will not be targeted by the marketing team when he should be targeted.
# 
# #### **How to reduce this loss i.e need to reduce False Negatives?**
# * The company wants Recall to be maximized, the greater the Recall lesser the chances of false negatives.

# ### **Building the model**
# 
# We will be building 4 different models:
# 
# - **Logistic Regression**
# - **Support Vector Machine(SVM)**
# - **Decision Tree**
# - **Random Forest**

# **Also, let's create a function to calculate and print the classification report and confusion matrix so that we don't have to rewrite the same code repeatedly for each model.**

# In[ ]:


# Creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    
    sns.heatmap(cm, annot=True,  fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# ### **Question 4: Logistic Regression (6 Marks)**

# #### **Question 4.1: Build a Logistic Regression model (Use the sklearn library) (1 Mark)**

# In[ ]:


# Fitting logistic regression model
lg = LogisticRegression()
lg.fit(X_train,y_train)


# #### **Question 4.2: Check the performance of the model on train and test data (2 Marks)**

# In[ ]:


# Checking the performance on the training data
y_pred_train = lg.predict(X_train)
metrics_score(y_train, y_pred_train)


# **Write your Answer here:**
# - We have been able to build a predictive model that can be used by the tourist company to predict the customers who are likely to accept the new package with a recall score of 25%.

# #### Let's check the performance on the test set

# In[ ]:


# Checking the performance on the test dataset
y_pred_test = lg.predict(X_test)
metrics_score(y_test, y_pred_test)


# **Write your Answer here:**
# * Using the model with default threshold the model gives a low recall but decent precision score.
# - We can’t have both precision and recall high. If you increase precision, it will reduce recall, and vice versa. This is called the precision/recall tradeoff.
# - So let's find an optimal threshold where we can balance both the metrics.

# 
# #### **Question 4.3: Find the optimal threshold for the model using the Precision-Recall Curve. (1 Mark)**
# 
# Precision-Recall curve summarizes the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
# 
# Let's use the Precision-Recall curve and see if we can find a **better threshold.**
# 

# In[ ]:


# Predict_proba gives the probability of each observation belonging to each class
y_scores_lg=lg.predict_proba(X_train)

precisions_lg, recalls_lg, thresholds_lg = precision_recall_curve(y_train, y_scores_lg[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_lg, precisions_lg[:-1], 'b--', label='precision')
plt.plot(thresholds_lg, recalls_lg[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# * We want to choose a threshold that has a high recall while also having a small drop in precision. High recall is necessary, simultaneously we also need to be careful not to lose precision too much. So the threshold value of 0.25 should be sufficient because it has good recall and does not cause a significant drop in precision.
# 
# **Note:** We are attempting to maximise recall because that is our metric of interest. Consider the F1 score as the metric of interest then we must find the threshold that provides balanced precision and recall values. In that case, the theshold value will be 0.30.

# In[ ]:


# Setting the optimal threshold
optimal_threshold = 0.25


# #### **Question 4.4: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# In[ ]:


# creating confusion matrix
y_pred_train = lg.predict_proba(X_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold)


# **Write your Answer here :**
# 
# * The model performance has improved as compared to our initial model.The recall has increased by 36%.

# #### Let's check the performance on the test set

# In[ ]:


y_pred_test = lg.predict_proba(X_test)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold)


# **Write your Answer here :**
# 
# * Using the model with a threshold of 0.25, the model has achieved a recall of 67% i.e. increase of 44%.
# - The precision has dropped compared to inital model but using optimial threshold the model is able to provide the balanced performance.
# 
# However the model performance is not good. So let's try building another model.

# ### **Question 5: Support Vector Machines (11 Marks)**

# To accelerate SVM training, let's scale the data for support vector machines.

# In[ ]:


scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train_scaled = scaling.transform(X_train)
X_test_scaled = scaling.transform(X_test)


# Let's build the models using the two of the widely used kernel functions:
# 
# 1.   **Linear Kernel**
# 2.   **RBF Kernel**
# 
# 

# #### **Question 5.1: Build a Support Vector Machine model using a linear kernel (1 Mark)**

# In[ ]:


svm = SVC(kernel='linear',probability=True) # Linear kernal or linear decision boundary
model = svm.fit(X= X_train_scaled, y = y_train)


# #### **Question 5.2: Check the performance of the model on train and test data (2 Marks)**

# In[ ]:


y_pred_train_svm = model.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# **Write your Answer here :**
# 
# - This model has completely failed to detect the class 1. The model predicted all the instances as class 0.
# - The model has an recall score of 0.
# 

# #### Checking model performance on test set

# In[ ]:


print("Testing performance:")
y_pred_test_svm = model.predict(X_test_scaled)
metrics_score(y_test, y_pred_test_svm)


# **Write your Answer here:**
# - As the dataset has an imbalanced class distribution the model almost always predicts 0. 
# - So for linear kernel the 0.5 threshold doesn't seems to work. So lets find the optimal threshold and check if the model performs well.
# 

# #### **Question 5.3: Find the optimal threshold for the model using the Precision-Recall Curve. (1 Mark)**
# 

# In[ ]:


# Predict on train data
y_scores_svm=model.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# - In this case the threshold value of 0.25 seems to be good as it has good recall and there isn't much drop in precision.

# In[ ]:


optimal_threshold_svm=0.25


# #### **Question 5.4: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# In[ ]:


print("Training performance:")
y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# **Write your Answer here :**
# 
# - The model performance has improved by selecting the optimal threshold of 0.25.
# - The recall has increased from 0 to 56%.

# In[ ]:


y_pred_test = model.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# **Write your Answer here :**
# 
# * SVM model with **linear kernel** is not overfitting as the accuracy is around 78% for both train and test dataset
# * The model has a **Recall** of 61% which is highest compared to the above models.
# * At the optimal threshold of .25, the model performance has improved really well. The F1 score has improved from 0.00 to 0.52.
# 
# Lets try using non-linear kernel and check if it can improve the performance.

# #### **Question 5.5: Build a Support Vector Machines model using an RBF kernel (1 Mark)**

# In[ ]:


svm_rbf=SVC(kernel='rbf',probability=True)
# Fit the model
svm_rbf.fit(X_train_scaled,y_train)


# #### **Question 5.6: Check the performance of the model on train and test data (2 Marks)**
# 
# 

# In[ ]:


y_pred_train_svm = svm_rbf.predict(X_train_scaled)
metrics_score(y_train, y_pred_train_svm)


# **Write your Answer here :**
# 
# - When compared to the baseline svm model with linear kernel, the model's performance on training data has been slightly improved by using an RBF kernel.

# #### Checking model performance on test set

# In[ ]:


y_pred_test = svm_rbf.predict(X_test_scaled)

metrics_score(y_test, y_pred_test)


# **Write your Answer here :**
# 
# - When compared to the baseline svm model with linear kernel, the recall score on testing data has increased from 0% to 26%.

# In[ ]:


# Predict on train data
y_scores_svm=svm_rbf.predict_proba(X_train_scaled)

precisions_svm, recalls_svm, thresholds_svm = precision_recall_curve(y_train, y_scores_svm[:,1])

# Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_svm, precisions_svm[:-1], 'b--', label='precision')
plt.plot(thresholds_svm, recalls_svm[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.show()


# In[ ]:


optimal_threshold_svm=0.17


# #### **Question 5.7: Check the performance of the model on train and test data using the optimal threshold. (2 Marks)**

# #### Checking model performance on training set

# In[ ]:


y_pred_train_svm = model.predict_proba(X_train_scaled)
metrics_score(y_train, y_pred_train_svm[:,1]>optimal_threshold_svm)


# **Write your Answer here :**
# 
# - SVM model with **RBF kernel** is performing better compared to the linear kernel.
# - The model has achieved a recall score of 0.78 but there is a slight drop in the precision value.
# - Using the model with a threshold of 0.17, the model gives a better recall score compared to the initial model.
# 

# #### Checking model performance on test set

# In[ ]:


y_pred_test = svm_rbf.predict_proba(X_test_scaled)
metrics_score(y_test, y_pred_test[:,1]>optimal_threshold_svm)


# **Write your Answer here :**
# 
# * The **recall score** for the model is around 69%.
# * At the optimal threshold of .17, the model performance has improved from 0.26 to 0.69.
# - This is the best performing model when compared to SVM with linear kernel and Logistic Regression because it provides good recall with no big drop in precision as well.
# 
# Let's build some non-linear models and see if they can outperform linear models.

# ### **Question 6: Decision Trees (7 Marks)**

# #### **Question 6.1: Build a Decision Tree Model (1 Mark)**

# In[ ]:


model_dt = DecisionTreeClassifier(random_state=1)
model_dt.fit(X_train, y_train)


# #### **Question 6.2: Check the performance of the model on train and test data (2 Marks)**

# In[ ]:


# Checking performance on the training dataset
pred_train_dt = model_dt.predict(X_train)
metrics_score(y_train, pred_train_dt)


# **Write your Answer here :**
# 
# * Almost 0 errors on the training set, each sample has been classified correctly.
# * Model has performed very well on the training set.
# * As we know a decision tree will continue to grow and classify each data point correctly if no restrictions are applied as the trees will learn all the patterns in the training set.
# * Let's check the performance on test data to see if the model is overfitting.

# #### Checking model performance on test set

# In[ ]:


pred_test_dt = model_dt.predict(X_test)
metrics_score(y_test, pred_test_dt)


# **Write your Answer here :**
# 
# * The decision tree model is clearly overfitting. However the decision tree has better performance compared to Logistic Regression and SVM models.
# * We will have to tune the decision tree to reduce the overfitting.

#  #### **Question 6.3: Perform hyperparameter tuning for the decision tree model using GridSearch CV (1 Mark)**

# In[ ]:


# Choose the type of classifier.
estimator = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    "max_depth": np.arange(1,100,10),
    "max_leaf_nodes": [50, 75, 150, 250],
    "min_samples_split": [10, 30, 50, 70],
}
# Run the grid search
grid_obj = GridSearchCV(estimator, parameters, cv=5,scoring='recall',n_jobs=-1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
estimator.fit(X_train, y_train)


# #### **Question 6.4: Check the performance of the model on the train and test data using the tuned model (2 Mark)**

# #### Checking performance on the training set 

# In[ ]:


# Checking performance on the training dataset
dt_tuned = estimator.predict(X_train)
metrics_score(y_train,dt_tuned)


# In[ ]:


# Checking performance on the training dataset
y_pred_tuned = estimator.predict(X_test)
metrics_score(y_test,y_pred_tuned)


# **Write your Answer here :**
# 
# * Decision tree model with default parameters is overfitting the training data and is not able to generalize well.
# * Tuned moded has provided a generalised performance with balanced precision and recall values.
# - However, there is still some overfitting, and model performance on test data has not significantly improved.

# #### **Visualizing the Decision Tree**

# In[ ]:


feature_names = list(X_train.columns)
plt.figure(figsize=(20, 10))
out = tree.plot_tree(
    estimator,
    max_depth=4,
    feature_names=feature_names,
    filled=True,
    fontsize=9,
    node_ids=False,
    class_names=None,
)
# below code will add arrows to the decision tree split if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")
        arrow.set_linewidth(1)
plt.show()


# #### **Question 6.5: What are some important features based on the tuned decision tree? (1 Mark)**

# In[ ]:


# Importance of features in the tree building
importances = model_dt.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# **Write your Answer here :**
# 
# * We can see that the tree has become simpler and the rules of the trees are readable.
# * The model performance of the model has been generalized.
# 
# 
# * We observe that the most important features are:
#     * Monthly Income
#     * Age
#     - Number of trips
# 

# ### **Question 7: Random Forest (4 Marks)**

# #### **Question 7.1: Build a Random Forest Model (1 Mark)**

# In[ ]:


rf_estimator = RandomForestClassifier( random_state = 1)

rf_estimator.fit(X_train, y_train)


# #### **Question 7.2: Check the performance of the model on the train and test data (2 Marks)**

# In[ ]:


y_pred_train_rf = rf_estimator.predict(X_train)

metrics_score(y_train, y_pred_train_rf)


# **Write your Answer here :**
# 
# - Almost 0 errors on the training set, each sample has been classified correctly.
# - Model has performed very well on the training set.

# In[ ]:


y_pred_test_rf = rf_estimator.predict(X_test)

metrics_score(y_test, y_pred_test_rf)


# **Write your Answer here :**
# 
# * The Random Forest classifier **seems to be overfitting**.
# - The recall score is 0.47 which is low compared to other models.
# - We can reduce overfitting and improve recall by hyperparameter tuning.

# #### **Question 7.3: What are some important features based on the Random Forest? (1 Mark)**

# Let's check the feature importance of the Random Forest

# In[ ]:


importances = rf_estimator.feature_importances_

columns = X_train.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)


plt.figure(figsize=(8, 8))
plt.title("Feature Importances")
sns.barplot(importance_df.Importance, importance_df.index,color="violet")


# **Write your Answer here :**
# 
# - The Random Forest further verifies the results from the decision tree, that the most important features are monthly income, age and number of trips.
# - monthly income is most important feature. If the monthly income of customer is high he/she is most likely to accept the tour package.
# - Age is also a key feature, probably as customers with age between 25-50 are most likely to accept the newly introduced tour package.

# ### **Conclusion:**
# - The SVM with RBF kenel has outperformed other models and provided balanced metrics.
# - We have been able to build a predictive model that can be used by the tourist company to predict the customers who are likely to accept the new package with the recall score of 0.69 formulate marketing policies accordingly.
# 

# ### **Question 8: Conclude ANY FOUR key takeaways for business recommendations (4 Marks)**

# **Write your Answer here :**
# 
# - Our analysis shows that very few customers have passports and they are more likely to purchase the travel package. The company should customize more international packages to attract more such customers.
# - We have customers from tier 1 and tier 3 cities but very few from tier 2 cities. The company should expand its marketing strategies to increase the number of customers from tier 2 cities.
# - We saw in our analysis that people with higher income or at high positions like AVP or VP are less likely to buy the product. The company can offer short-term travel packages and customize the package for higher- income customers with added luxuries to target such customers.
# - When implementing a marketing strategy, external factors, such as the number of follow-ups, time of call, should also be carefully considered as our analysis shows that the customers who have been followed up more are the ones buying the package.
# - After we identify a potential customer, the company should pitch packages as per the customer's monthly income, for example, do not pitch king packages to a customer with low income and such packages can be pitched more to the higher-income customers.
# - We saw in our analysis that young and single people are more likely to buy the offered packages. The company can offer discounts or customize the package to attract more couples, families, and customers above 30 years of age.

# ## **Happy Learning!**
