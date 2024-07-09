import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_excel('/content/drive/My Drive/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (2).xls')
df.head()

#get a html report
#report= data.profile_report()
#report.to_file(output_file=report.html)

#get data information
#Summary Statistics
df.info()

#Summary for Numerical Features
print (df.describe())

# Defining the list of categorical columns
categorical_columns = ['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment',
                       'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns)
print(df_encoded)

#Numerical Columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns.to_list()
print(numerical_columns)

#Checking for Missing Values
missing_values = df.isnull().sum()
print("missing_values:\n", missing_values)

#Checking for Outliers
numerical_cols=df.select_dtypes (include='int64').columns
plt.figure(figsize=(10,8))
for i, column in enumerate(numerical_cols):
    plt.subplot(5,4,i+1)
    sns.boxplot(x=df [column])
    plt.title('column')
    plt.tight_layout()
plt.show()


#Z-Scores for numerical Values
import scipy.stats as stats
z_scores=stats.zscore(df.select_dtypes(include='int64'))

#Outliers using 3 standard deviation threshold
outliers = df[(z_scores > 3).any(axis=1)]

#Remove Outliers
df_no_outliers = df[(z_scores <= 3).all(axis=1)]


df_transformed=np.log1p(df.select_dtypes(include='int64'))

#Data Training and Testing
#Features(X) and Target(Y)
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']

#Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Shapes of Split Data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['number']).columns
categorical_cols = X_train.select_dtypes(exclude=['number']).columns

# Create transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing values in categorical columns
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # One-hot encode categorical features

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Create a pipeline that includes preprocessing and the Random Forest model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier())])

# Fit the pipeline to your training data
pipeline.fit(X_train, y_train)

# Evaluate the model
print('Model Accuracy:', pipeline.score(X_test, y_test))


#Data Validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(pipeline,X_train,y_train,cv=5,scoring='r2')
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())


#Evaluation
#Regression Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred=pipeline.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test,y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test,y_pred))
print("R-Squared:", r2_score(y_test,y_pred))


#Classification Metrics
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

y_pred=pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))


#Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
best_pipeline=random_search.best_estimator_
y_pred=best_pipeline.predict(X_test)
print ("TestSet R-Squared Score:", r2_score(y_test,y_pred))


# Numerical Features Distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,8))
sns.histplot(df['Age'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(df['EmpHourlyRate'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of EmpHourlyRate')
plt.xlabel('EmpHourlyRate')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(df['EmpJobSatisfaction'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Distribution of EmpJobSatisfaction')
plt.xlabel('EmpJobSatisfaction')
plt.ylabel('Count')
plt.show()

#Categorical Features
gender_counts = df['Gender'].value_counts()

plt.figure(figsize=(10,8))
# Pass the Series gender_counts directly to the barplot function
sns.barplot(x=gender_counts.index, y=gender_counts.values)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

EmpDepartment_counts = df['EmpDepartment'].value_counts()

plt.figure(figsize=(10,8))
sns.barplot(x=EmpDepartment_counts.index, y=EmpDepartment_counts.values)
plt.title('EmpDepartment')
plt.xlabel('EmpDepartment')
plt.ylabel('Count')
plt.show()

EducationBackground_counts = df['EducationBackground'].value_counts()

plt.figure(figsize=(10,8))
sns.barplot(x=EducationBackground_counts.index, y=EducationBackground_counts.values)
plt.title('EducationBackground')
plt.xlabel('EducationBackground')
plt.ylabel('Count')
plt.show()

#Data Relationship Analysis
#Pairplot
sns.pairplot(df[['Age', 'EmpHourlyRate', 'EmpJobSatisfaction','PerformanceRating','TotalWorkExperienceInYears']], diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features', y=1)
plt.show()

#Correlation Matrix
# Select only numeric columns for correlation calculation
numerical_df = df.select_dtypes(include=['number'])

# Calculate correlation on the numerical dataframe
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(10,8))
sns.histplot(df['PerformanceRating'], bins=20, kde=True, color='blue', edgecolor='black')
plt.title('Performance Rating Distribution')
plt.xlabel('Performance Rating')
plt.ylabel('Count')
plt.show()

