# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Name: YOHESH KUMAR R.M
Reg NO: 212222240118
```
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:

### The dataset
![Screenshot 2023-08-27 155200](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/0833a37c-71ed-4f5e-9ecd-c57a6c8fcacf)

### Dropping unwanted features
![Screenshot 2023-08-27 155700](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/ce2f4978-5bc6-408f-ae70-f00114bfead5)

### Checking for duplication
![Screenshot 2023-08-27 160148](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/07dc5041-4f55-4d9a-ba6d-1d7588f79374)

### Describing the dataset
![Screenshot 2023-08-27 160525](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/8f74cfd8-9476-48d1-9e48-77c008c7591b)

### Scaling the values
![Screenshot 2023-08-27 160713](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/10d1fbb4-47bc-4da4-adc9-fd6591dcd9b2)

### X Features
![Screenshot 2023-08-27 160817](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/31b79862-9f30-4deb-aaf5-4b8b9508b822)

### Y Features
![Screenshot 2023-08-27 161038](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/139305ad-0c35-4788-acc7-f72a90757d42)

### Splitting the training and testing dataset
![Screenshot 2023-08-27 161203](https://github.com/Yamunaasri/Ex.No.1---Data-Preprocessing/assets/115707860/8daa270e-6b98-470c-b2e0-2529d60c000b)


## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle
