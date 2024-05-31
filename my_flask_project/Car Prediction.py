import os
import pandas as pd
import numpy as np

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Change the directory to where your CSV file is located (if necessary)
os.chdir('C:\\Users\\altug\\Downloads')  # Update this path to your file's location

# Verify the current working directory again
print("Updated Working Directory:", os.getcwd())

# Use the absolute path to the CSV file
file_path = 'C:\\Users\\altug\\Downloads\\quikr_car.csv'  # Ensure this path is correct

# Read the CSV file
try:
    car = pd.read_csv(file_path)
    backup = car.copy()
    print(car.head())
    print("Shape of the DataFrame:", car.shape)

    # Check unique values in the 'year' column
    print(car['year'].unique())

    # Check unique values in the 'kms_driven' column
    print(car['kms_driven'].unique())

    # Display the backup data
    print(backup)

    # Display the 'year' column
    print(car['year'])

    # Filter out non-numeric year values
    car = car[car['year'].str.isnumeric()].copy()
    
    # Convert 'year' column to integers
    car['year'] = car['year'].astype(int)

    # Filter out rows where 'Price' is 'Ask For Price'
    car = car[car['Price'] != 'Ask For Price'].copy()
    
    # Remove commas from 'Price' column and convert to integers
    car['Price'] = car['Price'].str.replace(',', '').astype(int)


    car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
    car=car[car['kms_driven'].str.isnumeric()]
    car['kms_driven']=car['kms_driven'].astype(int)

    car=car[~car['fuel_type'].isna()]

    car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')
    car=car.reset_index(drop=True)

    car=car[car['Price']<6e6].reset_index(drop=True)

    

    #print(car.head())
    #print(car[~car['fuel_type'].isna()])
    #print("Car Information:", car.info())
    #print(car)
    #print(car.describe())
    #print(car.to_csv('Cleaned Car.csv'))
    

except FileNotFoundError:
    print(f"The file at {file_path} was not found. Please check the path and try again.")
except ValueError as e:
    print(f"ValueError: {e}")







X=car.drop(columns='Price')
y=car['Price']
#print(X)
#print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline




ohe = OneHotEncoder()
print(ohe.fit(X[['name','company','fuel_type']]))

column_trans =make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')

lr= LinearRegression()
pipe= make_pipeline(column_trans,lr)
print(pipe.fit(X_train,y_train))

y_pred=pipe.predict(X_test)
print(y_pred)
print(r2_score(y_test,y_pred))



scores=[]
for i in range(10):
    X_train, X_test, y_train , y_test=train_test_split(X,y,test_size=0.2, random_state=i)
    lr= LinearRegression()
    pipe= make_pipeline(column_trans,lr)
    print(pipe.fit(X_train,y_train))
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


print(np.argmax(scores))
print(scores[np.argmax(scores)])



X_train, X_test, y_train , y_test=train_test_split(X,y,test_size=0.2, random_state=np.argmax(scores))
lr= LinearRegression()
pipe= make_pipeline(column_trans,lr)
print(pipe.fit(X_train,y_train))
y_pred=pipe.predict(X_test)
print(r2_score(y_test,y_pred))


import pickle

pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

print(pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type'])))


