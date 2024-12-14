#these are the libraries and packages which will be used for this or this type of dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Here is our dataset which is a csv file.So to read this file we'll use dataset.read_csv("")
en = pd.read_csv("C:\\Users\\Dell\\Downloads\\insurance_data.csv")

#To print the data of adataset:
en

#here we'll plot a basic graph for our data:
plt.figure(figsize=(15, 5))
x = en['age']
y = en['bought_insurance']
plt.title('insurance by age')
plt.xlabel(' AGE')
plt.ylabel('BOUGHT INSURANCE')
plt.grid(True)
plt.scatter(x, y)

#Now to understand the flow of data we'll perform linear regression and we'll train our model on this dataset:
rg = lm.LinearRegression()    #Used for linear regression 
logreg = LogisticRegression()
X = en[['age']].values
y = en['bought_insurance'].values
rg.fit(en[['bought_insurance']], en.age)


#used to plot linear regression plot.
plt.figure(figsize=(10, 5))
plt.scatter(en['age'], en['bought_insurance'], marker='X')
plt.plot(rg.predict(en[['bought_insurance']]),en['bought_insurance'], color='r')

#Define the Parts of linear regression equation (y=mx+c) for segmoid function:
logreg.fit(en[['bought_insurance']], en.age)
coef = logreg.coef_[1] #It will be 'm' of Linear regression.
intercept = logreg.intercept_[1] #It will be 'c' of linear regression.

# Extract the columns you want to use for modeling
X = en[['age']].values
y = en['bought_insurance'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

coef = model.coef_[0][0]
intercept = model.intercept_[0]

# Define the sigmoid function


def sigmoid(x):
    z = (coef * x) + intercept
    return 1 / (1 + np.exp(-z))


# Generate input values within the desired range
start = min(X)[0] - 1
end = max(X)[0] + 1
num_points = 100  # Increase the number of points for a smoother plot
x = np.linspace(start, end, num_points)

# Calculate sigmoid values using the sigmoid function
y_sigmoid = sigmoid(x)

# Plot the sigmoid-like curve
plt.plot(x, y_sigmoid, color='blue', label='Sigmoid Curve')
plt.scatter(X, y, color='red', marker='o', label='Data Points')
plt.xlabel('Age')
plt.ylabel('Probability (Bought Insurance)')
plt.title('Sigmoid-Like Plot for Age vs. Probability')
plt.grid(True)
plt.legend()
# plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
plt.show()
