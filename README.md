# Ex01-Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for a simple dataset with one input and one output.

## THEORY

We create a simple dataset with one input and one output.This data is then divided into test and training sets for our Neural Network Model to train and test on. <br>
The NN Model contains 5 nodes in the first layer, 10 nodes in the following layer, which is then connected to the final output layer with one node/neuron.
The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. <br>The model is then train for 2000 epochs.<br> We then perform an evaluation of the model with the test data. An user input is then predicted with the model. Finally, we plot the Error VS Iteration graph for the given model.

## Neural Network Model

![nn](https://user-images.githubusercontent.com/89703145/187489272-60ce3dd0-9a1c-4645-98ce-b7177bf750d0.jpg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
 Developed By : Gunanithi S
 
 Register Number : 212220220015
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("datasheet.csv")
df.head()

inp=df[["Input"]].values
out=df[["Output"]].values
Input_train,Input_test,Output_train,Output_test=train_test_split(inp,out,test_size=0.3,random_state=40)
Scaler=MinMaxScaler()
Scaler.fit(Input_train)
Scaler.fit(Input_test)
Input_train=Scaler.transform(Input_train)
Input_test=Scaler.transform(Input_test)

model=Sequential([Dense(5,activation='relu'),
                  Dense(7,activation='relu'),
                  Dense(1)])
model.compile(loss="mse",optimizer="rmsprop")
history=model.fit(Input_train,Output_train, epochs=5000,batch_size=32)

model.evaluate(Input_test,Output_test)

xn1=[[39]]
xn11=Scaler.transform(xn1)
model.predict(xn11)

import matplotlib.pyplot as plt
plt.suptitle("   Gunanithi")
plt.title("Error VS Iteration")
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.plot(pd.DataFrame(history.history))
plt.legend(['train'] )
plt.show()
```

## Dataset Information

![Screenshot (20)](https://user-images.githubusercontent.com/89703145/187491482-7b45469c-8df9-4612-85c3-bda97cae010a.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![Graph](https://user-images.githubusercontent.com/89703145/187490695-35c1f7d0-265d-4467-a994-3d3a713faa6b.png)

### Test Data Root Mean Squared Error

![Screenshot (22)](https://user-images.githubusercontent.com/89703145/187492669-40405d7a-0845-49dc-b0f6-cee65f44fa20.png)

### New Sample Data Prediction

![Screenshot (21)](https://user-images.githubusercontent.com/89703145/187492207-7143d407-c68b-4e7a-94cf-40df3ee7a670.png)

## RESULT
Hence, a simple Neural Network Model has been implemented successfully.
