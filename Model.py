import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout

def model_cnn():

	#Architecture of keras model
	x = input(shape=(784,))
	h = (Dense(64,activation='relu'))(x)
	y = (Dense(128, activation='relu'))(h)
	z = (Dense(256, activation='relu'))(y)
	res = (Dense(10, activation='softmax'))(z)

	return model(inputs=x,outputs=res)