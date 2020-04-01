import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix
from keras import optimizers

from train_data import read_train_data
from test_data import read_test_data
from model import model_cnn


x_train = read_train_data()
print("x : ",x_train.head(5))


x_test = read_test_data()
print("x_test : ",x_test.head(5))

#List of features
features = ["%s%s" %("pixel",pixel_no) for pixel_no in range(0,784)]
x_train_features = x_train[features]
print(x_train_features.shape)

#Test Data shape
print(x_test.shape)

#Convert single digit to one dimentional array
x_train_labels = x_train["label"]
x_train_labels_categorical = np_utils.to_categorical(x_train_labels)
x_train_labels_categorical[0:3]

#Train test split to train and test model
X_train,X_test,Y_train,Y_test = train_test_split(x_train_features,x_train_labels_categorical,test_size=0.10,random_state=32)

#model
model = model_cnn()
model.compile(loss= categorical_crossentropy,optimizer='Adam',metrics=['accuracy'])
model.fit(X_train.values, Y_train,batch_size=128,epochs=50,verbose=1)

#Predict the digit for given input
pred_classes = model.predict_classes(X_test.values)

#Finally generate kaggle submission file
submission = pd.DataFrame({
    "ImageId": X_test.index+1,
    "Label": pred_classes})
print(submission[0:10])

submission.to_csv('./keras_model_1.csv', index=False)
