import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import pandas as pd

# import, reindex, and validate data
mnist_train_small = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",")

mnist_train_small = mnist_train_small.reindex(
    np.random.permutation(mnist_train_small.index))

mnist_train_small.describe()

print(mnist_train_small)

# function to process raw data
def preprocess_data(mnist_train_small):
  """
  Args: mnist_train_small dataframe of number shapes and labels
  Returns: dataframe of number image pixels (important features)
  and array of image labels (which number it is)
  """
  output_targets=pd.DataFrame()
  output_features=mnist_train_small.drop("6",axis=1)
  output_targets["label"]=mnist_train_small["6"]
  return output_features,output_targets

# preprocess data
X,y = preprocess_data(mnist_train_small)

# one-hot encode targets
y_encoded = keras.utils.to_categorical(y,num_classes=10)

# train test split data
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded)

# neural net
from keras.models import Sequential
from keras.layers import Dense, Activation

# define layers of and compile model
model = Sequential([
    Dense(300, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(300,input_shape=(32,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model to train data
model.fit(X_train, y_train, epochs=100, batch_size=400)

# evaluate accuracy on test data
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
print(loss_and_metrics)
