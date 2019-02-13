import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation

# import, reindex, and validate data
mnist_train = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv", sep=",")

mnist_train = mnist_train.reindex(
    np.random.permutation(mnist_train.index))

# create 2 dataframe: bigger than 4 and less than or equal to 4
mnist_bigs_train = mnist_train[mnist_train['6']>4]
mnist_smalls_train = mnist_train[mnist_train['6']<=4]

print("Bigs:\n",mnist_bigs_train.describe())
print("Smalls:\n",mnist_smalls_train.describe())

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

X_bigs, y_bigs = preprocess_data(mnist_bigs_train)
X_smalls, y_smalls = preprocess_data(mnist_smalls_train)

# one-hot encode targets
y_bigs_encoded = keras.utils.to_categorical(y_bigs,num_classes=10)
y_smalls_encoded = keras.utils.to_categorical(y_smalls,num_classes=10)

# train test split data
X_bigs_train, X_bigs_test, y_bigs_train, y_bigs_test = train_test_split(X_bigs, y_bigs_encoded)
X_smalls_train, X_smalls_test, y_smalls_train, y_smalls_test = train_test_split(X_smalls, y_smalls_encoded)


def train_model(X_train, y_train, X_test, y_test):
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
    model.fit(X_train, y_train, epochs=50, batch_size=400)
    # evaluate accuracy on test data
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    print(loss_and_metrics)
    return model

print("\nBigs Model Training:")
bigs_model = train_model(X_bigs_train, y_bigs_train, X_bigs_test, y_bigs_test)
print("\nSmalls Model Training:")
smalls_model = train_model(X_smalls_train, y_smalls_train, X_smalls_test, y_smalls_test)

print(f"Big {bigs_model}\nSmall {smalls_model}")

def test_models(IMAGE_INDEX):
    # show sample image
    sample_image = (X_train.loc[IMAGE_INDEX])
    plt.imshow(sample_image.reshape(28,28))
    # find predictions for each model
    bigs_preds = (bigs_model.predict(np.expand_dims(sample_image,axis=0)))
    smalls_preds = (smalls_model.predict(np.expand_dims(sample_image,axis=0)))
    # plot predicitons
    plt.plot(bigs_preds[0])
    plt.plot(smalls_preds[0])
