import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_PATH = 'data.json'


def load_data(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return X, y


def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATA_PATH)

    # create the train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create the train validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # 3d array for each sample - convert (130, 13) to (130, 13, 1)
    X_train = X_train[..., np.newaxis]   # 4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    # create a model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #  flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):

    X = X[np.newaxis, ...]  # X should be a 4d array, so add an axis
    prediction = model.predict(X)  # predictions will be a 1d array of size 10
    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # output will be a 1d array of size 1
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))



if __name__ == '__main__':
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train the CNN
    model.fit(X_train, y_train,
              validation_data=(X_validation, y_validation),
              batch_size=64,
              epochs=60)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    print("Error on test set is: {}".format(test_error))

    # make prediction on a sample
    X = X_test[1]
    y = y_test[1]
    predict(model, X, y)
