import numpy as np
from data import Data
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def main():

    """ Main
    """

    labels, data = [], []

    # Read data from pickle (grayscale 50x50px)
    d = Data()
    labels, data = d.read_data_from_pickle()

    # Split data into train and test
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=0)

    # Convert class labels to one-hot labels
    labels_train = to_categorical(labels_train)
    labels_test = to_categorical(labels_test)

    # Cast train- and test-data to float values
    data_train = np.array(data_train, dtype=np.float32)
    data_test = np.array(data_test, dtype=np.float32)

    # Normalize data from [0;255] to [0;1]
    data_train /= 255
    data_test /= 255

    # Split train-data for validation (size of validation-data equals size of test-data)
    data_train, data_valid, labels_train, labels_valid = train_test_split(data_train, labels_train, test_size=0.20)

    # Initialize sequential model
    model = Sequential()

    model.add(Conv2D(filters=120, kernel_size=3, activation="relu", input_shape=(100, 100, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))
   
    model.add(Conv2D(filters=60, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=30, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(units=121, activation="softmax"))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

    model.fit(x=data_train, y=labels_train, validation_data=(data_valid, labels_valid), epochs=20, batch_size=120)
    score = model.evaluate(x=data_test, y=labels_test)
    # Print test classification results
    print("Test-accuracy: " + str(round(score[1], 3) * 100) + "%\n" + "Test-loss: " + str((score[0])))

    # Save model to file
    model.save("model.h5")


if __name__ == "__main__":
    main()
