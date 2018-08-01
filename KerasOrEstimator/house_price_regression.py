import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

house_data = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = house_data.load_data()

# data preprocessing
# shuffle data
order = np.argsort(np.random.random(train_labels.shape), axis=0)
train_data = train_data[order]
train_labels = train_labels[order]

# normalize ps:TensorFlow data is *not* used when calculating the mean and std.

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
# print(mean, std)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1),
])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), loss=keras.losses.mean_squared_error,
              metrics=[keras.metrics.mean_absolute_error])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.fit(train_data, train_labels, epochs=500, validation_split=20, callbacks=[early_stop])
eval_loss, eval_mae = model.evaluate(test_data, test_labels)
print('Test loss(mse) && metrics(mae):', eval_loss, eval_mae)