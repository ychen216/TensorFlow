# coding=utf-8
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

'''
text reviews classification : binary-classification positive && negative
'''
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data[0])
# train_data[0] like [2,23,4,1,34,...,...]  and  length of data is different
# need to preprocess: padding to same length tensor

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, padding='post', value=word_index["<PAD>"])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, padding='post', value=word_index["<PAD>"])
# print(train_data[0])

vocab_size = 10000
embedding_dim = 6
model = keras.Sequential(
    [
        keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

model.compile(optimizer=tf.train.AdamOptimizer(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=40, validation_data=(x_val, y_val))
# model.fit() returns a History object that contains a dictionary with everything that happened during training

result = model.evaluate(test_data, test_labels)
print('TensorFlow loss && accuracy:', result)

# print(history, history.history)

# plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()