# coding=utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.gca().grid(False)
# plt.show()

# preprocess data scalar
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt to verify data
# plt.figure(figsize=[10, 10])
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.grid('off')
#     plt.imshow(train_images[i], cmap=plt.cm.get_cmap('gray_r'))
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

classifier = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]), # 将image展平成28*28的一维向量
        keras.layers.Dense(128, activation=tf.nn.relu),
        # 可以用dropout或者regularization的方式避免过拟合
        # keras.layers.Dropout(rate=0.5),
        # keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

classifier.compile(
    optimizer=tf.train.AdagradOptimizer(0.01),
    loss=keras.losses.sparse_categorical_crossentropy, # 允许label是稀疏表示 eg. 0 1 2 3
    metrics=['accuracy', ],
)

classifier.fit(train_images, train_labels, batch_size=32, epochs=10)

test_loss, test_acc = classifier.evaluate(test_images, test_labels, batch_size=32)
print('TensorFlow accuray:', test_acc)
predictions = classifier.predict(test_images, batch_size=32)

plt.figure(figsize=[15, 15])
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i], cmap=plt.get_cmap('gray_r'))
    predict_label = np.argmax(predictions[i])
    if predict_label == test_labels[i]:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predict_label], class_names[test_labels[i]]), color=color)
plt.show()
