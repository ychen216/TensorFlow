# coding=utf-8
import tensorflow as tf
import numpy as np

train_data = np.random.random((1000, 32))
train_label = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_label = np.random.random((100, 10))

'''
# 层模型 Sequential()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
model.fit(train_data, train_label, batch_size=32, epochs=10,
         validation_data=(val_data, val_label))

# The Estimators API is used for training models for distributed environments
# estimator = tf.keras.estimator.model_to_estimator(model)
# estimator.train(input_fn=input_fn, step=10)
print('-----------')
model.evaluate(val_data, val_label, batch_size=32 )

print(model.predict(val_data, batch_size=32))

model.save('./model.h5')

'''

'''
# 自定义模型流

input = tf.keras.Input(shape=[32, ])

x = tf.keras.layers.Dense(64, activation='relu')(input)
x = tf.keras.layers.Dense(64, activation='relu')(x)
prediction = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=prediction)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy, ])
model.fit(train_data, train_label, batch_size=32, epochs=5)

'''
model = tf.keras.Sequential([tf.keras.layers.Dense(10,activation='softmax'),
                          tf.keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])





