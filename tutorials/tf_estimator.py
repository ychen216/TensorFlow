# coding=utf-8
import tensorflow as tf
import numpy as np

# feature 以字典形式存在
features = {'SepalLength': np.array([6.4, 5.0]),
            'SepalWidth':  np.array([2.8, 2.3]),
            'PetalLength': np.array([5.6, 3.3]),
            'PetalWidth':  np.array([2.2, 1.0])}
labels = np.array([2, 1])

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.1],
    'SepalWidth': [3.3, 3.0, 6.2],
    'PetalLength': [1.7, 4.2, 6.3],
    'PetalWidth': [0.5, 1.5, 6.4],
}

expected_labels = np.array([2, 0, 1])

# tf.data.Dataset类型的返回值的输入函数
# train和evaluation需要(feature，label)形式的数据
# predict需要feature形式的数据
def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(4)


def test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(predict_x), expected_labels))
    return dataset.batch(4)


def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(dict(predict_x))
    return dataset.batch(4)


def main():
    # 特征列
    feature_columns = []
    for key in features.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    # Estimator实例化
    classifier = tf.estimator.DNNClassifier(hidden_units=[10, 10], feature_columns=feature_columns, n_classes=3)
    classifier.train(input_fn=train_input_fn, steps=100)
    eval_result = classifier.evaluate(input_fn=test_input_fn)
    print("evaluation:", eval_result)
    predict = classifier.predict(input_fn=predict_input_fn)
    print("predict:", predict)


if __name__ == "__main__":
    main()