# coding=utf-8
import tensorflow as tf
import iris_data


def my_model(features, labels, mode, params):
    # input layer
    net = tf.feature_column.input_layer(features, feature_columns=params['feature_columns'])
    for units in params['hidden_units']:
        # 两种方式创建层 dense是快捷方式 可以直接创建和运行层 但不利于复用和自省
        # net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        # Dense应该以函数形式调用
        net = tf.layers.Dense(units=units, activation=tf.nn.relu)(net)

    # output
    # logits = tf.layers.dense(net, units=params['n_classes'], activation=None)
    logits = tf.layers.Dense(units=params['n_classes'], activation=None)(net)
    # prediction
    predict_class = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predict_class[:tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # metric
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict_class)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main():
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key))

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        },
        model_dir='./checkpoint',
    )
    # print dict(train_x)
    classifier.train(input_fn=lambda : iris_data.train_input_fn(train_x, train_y, 32), steps=100)
    eval = classifier.evaluate(input_fn=lambda : iris_data.eval_input_fn(test_x, test_y, 32))
    print(eval)
    classifier.predict(input_fn=lambda : iris_data.eval_input_fn(test_x, 32))


if __name__ == '__main__':
    main()