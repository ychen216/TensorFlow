import tensorflow as tf
from dataset import input_fn


def bulid_model(features, params):
    # input layer
    layer = tf.feature_column.input_layer(features, feature_columns=params['feature_columns'])
    # hidden layers
    for units in params['units']:
        layer = tf.layers.dense(layer, units=units, activation=tf.nn.relu)
    # output layer
    logits = tf.layers.dense(layer, 1, activation=tf.nn.sigmoid)
    return tf.reshape(logits, (-1,))


def model_fn(features, labels, mode, params):

    logits = bulid_model(features, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        op = tf.train.AdamOptimizer().minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=op)

def main():
    feature_columns = [tf.feature_column.numeric_column(key='CC_pctr_list'), ]
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./checkpoint',
        params={
            'units': [20, 10],
            'feature_columns': feature_columns,
        }
    )
    train_result = model.train(input_fn=input_fn)
    print(train_result)


if __name__ == '__main__':
    main()