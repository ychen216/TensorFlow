import tensorflow as tf

file_list = ['hdfs://default/home/rl/reco/samples/20180808/0000/nonterminal.tfrecords/part-r-00070', ]
feature_columns = [tf.feature_column.numeric_column(key='CC_pctr_list'), ]


def input_fn():
    dataset = tf.data.TFRecordDataset(filenames=file_list)
    # tf.parse_example 接收的是是batch_examples 所以必须先对dataset做batch操作
    dataset = dataset.shuffle(1000).repeat().batch(batch_size=32)
    def parse_example(serialized_example):
        feature = tf.parse_example(
            serialized_example,
            features=tf.feature_column.make_parse_example_spec(feature_columns)
        )
        # pctr = feature['CC_pctr_list']
        #
        # label = feature['Cis_click_reward']
        return feature

    dataset = dataset.map(parse_example)
    dataset = dataset.make_one_shot_iterator()
    return dataset.get_next()

def bulid_model(features, params):
    # input layer
    layer = tf.feature_column.input_layer(features, feature_columns=params['feature_columns'])
    # hidden layers
    for units in params['units']:
        layer = tf.layers.dense(layer, units=units, activation=tf.nn.relu)
    # output layer
    logits = tf.layers.dense(layer, 1, activation=None)
    return logits


def model_fn(features, mode, params):

    logits = bulid_model(features, params)
    labels = tf.zeros((32,1),dtype=tf.int32)
    print(features,labels,logits)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    print(loss)
    if mode == tf.estimator.ModeKeys.TRAIN:

        op = tf.train.AdamOptimizer().minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=op)

    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=logits)
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def main():

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'units': [20, 10],
            'feature_columns': feature_columns,
        }
    )
    model.train(input_fn=input_fn, steps=1000)
    print('train end')
    eval_r = model.evaluate(input_fn=input_fn, steps=1000)
    print('eval: ', eval_r)


if __name__ == '__main__':
    main()
