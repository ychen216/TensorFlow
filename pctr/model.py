# coding=utf-8
import tensorflow as tf

file_list = ['hdfs://default/home/rl/reco/samples/20180808/0000/nonterminal.tfrecords/part-r-00070', ]
# feature_columns including label
feature_columns = [tf.feature_column.numeric_column(key='CC_pctr_list'), tf.feature_column.numeric_column(key='Cis_click_reward'), ]
# feature_columns without label
feature_columns_nolabel = [tf.feature_column.numeric_column(key='CC_pctr_list'), ]


def input_fn():
    dataset = tf.data.TFRecordDataset(filenames=file_list)
    # tf.parse_example 接收的是是batch_serialized_examples 所以必须先对dataset做batch操作
    dataset = dataset.shuffle(500).repeat().batch(batch_size=32)

    def parse_example(serialized_example):
        feature = tf.parse_example(
            serialized_example,
            features=tf.feature_column.make_parse_example_spec(feature_columns)
        )
        # pctr = feature['CC_pctr_list']
        # label = feature['Cis_click_reward']

        # return features, labels 将label从feature里拿掉
        return feature, feature.pop('Cis_click_reward')

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
    logits = tf.layers.dense(layer, params['class_num'], activation=None)
    return logits


def model_fn(features, labels, mode, params):

    logits = bulid_model(features, params)

    # tf.losses.sparse_softmax_cross_entropy要求
    # labels :shape [batch_size] type=int
    # logits :shape [batch_size, class_num] type=float
    # 网络配合输出的unit个数应该为class_num
    # eg. 二分类 logits=[[0.1, 0.9],[0.7,0.3]] labels=[1,0]
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int32)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        op = tf.train.AdamOptimizer().minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=op)

    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, 1)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def main():

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'units': [20, 10],
            'feature_columns': feature_columns_nolabel,
            'class_num': 2,
        }
    )
    model.train(input_fn=input_fn, steps=1000)
    print('train end')
    eval_r = model.evaluate(input_fn=input_fn, steps=1000)
    print('eval: ', eval_r)
    # eval:  {'loss': 0.38704792, 'global_step': 1000, 'accuracy': 0.85796875


if __name__ == '__main__':
    main()
