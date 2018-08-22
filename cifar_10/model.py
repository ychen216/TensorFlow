# coding=utf-8
from __future__ import division
import tensorflow as tf
import tensorflow.contrib as contrib
import functools
from basic_model import ResNet
from input import Cifar10Dataset


class Cifar10Model(ResNet):
    """ Cifar10Model using ResNet """

    def __init__(self, class_num, batch_norm_epsilon, batch_norm_decay, filter_list, stride_list, residual_block,
                 per_num_in_block, is_training, data_format='channels_last'):
        """
        :param class_num: 分类类别数
        :param batch_norm_epsilon:
        :param batch_norm_decay:
        :param filter_list: 每一个block第一层filter数目
        :param stride_list: 每一个block第一层stride数目
        :param residual_block: residual block个数
        :param per_num_in_block: 每一个block中residual结构数目
        :param is_training:
        :param data_format:
        """
        super(Cifar10Model, self).__init__(is_training=is_training, data_format=data_format,
                                           batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon)
        self._class_num = class_num
        self._residual_block = residual_block
        assert len(filter_list) == residual_block + 1 and len(stride_list) == residual_block
        """
        Example:
        filter_list = [16,16,32,64]
        stride_list = [1,2,2]
        filter_list[i]是第i个block的in_filter数目,filter_list[i+1]是第i个block的out_filter数目
        """
        self._filter_list = filter_list
        self._stride_list = stride_list
        self._per_num_in_block = per_num_in_block

    def forward_pass(self, x):

        # image standardization
        x = x / 128 - 1
        # CNN Block
        with tf.name_scope('single_cnn') as name_scope:
            filter_num = self._filter_list[0] if len(self._filter_list) > 0 else 16
            x = self._cnn(x, filters=filter_num, kernel_size=3, strides=1)
            x = self._batch_norm(x)
            x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, 0.5)

        tf.logging.info('image after %s unit: %s', name_scope, x.get_shape())

        res_func = self._residual_v1

        # residual block stage
        for i in range(self._residual_block):
            with tf.name_scope('residual_block_stage'):
                for j in range(self._per_num_in_block):
                    if j == 0:
                        # First block in a stage, filters and strides may change.
                        x = res_func(x, 3, self._filter_list[i], self._filter_list[i+1], self._stride_list[i])
                    else:
                        # Following blocks in a stage, constant filters and unit stride.
                        x = res_func(x, 3, self._filter_list[i+1], self._filter_list[i+1], 1)

        x = self._global_avg_pool(x)
        x = self._fully_connected(x, self._class_num)
        return x


def model_fn(features, labels, mode, params):
    # 这里的参数有些应放在params 为了节省时间直接写死
    model = Cifar10Model(class_num=10, batch_norm_epsilon=0.001, batch_norm_decay=0.99, filter_list=[16, 16, 32, 64],
                         stride_list=[1, 2, 2], residual_block=3, per_num_in_block=2, is_training=True, data_format='channels_last')

    logits = model.forward_pass(features)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss, global_step=tf.train.get_or_create_global_step())
        # 使用contrib.layers.optimize_loss有更多功能 且能方便在tensorboard中观察各种权值
        op = contrib.layers.optimize_loss(
            loss=loss,
            global_step=contrib.framework.get_or_create_global_step(),
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9),
            learning_rate=0.01,
            # summaries=['learning_rate', 'gradients', 'loss'],
        )

        tensor_to_log = {'loss': loss}
        log_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)
        train_hook = [log_hook, ]
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=op, training_hooks=train_hook)

    metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions['classes'])
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def input_fn(data_dir, batch_size, subset, use_distortion):
    dataset = Cifar10Dataset(data_dir=data_dir, subset=subset, use_distortion=use_distortion)
    return dataset.make_batch(batch_size)


def main():
    data_dir = './cifar-10-data'
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir='./model')
    train_input_fn = functools.partial(input_fn, data_dir=data_dir, batch_size=32, subset='train', use_distortion=True)
    eval_input_fn = functools.partial(input_fn, data_dir=data_dir, batch_size=32, subset='eval', use_distortion=False)
    pred_input_fn = functools.partial(input_fn, data_dir=data_dir, batch_size=32, subset='eval', use_distortion=False)

    print('training...')
    model.train(input_fn=train_input_fn, max_steps=1000)

    print('evalation...')
    eval = model.evaluate(input_fn=eval_input_fn, steps=1000)
    print('eval', eval)

    print('prediction...')
    # predict is a generator
    prediction = model.predict(input_fn=pred_input_fn,)
    print('predict 10 examples')
    for i in range(10):
        print('item {}: class:{}'.format(i, next(prediction)['classes']))


main()