import tensorflow as tf




def model_fn(features, labels, mode, params):

    logits = bulid_model(features, params)
    labels = tf.reshape(features['Cis_click_reward'], (-1, 1))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

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
        model_dir='./checkpoint',
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