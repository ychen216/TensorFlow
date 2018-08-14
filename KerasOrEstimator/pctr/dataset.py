from __future__ import print_function
from __future__ import division
import tensorflow as tf

file_list = ['hdfs://default/home/rl/reco/samples/20180808/0000/nonterminal.tfrecords/part-r-00070', ]


def input_fn():
    dataset = tf.data.TFRecordDataset(filenames=file_list)

    def parse_example(serialized_example):
        feature = tf.parse_single_example(
            serialized_example,
            features={
                'Cis_click_reward': tf.FixedLenFeature([], tf.float32),
                'CC_pctr_list': tf.FixedLenFeature([], tf.float32),
            }
        )
        # pctr = feature['CC_pctr_list']
        #
        # label = feature['Cis_click_reward']
        return feature, feature['Cis_click_reward']

    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size=32).make_one_shot_iterator()
    return dataset.get_next()
