import tensorflow as tf
import os

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10Dataset(object):
    """ Cifar10 DateSet """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self._data_dir = data_dir
        self._subset = subset
        self._use_distortion = use_distortion

    def get_filenames(self):
        if self._subset in ['train', 'eval', 'validation']:
            return [os.path.join(self._data_dir, self._subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self._subset)

    def preprocess(self, image):
        if self._subset == 'train' and self._use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image

    def parse(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.decode_raw(features['image'], tf.int8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])
        image = tf.cast(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), tf.float32)
        label = tf.cast(features['label'], tf.int32)
        if self._subset == 'train' and self._use_distortion:
            image = self.preprocess(image)
        return image, label

    def make_batch(self, batch_size=32):
        filenames = self.get_filenames()
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        dataset = dataset.map(self.parse, num_parallel_calls=batch_size)
        if self._subset == 'train':
            min_queue_examples = int(
                Cifar10Dataset.num_examples_per_epoch(self._subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.next()
        return images, labels

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 45000
        elif subset == 'validation':
            return 5000
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)