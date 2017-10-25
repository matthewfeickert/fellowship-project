"""
Example toy model in TensorFlow that demonstrates how to start
writing a class

Closely follows Danijar Hafner's blog post:
https://danijar.com/structuring-your-tensorflow-models/
"""

# Following TensorFlow Style Guide
# https://www.tensorflow.org/community/style_guide
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def lazy_property(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    # Use property as a decorator for lazy-loading
    # If not familiar with @property c.f.:
    # https://docs.python.org/3/library/functions.html#property
    # https://dbader.org/blog/python-decorators
    # https://www.programiz.com/python-programming/property
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Model(object):
    """
    Toy model
    """

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias

        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        log_prob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.target * log_prob)
        optimizer = tf.train.RMSPropOptimizer(0.03)

        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1),
                                tf.argmax(self.prediction, 1))

        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(10):
            images, labels = mnist.test.images, mnist.test.labels
            error = sess.run(model.error, {image: images, label: labels})
            print('Test error: {:6.2f}%'.format(error * 100))
            for _ in range(60):
                images, labels = mnist.train.next_batch(100)
                sess.run(model.optimize, {image: images, label: labels})

if __name__ == '__main__':
    main()
