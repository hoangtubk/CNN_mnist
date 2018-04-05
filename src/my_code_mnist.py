"""
 * Created by PyCharm.
 * User: tuhoangbk
 * Date: 05/04/2018
 * Time: 10:37
 * Have a nice day　:*)　:*)
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_data_mnist():
    tensor_shap = [-1, 28, 28, 1]
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    X_train = mnist.train.images
    X_train = tf.reshape(X_train, tensor_shap)
    y_train = mnist.train.labels

    X_valid = mnist.validation.images
    X_valid = tf.reshape(X_valid, tensor_shap)
    y_valid = mnist.validation.labels

    X_test = mnist.test.images
    X_test = tf.reshape(X_test, tensor_shap)
    y_test = mnist.train.labels

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def build_model(features, labels, mode):
    conv1 = tf.layers.conv2d(inputs=features,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)
    pooling1 = tf.layers.max_pooling2d(inputs=conv1,
                                       pool_size=[2, 2],
                                       strides=2)
    conv2 = tf.layers.conv2d(inputs=pooling1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)
    pooling2 = tf.layers.max_pooling2d(inputs=conv2,
                                       pool_size=[2, 2],
                                       strides=2)
    pooling2_flat = tf.reshape(pooling2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pooling2_flat,
                            units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout,
                             units=10,
                             activation=tf.nn.relu)
    # output = tf.nn.softmax(logits=dense2)
    logits = dense2
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return None

def main(unused_argv):
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_mnist()
    mnist_classifier = tf.estimator.Estimator(model_fn=build_model,
                                              model_dir="/tmp/mnist_convnet_model")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=50)
    # Train the model
    print(X_train.shape)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X_train},
                                                        y=y_train,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)
    print(train_input_fn)

    mnist_classifier.train(input_fn=train_input_fn,
                           steps=20,
                           hooks=[logging_hook])
if __name__ == "__main__":
    tf.app.run(main)