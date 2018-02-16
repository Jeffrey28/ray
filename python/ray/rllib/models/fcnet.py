from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer

MODEL_CONFIGS = [
    # === Required options ===
    "fcnet_hiddens",  # Number of hidden layers
    # === Other options ===
    "fcnet_activation",  # Activation type
    "initial_weights",  # Initial Values of weights (for transfer learning)
]


class FullyConnectedNetwork(Model):
    """Generic fully connected network."""

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])

        weights = options.get("initial_weights", None)
        print(weights)

        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing fcnet {} {}".format(hiddens, activation))

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            if weights is not None:
                for _ in range(2):  # FIXME: hack
                    label = "fc{}".format(i)
                    size = np.asarray(
                        weights["tower/{}/weights".format(label)]).shape[1]
                    last_layer = slim.fully_connected(
                        last_layer, size,
                        weights_initializer=tf.constant_initializer(np.asarray(
                            weights["tower/{}/weights".format(label)],
                            dtype=np.float32)),
                        biases_initializer=tf.constant_initializer(np.asarray(
                            weights["tower/{}/biases".format(label)],
                            dtype=np.float32)),
                        activation_fn=activation,
                        scope=label)
                    i += 1
                label = "fc_out"
                output = slim.fully_connected(
                    last_layer, num_outputs,
                    weights_initializer=tf.constant_initializer(np.asarray(
                        weights["tower/fc_out/weights"],
                        dtype=np.float32)),
                    biases_initializer=tf.constant_initializer(np.asarray(
                        weights["tower/fc_out/biases"],
                        dtype=np.float32)),
                    activation_fn=None, scope=label)
            else:
                for size in hiddens:
                    label = "fc{}".format(i)
                    last_layer = slim.fully_connected(
                        last_layer, size,
                        weights_initializer=normc_initializer(1.0),
                        activation_fn=activation,
                        scope=label)
                    i += 1
                label = "fc_out"
                output = slim.fully_connected(
                    last_layer, num_outputs,
                    weights_initializer=normc_initializer(0.01),
                    activation_fn=None, scope=label)
            return output, last_layer


MODEL_CONFIGS_STATIC = [
    # === Required options ===
    "initial_weights",  # Values of weights (unchanging)
    # === Other options ===
    "fcnet_activation",  # Activation type
]


class StaticFullyConnectedNetwork(Model):
    """Non-trainable Fully connected network with pre-specified weights."""

    def _init(self, inputs, num_outputs, options):
        weights = options["static_weights"]

        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing static fcnet")

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for _ in range(2):  # FIXME: hack
                label = "fc{}".format(i)
                size = np.asarray(
                    weights["tower/{}/weights".format(label)]).size
                last_layer = slim.fully_connected(
                    last_layer, size,
                    weights_initializer=tf.constant_initializer(np.asarray(
                        weights["tower/{}/weights".format(label)],
                        dtype=np.float32)),
                    biases_initializer=tf.constant_initializer(np.asarray(
                        weights["tower/{}/biases".format(label)],
                        dtype=np.float32)),
                    activation_fn=activation,
                    scope=label,
                    trainable=False)
                i += 1
            label = "fc_out"
            output = slim.fully_connected(
                last_layer, num_outputs,
                weights_initializer=tf.constant_initializer(np.asarray(
                    weights["tower/fc_out/weights"],
                    dtype=np.float32)),
                biases_initializer=tf.constant_initializer(np.asarray(
                    weights["tower/fc_out/biases"],
                    dtype=np.float32)),
                activation_fn=None, scope=label,
                trainable=False)
            return output, last_layer
