from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer

MODEL_CONFIGS = [
    # === Required options ===
    "fcnet_hiddens",  # Number of hidden layers
    # === Other options ===
    "fcnet_activation",  # Activation type
]

class FullyConnectedNetwork(Model):
    """Generic fully connected network."""

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
        is_static = options.get("is_static", False)

        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing fcnet {} {}".format(hiddens, activation))

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            if is_static:
                weights = options["static_weights"]
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
