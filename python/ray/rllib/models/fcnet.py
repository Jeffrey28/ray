from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer

USER_DATA_CONFIGS = [
    "fcnet_tag",  # Optional tag for fcnets to allow for more than one
]


class FullyConnectedNetwork(Model):
    """Generic fully connected network."""

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing fcnet {} {}".format(hiddens, activation))

        user_data = options.get("user_data", {})
        for k in user_data.keys():
            if k not in USER_DATA_CONFIGS:
                raise Exception(
                    "Unknown config key `{}`, all keys: {}".format(k,
                                                            USER_DATA_CONFIGS))
        fcnet_tag = user_data.get("fcnet_tag", None)

        singular = fcnet_tag is None
        with tf.name_scope("fc_net"):
            i = 1
            last_layer = inputs
            for size in hiddens:
                label = ("fc{}" if singular else "fc{}_{}").format(fcnet_tag, i)
                last_layer = slim.fully_connected(
                    last_layer, size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    scope=label)
                i += 1
            label = "fc_out" if singular else "fc_out_{}".format(fcnet_tag, i)
            output = slim.fully_connected(
                last_layer, num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None, scope=label)
            return output, last_layer
