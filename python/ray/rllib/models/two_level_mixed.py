from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.model import Model
from ray.rllib.models.fcnet import FullyConnectedNetwork

MODEL_CONFIGS = [
    # === Required options ===
    "num_subpolicies",  # Number of subpolicies in two-level fcnet
    "hierarchical_fcnet_hiddens",  # Num. of hidden layers for two-level fcnet
    # === Other options ===
    "switching_fcnet_hiddens",  # Number of hidden layers for switching network
    # function which maps from observation to subpolicy observation
    "fn_subpolicy_state",
    # function which maps from observation to choice of subpolicy
    "fn_choose_subpolicy",
]


NUM_SUBPOLICIES = 2

class TwoLevelMixedNetwork(Model):
    """
    Two-level fully mixed network, consisting of a static, pre-trained policy
    and a trainable policy
    """
    def _init(self, inputs, num_outputs, options):
        custom_options = options["custom_options"]
        trainable_hiddens = custom_options.get("trainable_hiddens", [16, 16])
        pre_trained_weights = custom_options["pre_trained_weights"]

        # attention specified by the first observation
        attention = tf.one_hot(inputs[0], NUM_SUBPOLICIES)

        outputs = []
        # pre-trained network
        with tf.variable_scope("multi{}".format(0)):
            sub_options = options.copy()
            sub_options["is_static"] = True
            sub_options["static_weights"] = pre_trained_weights
            subinput = inputs[1:]
            fcnet = FullyConnectedNetwork(subinput, num_outputs, sub_options)
            output = fcnet.outputs
            rep_attention = tf.reshape(tf.tile(attention[:, 0],
                                      [num_outputs]),
                                      [-1, num_outputs])
        outputs.append(rep_attention * output)

        # trainable network
        with tf.variable_scope("multi{}".format(1)):
            sub_options = options.copy()
            sub_options.update({"fcnet_hiddens": trainable_hiddens})
            subinput = inputs[1:]
            fcnet = FullyConnectedNetwork(subinput, num_outputs, sub_options)
            output = fcnet.outputs
            rep_attention = tf.reshape(tf.tile(attention[:, 1],
                                      [num_outputs]),
                                      [-1, num_outputs])
        outputs.append(rep_attention * output)
        overall_output = tf.add_n(outputs)

        return overall_output, outputs
