from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.model import Model
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.models.two_level_fcnet import TwoLevelFCNetwork
from ray.rllib.models.fcnet import MODEL_CONFIGS as FCN_CONFIGS
from ray.rllib.models.two_level_fcnet import MODEL_CONFIGS as TFCN_CONFIGS
from ray.rllib.models.action_dist import Reshaper

# TODO(cathywu) support more network types
SUBMODEL_CONFIGS = {"FullyConnectedNetwork": FCN_CONFIGS,
                 "TwoLevelFCNetwork": TFCN_CONFIGS}


class MultiAgentFullyConnectedNetwork(Model):
    """Multiagent fully connected network."""

    def _init(self, inputs, num_outputs, options):
        # Split the input and output tensors
        input_shapes = options["custom_options"]["multiagent_obs_shapes"]
        output_shapes = options["custom_options"]["multiagent_act_shapes"]
        input_reshaper = Reshaper(input_shapes)
        output_reshaper = Reshaper(output_shapes)
        split_inputs = input_reshaper.split_tensor(inputs)
        num_actions = output_reshaper.split_number(num_outputs)

        custom_options = options["custom_options"]
        hiddens = custom_options.get("multiagent_hiddens",
                                     [[256, 256]] * 1)

        network_cls = globals()[custom_options.get("network_type", "FullyConnectedNetwork")]
        submodel_configs = SUBMODEL_CONFIGS[custom_options.get("network_type", "FullyConnectedNetwork")]

        # check for a shared model
        shared_model = custom_options.get("multiagent_shared_model", 0)
        reuse = tf.AUTO_REUSE if shared_model else False
        outputs = []
        for i in range(len(hiddens)):
            with tf.variable_scope("multi{}".format(i), reuse=reuse):
                sub_options = options.copy()
                for c in submodel_configs:
                    if c in options:
                        sub_options.update({c: options[c]})
                    if c in custom_options:
                        sub_options.update({c: custom_options[c]})
                if "hierarchical_fcnet_hiddens" in sub_options:
                    sub_options.update({"hierarchical_fcnet_hiddens": hiddens[i]})
                else:
                    sub_options.update({"fcnet_hiddens": hiddens[i]})
                fcnet = network_cls(split_inputs[i], int(num_actions[i]),
                                    sub_options)
                output = fcnet.outputs
                outputs.append(output)
        overall_output = tf.concat(outputs, axis=1)
        return overall_output, outputs