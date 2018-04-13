from torch import nn
from torch.nn import functional as F
import numpy as np
from . import FLAGS
import torch
from ipdb import set_trace


def _get_padding(padding_type, kernel_size):
    assert padding_type in ['SAME', 'VALID']
    if padding_type == 'SAME':
        padding = tuple([(k - 1) // 2 for k in range(kernel_size)])
    else:
        padding = tuple([0 for _ in range(kernel_size)])
    return padding


def _conv_branch(kernel_size, count, out_filters,
                 ch_mul=1, separable=False, stride=1):
    count = int(count.numpy())

    layers = [
        nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=1),
        nn.BatchNorm2d(out_filters),
        nn.ReLU()
    ]

    # there was original a difference between fixed_arc and sample_arc, but I
    # dont see it besides some batch_node stuff. so I'll ignore it for now
    for _ in range(count):
        if separable:
            layers.extend([
                nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=1,
                          groups=out_filters, stride=stride),
                nn.BatchNorm2d(out_filters),
                nn.ReLU()
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(out_filters),
                nn.ReLU()
            ])

    return nn.Sequential(*layers)


def _enas_layer(sample_arc, layer_id, prev_layers, start_idx, out_filters):
    num_branches = FLAGS.NUM_BRANCHES
    count = sample_arc[start_idx:start_idx + (2*num_branches)]
    branches = []

    branches.append(_conv_branch( kernel_size=3, count=count[1], out_filters=out_filters))

    if num_branches > 1:
        branches.append(_conv_branch(kernel_size=3, count=count[3], out_filters=out_filters,
                                     separable=True))

    if num_branches > 2:
        branches.append(_conv_branch(
            kernel_size=3, count=count[5], out_filters=out_filters))

    if num_branches > 3:
        branches.append(_conv_branch(kernel_size=3, count=count[7], out_filters=out_filters,
                                     separable=True))

    if num_branches > 4:
        branches.append(_conv_branch(kernel_size=3, count=count[9], out_filters=out_filters,
                                     separable=False, stride=2))

    if num_branches > 5:
        branches.append(_conv_branch(
            kernel_size=3, count=count[11], out_filters=out_filters, stride=2))

    # I dont really get what the final_conv block is doing
    # looks like it is masking certain things

    # this might be wrong but we'll figure it out
    out = nn.Sequential(*[
        nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=1),
        nn.BatchNorm2d(out_filters),
        nn.ReLU()
    ])

    branches.append(out)
    final_conv = nn.Sequential(*branches)

    return final_conv


def build_model(sample_arc, fixed_arc=False):
    # so basically this builds a model and immediately runs it
    # ideally what I would like is build a model, return it, and run it with fastai
    if not fixed_arc:
        sample_arc = sample_arc
    else:
        sample_arc = np.array([int(x) for x in sample_arc.split(" ") if x])

    layers = []

    out_filters = FLAGS.OUT_FILTERS

    stem_conv = nn.Sequential(*[
        nn.Conv2d(in_channels=FLAGS.INPUT_CHANNELS, out_channels=out_filters, kernel_size=3,
                  padding=1),
        nn.BatchNorm2d(out_filters),
        nn.ReLU()  # added this
    ])

    num_branches = 3  # FLAGS.NUM_BRANCHES
    num_layers = 5  # FLAGS.NUM_LAYERS

    layers.append(stem_conv)

    start_idx = num_branches
    # pool_distance = num_layers // 3
    # pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

    for layer_id in range(num_layers):
        if fixed_arc:
            raise NotImplementedError
        else:
            # filters = 10 if layer_id == num_layers-1 else out_filters
            x = _enas_layer(sample_arc, layer_id, layers,
                            start_idx, out_filters)
        layers.append(x)
        # going to ignore the factorized_reduction stuff for now, a bit overly complicated
#         if layer_id in pool_layers:
#             if not fixed_arc:
#                 out_filters *= 2
#             pooled_layers = []
#             for i, layer in enumerate(layers):
#                 x = _factorized_reduction(layer, out_filters, 2)
#                 pooled_layers.append(x)

#             layers = pooled_layers
        start_idx += (2*num_branches) + layer_id


    class ConstructedModel(nn.Module):
        def __init__(self):
            super(ConstructedModel, self).__init__()
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(FLAGS.R*FLAGS.C*out_filters, FLAGS.NUM_CLASSES)

        def forward(self, x):
            batch_size = x.shape[0]
            x = self.body(x)
            out = self.head(x.view(batch_size, -1))

            return F.log_softmax(out, dim=1)

    class FastaiWrapper():
        def __init__(self):
            self.model = ConstructedModel().double()
            #fix this
            self.crit = F.nll_loss
        
        def get_layer_groups(self):
            return self.model

    return FastaiWrapper()
