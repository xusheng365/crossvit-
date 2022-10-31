import mindspore.nn as nn


class Identity(nn.Cell):
    def construct(self, x):
        return x
