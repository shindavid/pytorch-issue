"""
python full_demo.py <N>

where N is the number of residual block layers to add to the network.
"""
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


num_residual_blocks = 19
if len(sys.argv) > 1:
    num_residual_blocks = int(sys.argv[1])


print(f'Full demo with {num_residual_blocks} residual blocks')


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(linewidth=200)
torch.use_deterministic_algorithms(True)


class ConvBlock(nn.Module):
    def __init__(self, n_input_channels: int, n_conv_filters: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        return F.relu(self.batch(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, n_conv_filters: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(n_conv_filters)
        self.conv2 = nn.Conv2d(n_conv_filters, n_conv_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(n_conv_filters)

    def forward(self, x):
        identity = x
        out = F.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        out += identity  # skip connection
        return F.relu(out)


class PolicyHead(nn.Module):
    def __init__(self, n_input_channels: int):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(n_input_channels, 2, kernel_size=1, stride=1, bias=False)
        self.batch = nn.BatchNorm2d(2)
        self.linear = nn.Linear(84, 7)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.relu(x)
        x = x.view(-1, 84)
        x = self.linear(x)
        return x


class Net(nn.Module):
    def __init__(self, n_res_blocks=19, n_conv_filters=64):
        super(Net, self).__init__()
        self.n_conv_filters = n_conv_filters
        self.n_res_blocks = n_res_blocks
        self.conv_block = ConvBlock(2, n_conv_filters)
        self.res_blocks = nn.ModuleList([ResBlock(n_conv_filters) for _ in range(n_res_blocks)])
        self.policy_head = PolicyHead(n_conv_filters)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x),


net = Net(num_residual_blocks).cuda().train()
for _ in range(100):
    net(torch.randn(128, 2, 7, 6).cuda())

torch.set_grad_enabled(False)
net.eval()


def get_output(batch_size):
    input_tensor = torch.zeros((batch_size, 2, 7, 6)).to('cuda', non_blocking=True)
    output_tuple = net(input_tensor)
    output_tensor = output_tuple[0]
    return output_tensor[:1].to('cpu')


out1 = get_output(1)
failed = False
for b in range(2, 64):
    out = get_output(b)
    if torch.all(out == out1):
        pass
    else:
        failed = True
        print('Batch size {} is NOT OK. Diffs: {}'.format(b, out - out1))


if not failed:
    print('All ok!')
