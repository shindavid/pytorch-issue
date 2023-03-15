"""
python demo.py model.pt
"""
import random
import sys

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(linewidth=200)
torch.use_deterministic_algorithms(True)

filename = sys.argv[1]
print('Testing: ' + filename)
net = torch.jit.load(filename)
net.to('cuda')
net.eval()
torch.set_grad_enabled(False)


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
