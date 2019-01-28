from modules import *
from loss import *
from optims import *
from dataset import *
from modules.config import *
import numpy as np
import argparse
import torch
from functools import partial

dataset = get_dataset('long')
train_iter = dataset(mode='train', batch_size=128,
                     device='cpu')

for batch in train_iter:
    print(batch)
    break
