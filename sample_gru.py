import os
from collections import defaultdict
from torch import randperm
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch._utils import _accumulate
from dataset import TwitterFileArchiveDataset
from gru import GRUCell

rnn = nn.GRU(10,20,2)
input1 = torch.autograd.Variable(torch.randn(5,3,10))
h0 = torch.randn(2,3,20)
output , hn = rnn(input1,h0)
