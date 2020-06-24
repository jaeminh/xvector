import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, din, dout, kernel, dilation, padding, dropout_p):
        super(TDNN, self).__init__()
        self.affine = nn.Conv1d(din, dout, kernel, 1, padding, dilation,
                                padding_mode='replicate')
        self.batchnorm = nn.BatchNorm1d(dout, eps=1e-3, momentum=0.99,
                                        affine=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        return self.dropout(self.batchnorm(F.relu(self.affine(x))))
