import torch.nn as nn
import torch.nn.functional as F
from models.TDNN import TDNN
from models.Pooling import StatsPooling, AttnPooling


class Xvector(nn.Module):
    def __init__(self, feat_dim, num_spks, dropout=0., extract=False):
        super(Xvector, self).__init__()
        # Frame-level
        self.tdnn1 = TDNN(feat_dim, 512, 5, 1, 2, dropout)
        self.tdnn2 = TDNN(512, 512, 3, 2, 2, dropout)
        self.tdnn3 = TDNN(512, 512, 3, 3, 3, dropout)
        self.tdnn4 = TDNN(512, 512, 1, 1, 0, dropout)
        self.tdnn5 = TDNN(512, 1500, 1, 1, 0, dropout)

        # Statistics pooling layer
        self.pooling = StatsPooling()

        # Segment-level
        self.affine6 = nn.Linear(2 * 1500, 512)
        self.batchnorm6 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
                                         affine=False)
        self.affine7 = nn.Linear(512, 512)
        self.batchnorm7 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
                                         affine=False)
        self.output = nn.Linear(512, num_spks)
        self.output.weight.data.fill_(0.)

        self.dropout = nn.Dropout(p=dropout)
        self.extract = extract

    def forward(self, x):
        # Frame-level
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # Statistics pooling layer
        x = self.pooling(x, 2)

        # Segment-level
        x = self.affine6(x)
        if self.extract:
            return x
        x = self.dropout(self.batchnorm6(F.relu(x)))
        x = self.dropout(self.batchnorm7(F.relu(self.affine7(x))))
        x = self.output(x)
        return x


class Xvector_AttnPooling(nn.Module):
    def __init__(self, feat_dim, num_spks, dropout=0., extract=False):
        super(Xvector_AttnPooling, self).__init__()
        # Frame-level
        self.tdnn1 = TDNN(feat_dim, 512, 5, 1, 2, dropout)
        self.tdnn2 = TDNN(512, 512, 3, 2, 2, dropout)
        self.tdnn3 = TDNN(512, 512, 3, 3, 3, dropout)
        self.tdnn4 = TDNN(512, 512, 1, 1, 0, dropout)
        self.tdnn5 = TDNN(512, 1500, 1, 1, 0, dropout)

        # Self-attention pooling layer
        self.pooling = AttnPooling(1500)

        # Segment-level
        self.affine6 = nn.Linear(2 * 1500, 512)
        self.batchnorm6 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
                                         affine=False)
        self.affine7 = nn.Linear(512, 512)
        self.batchnorm7 = nn.BatchNorm1d(512, eps=0.001, momentum=0.99,
                                         affine=False)
        self.output = nn.Linear(512, num_spks)
        self.output.weight.data.fill_(0.)

        self.dropout = nn.Dropout(p=dropout)
        self.extract = extract

    def forward(self, x):
        # Frame-level
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # Self-attention pooling layer
        x = self.pooling(x, 2)

        # Segment-level
        x = self.affine6(x)
        if self.extract:
            return x
        x = self.dropout(self.batchnorm6(F.relu(x)))
        x = self.dropout(self.batchnorm7(F.relu(self.affine7(x))))
        x = self.output(x)
        return x
