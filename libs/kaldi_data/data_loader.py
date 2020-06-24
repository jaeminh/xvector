import os
import torch
from torch.utils.data import Dataset
from kaldi_io import read_mat, read_mat_ark
from kaldi_data._io_kernel import read_nnet3_egs, read_nnet3_egs_ark


def load_label(label_path):
    char2index = dict()
    index2char = dict()
    with open(label_path, 'rt') as f:
        labels = f.readline().split(',')

    for i in range(len(labels)):
        char2index[labels[i]] = i
        index2char[i] = labels[i]

    return char2index, index2char


def load_params(pth_file, model, device):
    if os.path.isfile(pth_file):
        model.load_state_dict(
            torch.load(pth_file, map_location=device))
    else:
        raise FileNotFoundError(f"Can't find the {pth_file}")


class MfccDataset(Dataset):
    def __init__(self, datadir, char2index):
        self.scp = []
        with open(f'{datadir}/feats.scp', 'rt') as f:
            for line in f.readlines():
                parts = line.split()
                self.scp.append({'uttid': parts[0], 'feats': parts[1]})

        self.text = {}
        with open(f'{datadir}/text', 'rt') as f:
            for line in f.readlines():
                parts = line.strip().split(maxsplit=1)
                self.text[parts[0]] = parts[1]

        self.char2index = char2index

    def __len__(self):
        return len(self.scp)

    def __getitem__(self, idx):
        uttid = self.scp[idx]['uttid']
        parts = self.scp[idx]['feats'].split(':')
        with open(parts[0], 'rb') as f:
            f.seek(int(parts[1]))
            feature = read_mat(f)

        text = self.text[uttid]
        text = list(filter(None, [self.char2index[x] for x in list(text)]))
        text = [self.char2index['<sos>']] + text + [self.char2index['<eos>']]

        return torch.FloatTensor(feature.transpose()), text


# class EgsDataset(Dataset):
#     def __init__(self, egs_scp):
#         super(EgsDataset, self).__init__()
#         self.scp = []
#         with open(egs_scp, 'rt') as f:
#             for line in f.readlines():
#                 parts = line.split()
#                 self.scp.append({'uttid': parts[0], 'egs': parts[1]})

#     def __len__(self):
#         return len(self.scp)

#     def __getitem__(self, idx):
#         # uttid = self.scp[idx]['uttid']
#         parts = self.scp[idx]['egs'].split(':')
#         with open(parts[0], 'rb') as f:
#             f.seek(int(parts[1]))
#             fm, lm = read_nnet3_egs(f)

#         return fm['matrix'].transpose(), torch.tensor(lm['matrix'][0][0][0])


class EgsDataset(Dataset):
    def __init__(self, egs_ark):
        super(EgsDataset, self).__init__()
        self.scp_path = egs_ark.replace('ark', 'scp')
        self.file = open(egs_ark, 'rb')
        self.ark = read_nnet3_egs_ark(self.file)

    def __len__(self):
        with open(self.scp_path, 'rt') as f:
            lenght = len(f.readlines())
        return lenght

    def __getitem__(self, idx):
        # idx can't be used. Because egs.ark file is already shuffled, just load dataset.
        _, egs = next(self.ark)
        feature, label = egs[0]['matrix'], egs[1]['matrix']
        return feature.transpose(), torch.tensor(label[0][0][0])


class ExtractorDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = datadir

        command = "ark:apply-cmvn-sliding --norm-vars=false --center=true " \
            + f"--cmn-window=300 scp:{datadir}/feats.scp ark:- | " \
            + f"select-voiced-frames ark:- scp,s,cs:{datadir}/vad.scp ark:- |"
        self.data = read_mat_ark(command)

    def __len__(self):
        with open(f'{self.datadir}/feats.scp', 'rt') as f:
            length = len(f.readlines())
        return length

    def __getitem__(self, idx):
        key, data = next(self.data)
        return torch.from_numpy(data.transpose()), key
