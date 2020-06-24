#!/usr/bin/env python3
import argparse
import logging
import os
from tqdm import tqdm
import torch
from kaldi_io import write_vec_flt
from models.Xvector import Xvector, Xvector_AttnPooling
from kaldi_data.data_loader import ExtractorDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s] %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--attention", type=bool, default=True)

    parser.add_argument("--nj", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpu-idx", type=int, required=True)

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu_idx - 1}')

    # Load model
    with open(f'{args.model_dir}/feat_dim', 'rt') as f:
        feat_dim = int(f.readline().strip())
    with open(f'{args.model_dir}/num_spks', 'rt') as f:
        num_spks = int(f.readline().strip())

    if not args.attention:
        model = Xvector(feat_dim, num_spks, extract=True)
    else:
        model = Xvector_AttnPooling(feat_dim, num_spks, extract=True)
    model = model.to(device)

    state = torch.load(f'{args.model_dir}/final.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Extract X-vector
    for i in range(args.gpu_idx, args.nj + 1, args.num_gpus):
        dataset = ExtractorDataset(f"{args.data_dir}/split{args.nj}/{i}")
        ark = open(f'{args.output_dir}/xvector.{i}.ark', 'wb')
        scp = open(f'{args.output_dir}/xvector.{i}.scp', 'wt')

        desc = f" (xvector.{i}.ark,scp) "
        for feat, uttid in tqdm(dataset, desc=desc, leave=False, ncols=79):
            feat = feat.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(feat).squeeze(0).to('cpu').detach().numpy()

            ark.write((uttid + ' ').encode("latin1"))
            point = ark.tell()
            write_vec_flt(ark, embedding)
            scp.write(f'{uttid} {args.output_dir}/xvector.{i}.ark:{point}\n')
        ark.close()
        scp.close()


if __name__ == "__main__":
    main()
