#!/bin/bash
. ./cmd.sh
. ./path.sh
set -e

stage=0
. ./utils/parse_options.sh

voxceleb1_trials=data/voxceleb1_test/trials
egs_dir=/Features/voxceleb/egs_xvec
model_dir=exp/xvector_pytorch_1a

# We start training network using Pytorch.
if [ $stage -le 0 ]; then
  feat_dim=`cat $egs_dir/info/feat_dim`
  num_spks=`cat $egs_dir/temp/spk2int |wc -l`

  mkdir -p $model_dir
  echo $feat_dim > $model_dir/feat_dim
  echo $num_spks > $model_dir/num_spks

  train_xvector.py \
    --batch-size 1024 \
    --voxceleb-egs $egs_dir \
    --model-dir $model_dir \
    --feat-dim $feat_dim \
    --num-spks $num_spks || exit 1:
fi

num_gpus=`nvidia-smi --query-gpu=name --format=csv,noheader |wc -l`
# num_gpus=`lspci |grep VGA |wc -l`
# Now we extract speaker embedding vectors(X-vector) using the trained network. 
if [ $stage -le 1 ]; then
  # data/train
  nj=80
  utils/split_data.sh data/train $nj

  dir=$model_dir/xvectors_train
  mkdir -p $dir/log

  $train_cmd JOB=1:$num_gpus $dir/log/extract.JOB.log \
    extract_xvector.py --nj $nj --num-gpus $num_gpus --gpu-idx JOB \
      --data-dir data/train --model-dir $model_dir --output-dir $dir || exit 1;
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp
  
  # data/voxceleb1_test
  nj=40
  utils/split_data.sh data/voxceleb1_test $nj

  dir=$model_dir/xvectors_voxceleb1_test
  mkdir -p $dir/log

  $train_cmd JOB=1:$num_gpus $dir/log/extract.JOB.log \
    extract_xvector.py --nj $nj --num-gpus $num_gpus --gpu-idx JOB \
      --data-dir data/voxceleb1_test --model-dir $model_dir --output-dir $dir || exit 1;
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp
fi

if [ $stage -le 2 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $model_dir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$model_dir/xvectors_train/xvector.scp \
    $model_dir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $model_dir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$model_dir/xvectors_train/xvector.scp ark:- |" \
    ark:data/train/utt2spk $model_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $model_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$model_dir/xvectors_train/xvector.scp ark:- | transform-vec $model_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $model_dir/xvectors_train/plda || exit 1;
fi

if [ $stage -le 3 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $model_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $model_dir/xvectors_train/mean.vec scp:$model_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $model_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $model_dir/xvectors_train/mean.vec scp:$model_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $model_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 4 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

exit 0;
