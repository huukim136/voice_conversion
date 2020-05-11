#!/bin/bash

# you can set the hparams by using --hparams=xxx
#CUDA_VISIBLE_DEVICES=0 python train.py -l logdir \
#-o outdir_100speaker_ref_attention_addition -c /home/hk/voice_conversion/nonparaSeq2seqVC_text-dependent_SE/outdir_100speaker_ref_attention_addition/checkpoint_107000  --hparams=speaker_adversial_loss_w=30.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.

CUDA_VISIBLE_DEVICES=0 python train.py -l logdir \
-o outdir_100_add_loss_MI_add_zero_grad -c /home/hk/voice_conversion/nonparaSeq2seqVC_text-dependent_SE/outdir_100_add_loss_MI_add_zero_grad/checkpoint_66000 --hparams=speaker_adversial_loss_w=30.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.
