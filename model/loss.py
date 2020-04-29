import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_mask_from_lengths
import pdb

class ParrotLoss(nn.Module):
    def __init__(self, hparams):
        super(ParrotLoss, self).__init__()
        self.hidden_dim = hparams.encoder_embedding_dim
        self.ce_loss = hparams.ce_loss

        self.L1Loss = nn.L1Loss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.n_frames_per_step = hparams.n_frames_per_step_decoder
        self.eos = hparams.n_symbols
        self.predict_spectrogram = hparams.predict_spectrogram

        self.contr_w = hparams.contrastive_loss_w
        self.consi_w = hparams.consistent_loss_w
        self.spenc_w = hparams.speaker_encoder_loss_w
        self.texcl_w = hparams.text_classifier_loss_w
        self.spadv_w = hparams.speaker_adversial_loss_w
        self.spcla_w = hparams.speaker_classifier_loss_w

    def parse_targets(self, targets, text_lengths):
        '''
        text_target [batch_size, text_len]
        mel_target [batch_size, mel_bins, T]
        spc_target [batch_size, spc_bins, T]
        speaker_target [batch_size]
        stop_target [batch_size, T]
        '''
        # pdb.set_trace()
        text_target, mel_target, speaker_target, stop_target = targets

        B = stop_target.size(0)
        stop_target = stop_target.reshape(B, -1, self.n_frames_per_step)
        stop_target = stop_target[:, :, 0]

        return text_target, mel_target, speaker_target, stop_target
    
    def forward(self, model_outputs, targets, input_text, eps=1e-5):

        '''
        predicted_mel [batch_size, mel_bins, T]
        predicted_stop [batch_size, T/r]
        alignment 
            when input_text==True [batch_size, T/r, max_text_len] 
            when input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, max_text_len, hidden_dim]
        text_logit_from_mel_hidden [B, max_text_len+1, n_symbols+1]
        speaker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, max_text_len, n_speakers]
        text_lengths [B,]
        mel_lengths [B,]
        # '''

        text_hidden, text_logit, text_lengths = model_outputs
        # text_hidden, text_logit, \
        # speaker_logits, \
        # text_lengths, mel_lengths, SE_alignments = model_outputs

        text_target, mel_target, speaker_target, stop_target  = self.parse_targets(targets, text_lengths)

        ## get masks ##
        # mel_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, mel_target.size(1), -1).float()

        # mel_step_lengths = torch.ceil(mel_lengths.float() / self.n_frames_per_step).long()

        text_mask = get_mask_from_lengths(text_lengths).float()
        text_mask_plus_one = get_mask_from_lengths(text_lengths + 1).float()

        n_symbols_plus_one = text_logit.size(2)

        # speaker classification loss #

        # speaker_encoder_loss = nn.CrossEntropyLoss()(speaker_logits, speaker_target)
        # _, predicted_speaker = torch.max(speaker_logits,dim=1)
        # speaker_encoder_acc = ((predicted_speaker == speaker_target).float()).sum() / float(speaker_target.size(0))

        text_logit_flatten = text_logit.reshape(-1, n_symbols_plus_one)
        text_target_flatten = text_target.reshape(-1)
        _, predicted_text =  torch.max(text_logit_flatten, dim=1)
        text_classification_acc = ((predicted_text == text_target_flatten).float()*text_mask.reshape(-1)).sum()/text_mask.sum()
        loss = self.CrossEntropyLoss(text_logit_flatten, text_target_flatten)
        text_classification_loss = torch.sum(loss * text_mask.reshape(-1)) / torch.sum(text_mask)

        # loss_list = [
        #         speaker_encoder_loss,
        #         text_classification_loss]    
        loss_list = [
                text_classification_loss] 

        # acc_list = [speaker_encoder_acc, text_classification_acc]        
        acc_list = [text_classification_acc]   

        # combined_loss1 = self.spenc_w * speaker_encoder_loss +  self.texcl_w * text_classification_loss
        combined_loss1 = self.texcl_w * text_classification_loss

        combined_loss2 = self.spenc_w * text_classification_loss*0
        
        return loss_list, acc_list, combined_loss1, combined_loss2

