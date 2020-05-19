import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder_9 import Decoder
from .basic_layers import ConvNorm, LinearNorm
from torch.nn import functional as F
from .layers_10 import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet#, GST
import pdb

# path_save = "/home/hk/voice_conversion/nonparaSeq2seqVC_code/pre-train/reader/spk_embeddings"

class RefAttention(nn.Module):
    def __init__(self, hparams, attn_dropout=0.1):
    # def __init__(self, encoder_embedding_dim, attention_dim, ref_enc_gru_size,attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.query_layer = LinearNorm(hparams.encoder_embedding_dim, hparams.speaker_embedding_dim)
        self.key_layer = LinearNorm(hparams.speaker_encoder_hidden_dim//2, hparams.speaker_embedding_dim)
        self.value_layer = LinearNorm(hparams.speaker_encoder_hidden_dim//2, hparams.speaker_embedding_dim)
        self.projection1 = LinearNorm(hparams.speaker_embedding_dim, 
                                      hparams.encoder_embedding_dim, 
                                      w_init_gain='tanh')

        self.split_size = hparams.speaker_encoder_hidden_dim//2
        self.softmax = nn.Softmax(dim=2)
        self.scale = 1.0/np.sqrt(hparams.speaker_embedding_dim)

#    def forward(self, q, k, v, mask=None):             # original
    def forward(self, text_embedding, spk_embeddings, mask=None):
        #pdb.set_trace()
        B, len_q = text_embedding.size(0), text_embedding.size(1)      # B, t_L
        len_k = spk_embeddings.size(1)                           # p_l
        len_v = len_k

        att_weights = torch.cuda.FloatTensor(B, len_q, len_k).zero_()
        k, v = torch.split(spk_embeddings, self.split_size, dim=-1)
        key = self.key_layer(k)
        value = self.value_layer(v)
        query = self.query_layer(text_embedding)
        attn = torch.bmm(query / self.scale, key.transpose(1,2))
        if mask is not None:
            attn[~mask]= float('-inf')
        attn = self.dropout(F.softmax(attn, dim=-1))        
        output = torch.bmm(attn, value)                 # p_t (prosody-text_side)
        output = F.tanh(self.projection1(output))
        return output, attn

class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        #print hparams
        # plus <sos> 
        self.embedding = nn.Embedding(
            hparams.n_symbols + 1, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std

        self.sos = hparams.n_symbols

        self.embedding.weight.data.uniform_(-val, val)

        self.text_encoder = TextEncoder(hparams)

        self.audio_seq2seq = AudioSeq2seq(hparams)

        self.merge_net = MergeNet(hparams)

        self.speaker_encoder = SpeakerEncoder(hparams)

        #self.gst = GST(hparams)

        self.speaker_classifier = SpeakerClassifier(hparams)

        self.decoder = Decoder(hparams)
        
        self.postnet = PostNet(hparams)

        self.spemb_input = hparams.spemb_input

        self.se_alignment = RefAttention(hparams)

    def grouped_parameters(self,):

        params_group1 = [p for p in self.embedding.parameters()]
        params_group1.extend([p for p in self.text_encoder.parameters()])
        params_group1.extend([p for p in self.audio_seq2seq.parameters()])
        params_group1.extend([p for p in self.speaker_encoder.parameters()])
        params_group1.extend([p for p in self.se_alignment.parameters()])
        #params_group1.extend([p for p in self.gst.parameters()])
        params_group1.extend([p for p in self.merge_net.parameters()])
        params_group1.extend([p for p in self.decoder.parameters()])
        params_group1.extend([p for p in self.postnet.parameters()])

        return params_group1, [p for p in self.speaker_classifier.parameters()]

    def parse_batch(self, batch):
        text_input_padded, mel_padded, speaker_id, \
                    text_lengths, mel_lengths, stop_token_padded = batch
        
        text_input_padded = to_gpu(text_input_padded).long()
        mel_padded = to_gpu(mel_padded).float()
        # spc_padded = to_gpu(spc_padded).float()
        speaker_id = to_gpu(speaker_id).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_lengths = to_gpu(mel_lengths).long()
        stop_token_padded = to_gpu(stop_token_padded).float()

        return ((text_input_padded, mel_padded, text_lengths, mel_lengths),
                (text_input_padded, mel_padded,  speaker_id, stop_token_padded))

    def get_mask(self, text, text_lengths, length_dif=False):
        # pdb.set_trace()
        max_text_len = torch.max(text_lengths).item()
        #mask = torch.Tensor(text.size(0), max_text_len)
        if length_dif:
            mask = torch.zeros((text.size(0), max_text_len+1), dtype=torch.uint8)
        else:
            mask = torch.zeros((text.size(0), max_text_len), dtype=torch.uint8)
        mask = to_gpu(mask)
        #mask.zero_()

        for i in range(text.shape[0]):
            mask[i, :text_lengths[i]] = 1
        mask = mask.unsqueeze(1)
        return mask


    def forward(self, inputs, input_text):
        '''
        text_input_padded [batch_size, max_text_len]
        mel_padded [batch_size, mel_bins, max_mel_len]
        text_lengths [batch_size]
        mel_lengths [batch_size]

        #
        predicted_mel [batch_size, mel_bins, T]
        predicted_stop [batch_size, T/r]
        alignment input_text==True [batch_size, T/r, max_text_len] or input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, T/r, hidden_dim]
        spearker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, T/r, n_speakers]
        text_logit_from_mel_hidden [B, T/r, n_symbols]

        '''

        # pdb.set_trace()
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]
        text_hidden = self.text_encoder(text_input_embedded, text_lengths) # -> [B, max_text_len, hidden_dim]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        # -> [B, speaker_embedding_dim] 
        speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths) 
        #speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths) 

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat([mel_padded, 
                speaker_embedding.detach().unsqueeze(2).expand(-1, -1, T)], 1)
        else:
            audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments = self.audio_seq2seq(
                audio_input, mel_lengths, text_input_embedded, start_embedding) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]
        
        
        speaker_logit_from_mel_hidden = self.speaker_classifier(audio_seq2seq_hidden) # -> [B, text_len, n_speakers]

        if input_text:
            hidden = self.merge_net(text_hidden, text_lengths)
            # speaker_logit_from_mel, speaker_embedding, SE_alignments = self.speaker_encoder(mel_padded, mel_lengths,text_hidden,text_lengths )
        else:
            hidden = self.merge_net(audio_seq2seq_hidden, text_lengths)
            # speaker_logit_from_mel, speaker_embedding, SE_alignments = self.speaker_encoder(mel_padded, mel_lengths,audio_seq2seq_hidden,text_lengths )

        # if (frame_spk_embeddings.size(1) != mel_lengths[torch.argmax(mel_lengths)]):

        #     mask = self.get_mask(frame_spk_embeddings, mel_lengths, True)
        # else:
        #     mask = self.get_mask(frame_spk_embeddings, mel_lengths)
        #mask = mask.expand(-1,hidden.size(1),-1)
        #contexts , scores = self.se_alignment(hidden, frame_spk_embeddings, mask)
        # pdb.set_trace()
        L = hidden.size(1)
        # hidden = torch.cat([hidden, contexts.detach()], -1)
        # hidden = torch.cat([hidden, contexts], -1)
        #hidden = hidden +  contexts
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_stop, alignments = self.decoder(hidden, mel_padded, text_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_stop, alignments,
                  text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments, 
                  speaker_logit_from_mel, speaker_logit_from_mel_hidden,
                  text_lengths, mel_lengths]

        #outputs = [predicted_mel, post_output, predicted_stop, alignments,
        #          text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments, 
        #          speaker_logit_from_mel_hidden,
        #          text_lengths, mel_lengths]

        return outputs

    
    def inference(self, inputs, input_text, mel_reference, beam_width, spk_embeddings_save=False):
        '''
        decode the audio sequence from input
        inputs x
        input_text True or False
        mel_reference [1, mel_bins, T]
        '''
        #pdb.set_trace()
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs
        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2)
        text_hidden = self.text_encoder.inference(text_input_embedded)

        B = text_input_padded.size(0) # B should be 1
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding) # [1, embedding_dim]

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]
        speaker_id, speaker_embedding = self.speaker_encoder.inference(mel_reference)
        if spk_embeddings_save:
            spk_embed = speaker_embedding.data.cpu().numpy()[0]
            spk_embeddings_path = os.path.join(path_save, 'spk_embedding.npy')
            np.save(spk_embeddings_path, spk_embed)    
   
        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat([mel_padded, 
                speaker_embedding.detach().unsqueeze(2).expand(-1, -1, T)], 1)
        else:
            audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments = self.audio_seq2seq.inference_beam(
                audio_input, start_embedding, self.embedding, beam_width=beam_width) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        # -> [B, n_speakers], [B, speaker_embedding_dim] 

        if input_text:
            hidden = self.merge_net.inference(text_hidden)
        else:
            hidden = self.merge_net.inference(audio_seq2seq_hidden)

        #contexts , scores = self.se_alignment(hidden, frame_spk_embeddings)

        L = hidden.size(1)

        #hidden = hidden +  contexts
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)
          
        predicted_mel, predicted_stop, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_stop, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            #speaker_id, speaker_embedding)
            speaker_id)
