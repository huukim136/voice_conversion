import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder
from torch.nn import functional as F
from .layers_6 import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet, GST
import pdb

# path_save = "/home/hk/voice_conversion/nonparaSeq2seqVC_code/pre-train/reader/spk_embeddings"


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

        self.se_alignment = VanillaAttention(hparams)

    def grouped_parameters(self,):

        params_group1 = [p for p in self.embedding.parameters()]
        params_group1.extend([p for p in self.text_encoder.parameters()])
        params_group1.extend([p for p in self.audio_seq2seq.parameters()])
        params_group1.extend([p for p in self.speaker_encoder.parameters()])
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

        pdb.set_trace()
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]
        text_hidden = self.text_encoder(text_input_embedded, text_lengths) # -> [B, max_text_len, hidden_dim]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        # -> [B, speaker_embedding_dim] 
        speaker_logit_from_mel, speaker_embedding, frame_spk_embeddings = self.speaker_encoder(mel_padded, mel_lengths) 
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

        contexts = []
        scores = []
        text_encoded =[]
        hidden = hidden.transpose(0,1)
        while len(contexts)< hidden.size(0):
            t = hidden[len(contexts)]
            # out , scores = self.stl(mel, text)
            # (output, score)  = self.se_alignment(mask, mel, text)
            (output, score)  = self.se_alignment(t, frame_spk_embeddings)
            contexts += [output]
            scores += [score] 
            # text_encoded += [t_encodeded]
        # pdb.set_trace()
        contexts = torch.stack(contexts).contiguous()
        scores = torch.stack(scores).contiguous()
        scores= scores.permute(1,0,2)
        contexts = contexts.permute(1,0,2)
        hidden = hidden.transpose(1,0)

        L = hidden.size(1)
        # hidden = torch.cat([hidden, contexts.detach()], -1)
        hidden = torch.cat([hidden, contexts], -1)

        predicted_mel, predicted_stop, alignments = self.decoder(hidden, mel_padded, text_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_stop, alignments,
                  text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments, 
                  speaker_logit_from_mel, speaker_logit_from_mel_hidden,
                  text_lengths, mel_lengths, scores]

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

        L = hidden.size(1)
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)
          
        predicted_mel, predicted_stop, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_stop, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            #speaker_id, speaker_embedding)
            speaker_id)


class VanillaAttention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, hparams, method="bahdanau", mlp=False):
        super(VanillaAttention, self).__init__()
        self.query_dim = hparams.encoder_embedding_dim  # q
        self.num_hidden = hparams.E # h
        self.embedding_dim = hparams.n_mel_channels # e
        self.output = hparams.E
        self.method = "bahdanau"

        if method == 'dot':
            pass
        elif method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == "concat":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        elif method == 'bahdanau':
            self.Wa = nn.Linear(self.query_dim, self.num_hidden, bias=False)
            # self.Ua = nn.Linear(self.embedding_dim, self.num_hidden, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hparams.batch_size, self.num_hidden))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.va, 0, 0.1)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        # pdb.set_trace()
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies , encoded_text, query = self._score(last_hidden, encoder_outputs, self.method)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))
        attention_energies = attention_energies.unsqueeze(1)
        # if (attention_energies.size(2) != mask.size(2)):
        #     # pdb.set_trace()
        #     zeros = torch.zeros(attention_energies.size(0), attention_energies.size(1), attention_energies.size(2))
        #     # source = torch.ones(30, 35, 49)
        #     zeros[:, :, :mask.size(2)] = mask
        #     mask = zeros
        #     mask = mask.type('torch.ByteTensor')
        #     mask = to_gpu(mask)
        # attention_energies[~mask] = float('-inf')
        scores = F.softmax(attention_energies, -1)
        expectations_of_sampling = torch.bmm(scores, encoded_text).squeeze(1)
        scores = scores.squeeze(1)
        return expectations_of_sampling, scores

    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """
        # pdb.set_trace()
        # assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        elif method == "bahdanau":
            x = last_hidden.unsqueeze(1)
            b = encoder_outputs
            a = self.Wa(x)
            out = F.tanh(a + b)
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1), b, a 

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)
