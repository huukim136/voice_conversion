import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .basic_layers import sort_batch, ConvNorm, LinearNorm, Attention, tile
from .utils import get_mask_from_lengths
from .beam import Beam, GNMTGlobalScorer
import pdb
        
class SpeakerClassifier(nn.Module):
    '''
    - n layer CNN + PROJECTION
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        
        convolutions = []
        for i in range(hparams.SC_n_convolutions):
            #parse dim
            if i == 0:
                in_dim = hparams.encoder_embedding_dim
                out_dim = hparams.SC_hidden_dim
            elif i == (hparams.SC_n_convolutions-1):
                in_dim = hparams.SC_hidden_dim
                out_dim = hparams.SC_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(in_dim,
                         out_dim,
                         kernel_size=hparams.SC_kernel_size, stride=1,
                         padding=int((hparams.SC_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='leaky_relu',
                         param=0.2),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.projection = LinearNorm(hparams.SC_hidden_dim, hparams.n_speakers)

    def forward(self, x):
        # x [B, T, dim]

        # -> [B, DIM, T]
        hidden = x.transpose(1, 2)
        for conv in self.convolutions:
            hidden = conv(hidden)
        
        # -> [B, T, dim]
        hidden = hidden.transpose(1, 2)
        logits = self.projection(hidden)

        return logits

class SpeakerEncoder(nn.Module):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling

    '''
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(hparams.n_mel_channels, int(hparams.speaker_encoder_hidden_dim / 2), 
                            num_layers=2, batch_first=True,  bidirectional=True, dropout=hparams.speaker_encoder_dropout)
        self.projection1 = LinearNorm(hparams.speaker_encoder_hidden_dim, 
                                      hparams.speaker_encoder_hidden_dim, 
                                      w_init_gain='tanh')
        self.projection2 = LinearNorm(hparams.speaker_encoder_hidden_dim, hparams.n_speakers) 
    
    def forward(self, x, input_lengths):
        '''
         x  [batch_size, mel_bins, T]

         return 
         logits [batch_size, n_speakers]
         embeddings [batch_size, embedding_dim]
        '''
        # pdb.set_trace()
        x = x.transpose(1,2)
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
 
        # outputs = torch.sum(outputs,dim=1) / sorted_lengths.unsqueeze(1).float() # mean pooling -> [batch_size, dim]

        # outputs = F.tanh(self.projection1(outputs))
        outputs = outputs[initial_index]
        # # L2 normalizing #
        # embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        # logits = self.projection2(outputs)

        return outputs
    
    def inference(self, x): 
        
        x = x.transpose(1,2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        # outputs = torch.sum(outputs,dim=1) / float(outputs.size(1)) # mean pooling -> [batch_size, dim]
        frame_spk_embeddings = F.tanh(self.projection1(outputs))
        frame_spk_embeddings = frame_spk_embeddings / torch.norm(frame_spk_embeddings, dim=2, keepdim=True)
        # logits = self.projection2(outputs)

        outputs = torch.sum(outputs,dim=1) / float(outputs.size(1))
        outputs = F.tanh(self.projection3(outputs))
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        pid = torch.argmax(logits, dim=1)

        return pid, embeddings, frame_spk_embeddings

class DenseBlock(nn.Module):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling

    '''
    def __init__(self, hparams):
        super(DenseBlock, self).__init__()
        self.projection1 = LinearNorm(hparams.speaker_encoder_hidden_dim, 
                                      hparams.speaker_encoder_hidden_dim, 
                                      w_init_gain='tanh')
        self.projection2 = LinearNorm(hparams.speaker_encoder_hidden_dim, 
                                      hparams.speaker_encoder_hidden_dim, 
                                      w_init_gain='tanh')
        self.projection3 = LinearNorm(hparams.speaker_encoder_hidden_dim, hparams.n_speakers) 
    
    def forward(self, x, input_lengths):
        '''
         x  [batch_size, mel_bins, T]

         return 
         logits [batch_size, n_speakers]
         embeddings [batch_size, embedding_dim]
        '''
        # pdb.set_trace()

        # outputs = torch.sum(outputs,dim=1) / sorted_lengths.unsqueeze(1).float() # mean pooling -> [batch_size, dim]

        x = F.tanh(self.projection1(x))
        x = F.tanh(self.projection2(x))
        x = torch.sum(x,dim=1) / input_lengths.unsqueeze(1).float() # mean pooling -> [batch_size, dim]

        logits = self.projection3(x)

        return logits, x
    
    def inference(self, x): 
        
        x = x.transpose(1,2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        # outputs = torch.sum(outputs,dim=1) / float(outputs.size(1)) # mean pooling -> [batch_size, dim]
        frame_spk_embeddings = F.tanh(self.projection1(outputs))
        frame_spk_embeddings = frame_spk_embeddings / torch.norm(frame_spk_embeddings, dim=2, keepdim=True)
        # logits = self.projection2(outputs)

        outputs = torch.sum(outputs,dim=1) / float(outputs.size(1))
        outputs = F.tanh(self.projection3(outputs))
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        pid = torch.argmax(logits, dim=1)

        return pid, embeddings, frame_spk_embeddings


class Speaker_Text_Alignments(nn.Module):
    '''
    one layer bi-lstm
    '''
    def __init__(self, hparams):
        super(Speaker_Text_Alignments, self).__init__()
        self.se_alignment = VanillaAttention(hparams)
        # self.lstm = nn.LSTM(hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim/2), 
        #                     num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, x, input_lengths):
        '''
        x [B, T, dim]
        '''
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = outputs[initial_index]

        return outputs   


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
            self.Ua = nn.Linear(self.embedding_dim, self.num_hidden, bias=False)
            self.va = nn.Parameter(torch.FloatTensor(hparams.batch_size, self.num_hidden))
        else:
            raise NotImplementedError

        self.mlp = mlp
        if mlp:
            self.phi = nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.va, 0, 0.1)

    def forward(self, last_hidden, encoder_outputs, mask, seq_len=None):
        # pdb.set_trace()
        batch_size, seq_lens, _ = encoder_outputs.size()
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        attention_energies , encoded_text, query = self._score(last_hidden, encoder_outputs, self.method)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))
        attention_energies = attention_energies.unsqueeze(1)
        if (attention_energies.size(2) != mask.size(2)):
            # pdb.set_trace()
            zeros = torch.zeros(attention_energies.size(0), attention_energies.size(1), attention_energies.size(2))
            # source = torch.ones(30, 35, 49)
            zeros[:, :, :mask.size(2)] = mask
            mask = zeros
            mask = mask.type('torch.ByteTensor')
            mask = to_gpu(mask)
        attention_energies[~mask] = float('-inf')
        scores = F.softmax(attention_energies, -1)
        expectations_of_sampling = torch.bmm(scores, encoded_text).squeeze(1)
        scores = scores.squeeze(1)
        return expectations_of_sampling, scores, query

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
            b = self.Ua(encoder_outputs)
            a = self.Wa(x)
            out = F.tanh(a + b)
            return out.bmm(self.va.unsqueeze(2)).squeeze(-1), b, a 

        else:
            raise NotImplementedError

    def extra_repr(self):
        return 'score={}, mlp_preprocessing={}'.format(
            self.method, self.mlp)


class MergeNet(nn.Module):
    '''
    one layer bi-lstm
    '''
    def __init__(self, hparams):
        super(MergeNet, self).__init__()
        self.lstm = nn.LSTM(hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim/2), 
                            num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, x, input_lengths):
        '''
        x [B, T, dim]
        '''
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = outputs[initial_index]

        return outputs

    def inference(self,x):

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class AudioEncoder(nn.Module):
    '''
    - Simple 2 layer bidirectional LSTM

    '''
    def __init__(self, hparams):
        super(AudioEncoder, self).__init__()

        if hparams.spemb_input:
            input_dim = hparams.n_mel_channels + hparams.speaker_embedding_dim
        else:
            input_dim = hparams.n_mel_channels
        
        self.lstm1 = nn.LSTM(input_dim, int(hparams.audio_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hparams.audio_encoder_hidden_dim*hparams.n_frames_per_step_encoder,
                             int(hparams.audio_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)

        self.concat_hidden_dim = hparams.audio_encoder_hidden_dim*hparams.n_frames_per_step_encoder
        self.n_frames_per_step = hparams.n_frames_per_step_encoder
    
    def forward(self, x, input_lengths):
        '''
        x  [batch_size, mel_bins, T]

        return [batch_size, T, channels]
        '''
        x = x.transpose(1, 2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x_packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=x.size(1)) # use total_length make sure the recovered sequence length not changed

        outputs = outputs.reshape(x.size(0), -1, self.concat_hidden_dim)

        output_lengths = torch.ceil(sorted_lengths.float() / self.n_frames_per_step).long()
        outputs = nn.utils.rnn.pack_padded_sequence(
            outputs, output_lengths.cpu().numpy() , batch_first=True)

        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs[initial_index], output_lengths[initial_index]
    
    def inference(self, x):
        
        x = x.transpose(1, 2)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x)
        outputs = outputs.reshape(1, -1, self.concat_hidden_dim)
        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        return outputs


class AudioSeq2seq(nn.Module):
    '''
    - Simple 2 layer bidirectional LSTM

    '''
    def __init__(self, hparams):
        super(AudioSeq2seq, self).__init__()

        self.encoder = AudioEncoder(hparams)

        self.decoder_rnn_dim = hparams.audio_encoder_hidden_dim

        self.attention_layer = Attention(self.decoder_rnn_dim, hparams.audio_encoder_hidden_dim,
            hparams.AE_attention_dim, hparams.AE_attention_location_n_filters,
            hparams.AE_attention_location_kernel_size)

        self.decoder_rnn =  nn.LSTMCell(hparams.symbols_embedding_dim + hparams.audio_encoder_hidden_dim, 
            self.decoder_rnn_dim)
   
        def _proj(activation):
            if activation is not None:
                return nn.Sequential(LinearNorm(self.decoder_rnn_dim+hparams.audio_encoder_hidden_dim, 
                                     hparams.encoder_embedding_dim,
                                     w_init_gain=hparams.hidden_activation),
                                     activation)
            else:
                return LinearNorm(self.decoder_rnn_dim+hparams.audio_encoder_hidden_dim, 
                                  hparams.encoder_embedding_dim,
                                  w_init_gain=hparams.hidden_activation)
        
        if hparams.hidden_activation == 'relu':
            self.project_to_hidden = _proj(nn.ReLU())
        elif hparams.hidden_activation == 'tanh':
            self.project_to_hidden = _proj(nn.Tanh())
        elif hparams.hidden_activation == 'linear':
            self.project_to_hidden = _proj(None)
        else:
            print('Must be relu, tanh or linear.')
            assert False
        
        self.project_to_n_symbols= LinearNorm(hparams.encoder_embedding_dim,
            hparams.n_symbols + 1) # plus the <eos>
        self.eos = hparams.n_symbols
        self.activation = hparams.hidden_activation
        self.max_len = 100

    def initialize_decoder_states(self, memory, mask):
   
        B = memory.size(0)
        
        MAX_TIME = memory.size(1)
        
        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        
        self.attention_weigths = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weigths_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
    
    def map_states(self, fn):
        '''
        mapping the decoder states using fn
        '''
        self.decoder_hidden = fn(self.decoder_hidden, 0)
        self.decoder_cell = fn(self.decoder_cell, 0)
        self.attention_weigths = fn(self.attention_weigths, 0)
        self.attention_weigths_cum = fn(self.attention_weigths_cum, 0)
        self.attention_context = fn(self.attention_context, 0)


    def parse_decoder_outputs(self, hidden, logit, alignments):

        # -> [B, T_out + 1, max_time]
        alignments = torch.stack(alignments).transpose(0,1)
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1,  n_symbols] 
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()

        return hidden, logit, alignments

    def decode(self, decoder_input):
        # import pdb
        # pdb.set_trace()

        cell_input = torch.cat((decoder_input, self.attention_context),-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, 
            (self.decoder_hidden, 
            self.decoder_cell))

        attention_weigths_cat = torch.cat(
            (self.attention_weigths.unsqueeze(1),
            self.attention_weigths_cum.unsqueeze(1)),dim=1)
        
        self.attention_context, self.attention_weigths = self.attention_layer(
            self.decoder_hidden, 
            self.memory, 
            self.processed_memory, 
            attention_weigths_cat, 
            self.mask)

        self.attention_weigths_cum += self.attention_weigths
        
        hidden_and_context = torch.cat((self.decoder_hidden, self.attention_context), -1)

        hidden = self.project_to_hidden(hidden_and_context)
        normed_hidden = hidden/torch.norm(hidden,dim=1,keepdim=True)
        
        # dropout to increasing g
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return normed_hidden, logit, self.attention_weigths
    
    def forward(self, mel, mel_lengths, decoder_inputs, start_embedding):
        '''
        decoder_inputs: [B, channel, T] 

        start_embedding [B, channel]

        return 
        hidden_outputs [B, T+1, channel]
        logits_outputs [B, T+1, n_symbols]
        alignments [B, T+1, max_time]

        '''

        memory, memory_lengths = self.encoder(mel, mel_lengths)

        decoder_inputs = decoder_inputs.permute(2, 0, 1) # -> [T, B, channel]
        decoder_inputs = torch.cat((start_embedding.unsqueeze(0), decoder_inputs), dim=0)

        self.initialize_decoder_states(memory, 
            mask=~get_mask_from_lengths(memory_lengths))

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):

            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)

            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]
        
        hidden_outputs, logit_outputs, alignments = \
            self.parse_decoder_outputs(
                hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments
        
    '''
    use beam search ? 
    '''
    def inference_greed(self, x, start_embedding, embedding_table):
        '''
        decoding the phone sequence using greed algorithm
        x [1, mel_bins, T]
        start_embedding [1,embedding_dim]
        embedding_table nn.Embedding class

        return
        hidden_outputs [1, ]
        '''
        import pdb
        pdb.set_trace()
        MAX_LEN = 100

        decoder_input = start_embedding
        memory = self.encoder.inference(x)

        self.initialize_decoder_states(memory, mask=None)

        hidden_outputs, alignments, phone_ids = [], [], []
        while True:
            hidden, logit, attention_weights = self.decode(decoder_input)

            hidden_outputs += [hidden]
            alignments += [attention_weights]
            phone_id = torch.argmax(logit,dim=1)
            phone_ids += [phone_id]

            # if reaches the <eos>
            if phone_id.squeeze().item() == self.eos:
                break
            if len(hidden_outputs) == self.max_len:
                break
                print('Warning! The decoded text reaches the maximum lengths.')

            # embedding the phone_id
            decoder_input = embedding_table(phone_id) # -> [1, embedding_dim]

        hidden_outputs, phone_ids, alignments = \
            self.parse_decoder_outputs(hidden_outputs, phone_ids, alignments)

        return hidden_outputs, phone_ids, alignments

    def inference_beam(self, x, start_embedding, embedding_table,
            beam_width=20,):
        #import pdb
        #pdb.set_trace()
 
        memory = self.encoder.inference(x).expand(beam_width, -1,-1)

        MAX_LEN = 100
        n_best = 5

        self.initialize_decoder_states(memory, mask=None)
        decoder_input = tile(start_embedding, beam_width) # copy beam_width times the start_embedding --> we have decoder_input.shape = [10, 512]


        beam = Beam(beam_width, 0, self.eos, self.eos, 
            n_best=n_best, cuda=True, global_scorer=GNMTGlobalScorer())
        
        hidden_outputs, alignments, phone_ids = [], [], []

        for step in range(MAX_LEN):
            if beam.done():
                break
            
            hidden, logit, attention_weights = self.decode(decoder_input)
            logit = F.log_softmax(logit, dim=1)

            beam.advance(logit, attention_weights, hidden)
            select_indices = beam.get_current_origin()

            self.map_states(lambda state, dim: state.index_select(dim, select_indices))

            decoder_input = embedding_table(beam.get_current_state())

        scores, ks = beam.sort_finished(minimum=n_best)
        hyps, attn, hiddens = [], [], []

        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att, hid = beam.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
            hiddens.append(hid)

        return hiddens[0].unsqueeze(0), hyps[0].unsqueeze(0), attn[0].unsqueeze(0)
        


class TextEncoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(TextEncoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.dropout = hparams.text_encoder_dropout 

        
        def _proj(activation):
            if activation is not None:
                return nn.Sequential(LinearNorm(hparams.encoder_embedding_dim, 
                                     hparams.encoder_embedding_dim,
                                     w_init_gain=hparams.hidden_activation),
                                     activation)
            else:
                return LinearNorm(hparams.encoder_embedding_dim, 
                                  hparams.encoder_embedding_dim,
                                  w_init_gain=hparams.hidden_activation)
        
        if hparams.hidden_activation == 'relu':
            self.projection = _proj(nn.ReLU())
        elif hparams.hidden_activation == 'tanh':
            self.projection = _proj(nn.Tanh())
        elif hparams.hidden_activation == 'linear':
            self.projection = _proj(None)
        else:
            print('Must be relu, tanh or linear.')
            assert False

        # self.project_to_n_symbols= LinearNorm(hparams.encoder_embedding_dim,
        #     hparams.n_symbols + 1) # plus the <eos>
        self.project_to_n_symbols= LinearNorm(hparams.encoder_embedding_dim,
            hparams.n_symbols)    

    def forward(self, x, input_lengths):
        '''
        x: [batch_size, channel, T]

        return [batch_size, T, channel]
        '''
        # pdb.set_trace()
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout, self.training)

        # -> [batch_size, T, channel]
        x = x.transpose(1, 2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        # pytorch tensor are not reversible, hence the conversion
        sorted_lengths = sorted_lengths.cpu().numpy()

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        outputs = self.projection(outputs)
        outputs = outputs/torch.norm(outputs, dim=2,keepdim=True)
        logits = self.project_to_n_symbols(F.dropout(outputs, 0.5, self.training))
        return outputs[initial_index], logits

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs = self.projection(outputs)
        outputs = outputs/torch.norm(outputs, dim=2,keepdim=True)

        return outputs


class PostNet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_dim,
                             hparams.postnet_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_dim))
            )


        if hparams.predict_spectrogram:
            out_dim = hparams.n_spc_channels
            self.projection = LinearNorm(hparams.n_mel_channels, hparams.n_spc_channels, bias=False)
        else:
            out_dim = hparams.n_mel_channels
        
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_dim, out_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(out_dim))
            )
        self.dropout = hparams.postnet_dropout
        self.predict_spectrogram = hparams.predict_spectrogram

    def forward(self, input):
        # input [B, mel_bins, T]

        x = input
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), self.dropout, self.training)

        if self.predict_spectrogram:
            o = x + self.projection(input.transpose(1,2)).transpose(1,2)
        else:
            o = x + input

        return o


class GST(nn.Module):

    def __init__(self, hparams):

        super(GST, self).__init__()
        self.encoder = ReferenceEncoder(hparams)
        self.stl = STL(hparams)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, hparams):

        super(ReferenceEncoder, self).__init__()
        K = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hparams.ref_enc_filters[-1] * out_channels,
                          hidden_size=hparams.E // 2,
                          batch_first=True)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, hparams.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, hparams):

        super(STL, self).__init__()
        self.embed = nn.flatten_parameters(torch.FloatTensor(hparams.token_num, hp.E // hparams.num_heads))
        d_q = hparams.E // 2
        d_k = hparams.E // hparams.num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hparams.E, num_heads=hparams.num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super(MultiHeadAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out