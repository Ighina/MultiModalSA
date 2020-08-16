# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:57:23 2020

@author: Iacopo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence as PAD
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def create_mask(src, lengths):
    """Create a mask hiding future tokens
    Parameters:
        src (tensor): the source tensor having shape [batch_size, number_of_steps, features_dimensions]
        length (list): a list of integers representing the length (i.e. number_of_steps) of each sample in the batch."""
    mask = []
    max_len = src.shape[1]
    for index, i in enumerate(src):
        # The mask consists in tensors having false at the step number that doesn't need to be hidden and true otherwise
        mask.append([False if i<lengths[index] else True for i in range(max_len)])
    return torch.tensor(mask)

class RNN(nn.Module):
    """Class implementing recurrent networks (LSTM/GRU)"""
    def __init__(self, embed_size,hidden_size,num_layers=1,labels=1,
                  bidirectional=False, dropout_in=0.0,
                  dropout_out=0.0,padding_idx=0, batch_first=True,
                  LSTM=True):
        super(RNN,self).__init__()
        self.embed_size = embed_size # embedding/input dimensions
        self.hidden_size = hidden_size # hidden layers' dimensions
        self.labels = labels # output classes
        self.num_layers = num_layers # number of recurrent layers
        self.bidirectional=bidirectional # boolean: bidirectional=True makes the network bidirectional
        
        
        if LSTM:
            # If LSTM is true, use LSTM else use GRU
            self.rnn = nn.LSTM(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
            
        else:
            self.rnn = nn.GRU(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
        
        self.dropout_in = dropout_in
        
        self.dropout_out = dropout_out
        
        if self.bidirectional:
            self.output = nn.Linear(hidden_size*2,labels)
            self.polarity = nn.Linear(hidden_size*2,2) # option to perform multitask learning with polarity of the sentence
            self.arousal = nn.Linear(hidden_size*2, 4) # option to perform multitask learning with arousal of the sentence
        else:
            self.output = nn.Linear(hidden_size, labels)
            self.polarity = nn.Linear(hidden_size,2) # option to perform multitask learning with polarity of the sentence
            self.arousal = nn.Linear(hidden_size, 4) # option to perform multitask learning with arousal of the sentence
        
        
        
    def forward(self, line, line_len=None, apply_softmax=False, return_final=False, classifier=False):
        """
        Parameters:
            line (tensor): the input tensor having shape [batch_size, number_of_steps, features_dimensions]
            line_len (list): a list containing the length of each sample in the batch. If no list is passed, then the function assumes all samples to have same length (i.e. no padding)
            apply_softmax (boolean): whether to apply the softmax function or not (as in the case for cross-entropy loss) after the classifier layer
            return_final (boolean): whether or not to return the final hidden state (e.g. to use it as first hidden state in a decoder)
            classifier (boolean): whether the network has a classifier layer or it acts just as an encoder"""
        
        if self.dropout_in:
            # if dropout_in value is not 0, then apply the dropout
            line = F.dropout(line, p=self.dropout_in)
        
        if line_len is not None:
            # if lengths are provided, pack the input tensor, else nothing happens
            embedded = PACK(line, line_len.data.tolist(), batch_first=True, enforce_sorted=False)
        else:
            embedded = line
        
        if self.bidirectional:
            # if the network is bidirectional, first create the initial hidden and memory cell states (for LSTM)
            batch_size = line.shape[0]
            
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
            
            
            hidden_initial = line.new_zeros(*state_size)
            
            cells_initial = line.new_zeros(*state_size)
            
            packed_out, (final_hidden_states, final_cell_states) = self.rnn(embedded,(hidden_initial,cells_initial))
            
            rnn_out, _ = PAD(packed_out, batch_first=True) # unpack the rnn output and pad it with 0s where needed
            
            if self.dropout_out:
                # if dropout_out is not 0, apply dropout to the rnn output
                rnn_out = F.dropout(rnn_out, p=self.dropout_out)
            
            if classifier:
                # if the network is a classifier: apply the classification layer
                rnn_out_new = rnn_out[:,-1,:].squeeze(1)
                out = self.output(rnn_out_new)
                
            else:
                # else no output is required (the output of the network as encoder is the rnn_out)
                out = None
            
            
            
            if return_final:
                # if returning final hidden state, concatenate the final hidden state from forward and backward layers of the network
                def combine_directions(outs):
                    return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
                final_hidden_states = combine_directions(final_hidden_states)
                final_cell_states = combine_directions(final_cell_states)
                return out, rnn_out, (final_hidden_states, final_cell_states)
            
            else:
                return out, rnn_out
            
        
        else:
            # same as the bidirectional case, but with less operations needed
            
            rnn_out,h_n = self.rnn(embedded)
            
            if self.dropout_out:
                rnn_out = F.dropout(rnn_out, p=self.dropout_out)
            
            if line_len is not None:
                lengths = torch.tensor([line_len]*line.shape[0])
                rnn_out_new = column_gatherer(rnn_out, lengths)
            else:
                batch_size, seq_size, feat_size = rnn_out.shape
                rnn_out_new = rnn_out.contiguous().view(batch_size,seq_size, feat_size)[:,-1]
            
            if classifier:
                out = self.output(rnn_out_new)
                if apply_softmax:
                    out = F.softmax(out,dim=1)
            else:
                out = None
            if return_final:
                return out, rnn_out, h_n
            else:
                return out, rnn_out
    
def column_gatherer(y_out, lengths):
    """Gather the final states from a RNN with padded inputs, so that the
    actual final state for each sample can be used for classification.
    Parameters:
        y_out (tensor): the RNN output
        lengths (tensor): the individual lengths of each sample in the batch"""
    lengths = lengths.long().detach().numpy()-1
    out = []
    for batch_index, column_index in enumerate(lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
              for i in range(0, d_model, 2):
                  pe[pos, i] = \
                  math.sin(pos / (10000 ** ((2 * i)/d_model)))
                  try:
                    pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                  except IndexError:
                    pass
        self.pe = pe
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    """Class implementing transformer ecnoder, partially based on
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, in_dim, h_dim, n_heads, n_layers, dropout=0.2, drop_out = 0.0):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(in_dim, n_heads, h_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers, norm=nn.LayerNorm(in_dim))
        self.in_dim = in_dim
        self.drop_out = drop_out
        
    def forward(self, src, line_len=None):
        src = src * math.sqrt(self.in_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        if self.drop_out:
            output = F.dropout(output, p = self.drop_out)
        return src, output
    
class ConvolNet(nn.Module):
    """A simple single convolution layer for input preprocessing"""
    def __init__(self, in_dim, h_dim, kernel=1):
        super(ConvolNet, self).__init__()
        self.conv = nn.Conv1d(in_dim, h_dim, kernel)
        self.activation = nn.ReLU()
        
    def forward(self, src):
        return (src,self.activation(self.conv(src)))

class AttentionMultiModal(nn.Module):
    """Multimodal network based on two separate attention mechanisms.
    In this network the central modality (here, text) is assumed to consist
    of a single tensor per sentence (i.e. sentence embedding), while the other modalities
    (e.g. audio/video), consist in multiple frames per utterance. The model, then, works
    by first encoding (or not, if the raw option is selected) the audio/video modalities.
    Then, two separate attention mechanisms (one for text-audio, one for text-video) create
    the sentence representation for the two modalities, that is driven (via the attention)
    by the central modality (i.e. text). In both cases the text sentence representation
    is concatenated to the audio/video one, projected, passed into a nonlinearity and, finally,
    the two separate results from text-audio and text-video are concatenated and projected
    into the number of output dimensions for classification.
    This implementation allows the use and experimentation of many different encoders for
    the audio/video modalities. The default is a bidirectional LSTM, but other options
    can be selected by passing True to the relative parameter when instantiating the
    model (e.g. model = AttentionMultiModal(transformer=True))"""
    def __init__(self, text_shape=768, audio_shape=74, 
                 video_shape=35, attn_project=32, hidden_encoder_audio=74,
                 kern = 1, hidden_encoder_video = 35, out_dim = 1,
                 nlayers=1, drop_enc_in= 0.0, drop_enc_out = 0.0,
                 drop_in = 0.2, drop_out = 0.0, attn_bias = False,
                 video=False, audio=False, transformer=False, 
                 raw=False, convolutional=False, bidirectional = True):
        super(AttentionMultiModal, self).__init__()
        self.audio_shape = audio_shape
        self.video_shape = video_shape
        self.raw = False
        self.convolution = False
        self.bilstm = False
        self.drop_out = drop_out
        
        if video and audio:
            self.out = nn.Linear(attn_project*2, out_dim)
        else:
            self.out = nn.Linear(attn_project, out_dim)
        
        if transformer:
            self.audio_heads = max([i for i in range(1,audio_shape) if audio_shape%i==0])
            self.audio = Transformer(audio_shape, hidden_encoder_audio, self.audio_heads, nlayers, drop_enc_in)
            self.text2audio = nn.Linear(text_shape, audio_shape)
            self.attn_audio = AttentionLayer(audio_shape, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            self.video_heads = max([i for i in range(1,video_shape) if video_shape%i==0])
            self.video = Transformer(video_shape, hidden_encoder_video, self.video_heads, nlayers, drop_enc_in)
            self.text2video = nn.Linear(text_shape, video_shape)
            self.attn_video = AttentionLayer(video_shape, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            
            
        elif raw:
            self.raw = True
            self.text2audio = nn.Linear(text_shape, audio_shape)
            self.attn_audio = AttentionLayer(audio_shape, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            self.video = ConvolNet(video_shape, video_shape*2)
            self.text_2_video = nn.Linear(text_shape, video_shape)
            self.attn_video = AttentionLayer(video_shape, attn_project, attn_bias = attn_bias, drop_in = drop_in)
        
        elif convolutional:
            self.convolution = True
            self.audio = ConvolNet(audio_shape, hidden_encoder_audio, kern)
            self.text2audio = nn.Linear(text_shape, hidden_encoder_audio)
            self.attn_audio = AttentionLayer(hidden_encoder_audio, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            self.video = ConvolNet(video_shape, hidden_encoder_video, kern)
            self.text_2_video = nn.Linear(text_shape, hidden_encoder_video)
            self.attn_video = AttentionLayer(hidden_encoder_video, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            
        else:
            audio_shape_out = hidden_encoder_audio*2 if bidirectional else hidden_encoder_audio
            video_shape_out = hidden_encoder_video*2 if bidirectional else hidden_encoder_video
            self.bilstm = True
            self.audio = RNN(audio_shape, hidden_encoder_audio, nlayers, bidirectional=bidirectional, 
                             dropout_in=drop_enc_in, dropout_out = drop_enc_out)
            self.text2audio= nn.Linear(text_shape, audio_shape_out)
            self.attn_audio = AttentionLayer(audio_shape_out, attn_project, attn_bias = attn_bias, drop_in = drop_in)
            self.video = RNN(video_shape, hidden_encoder_video, nlayers, bidirectional=bidirectional,
                             dropout_in=drop_enc_in, dropout_out = drop_enc_out)
            self.text2video = nn.Linear(text_shape, video_shape_out)
            self.attn_video = AttentionLayer(video_shape_out, attn_project, attn_bias = attn_bias, drop_in = drop_in)
    
    def forward(self,audio_input, text_input, video_input, mask=None, audio=True, video=True, line_len=None, modified_attention=False, no_attention=False):
        assert video or audio, 'At least one of video and audio input must be set to true!'
        if audio:
            if self.convolution:
                batch_size = text_input.shape[0]
                enc_audio = self.audio(audio_input.contiguous().view(batch_size,self.audio_shape,-1))[0].contiguous().view(batch_size,-1,self.audio_shape)
                
                    
            elif self.raw:
                enc_audio = audio_input
            
            else:
                enc_audio = self.audio(audio_input, line_len)[1]
            
            text_projection_audio = self.text2audio(text_input)
            attn_out_audio, attn_weights_audio = self.attn_audio(text_projection_audio, enc_audio, mask, modified_attention = modified_attention, no_attention=no_attention)
        
        if video:
            if self.convolution:
                batch_size = text_input.shape[0]
                enc_video = self.video(video_input.contiguous().view(batch_size,self.video_shape,-1))[0].contiguous().view(batch_size, -1, self.video_shape)
                
            elif self.raw:
                enc_video = video_input
                
            else:
                enc_video = self.video(video_input, line_len)[1]
            text_projection_video = self.text2video(text_input)
            attn_out_video, attn_weights_video = self.attn_video(text_projection_video, enc_video, mask, modified_attention = modified_attention, no_attention=no_attention)
        
        if audio and video:
            attn_out = torch.cat((attn_out_audio, attn_out_video), dim=1)
            output = self.out(attn_out)
        elif audio:
            output = self.out(attn_out_audio)
        else:
            output = self.out(attn_out_video)
        outs = (output, attn_weights_audio, attn_weights_video)
        
            
        return outs
    
class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims, attn_bias=False, drop_in=0.0):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, input_dims, bias=attn_bias)
        self.context_plus_hidden_projection = nn.Linear(input_dims*2, output_dims, bias=attn_bias)
        self.drop_in = drop_in
    
    def forward(self, tgt_input, encoder_out, src_mask, modified_attention=False, no_attention=False):
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [batch_size, src_time_steps, output_dims]
        # src_mask has shape = [batch_size, src_time_steps]

        encoder_out = F.dropout(encoder_out, p = self.drop_in)
        
        if no_attention:
            attn_context = torch.mean(encoder_out, dim=1)
        else:
            # Get attention scores
            # [batch_size, src_time_steps, output_dims]
            attn_scores = self.score(tgt_input, encoder_out)
            
            if modified_attention:
                
                attn_weights = attn_scores
            
                attn_context = torch.bmm(attn_scores, encoder_out).squeeze(dim=1)
            
            else:
                
                if src_mask is not None:
                    src_mask = src_mask.unsqueeze(dim=1)
                    attn_scores.masked_fill_(src_mask, float('-inf'))
                
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)
        
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        
        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """

        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)
        
        return attn_scores

# Below subnetwork classes are taken from https://github.com/Justin1904/TensorFusionNetworks
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3
    
class TextSubNet(nn.Module):
    '''
    Simple projection layer to parameterise and reduce the text modality to suitable dimension
    '''

    def __init__(self, in_size, out_size, dropout=0.2):
        '''
        Args:
            in_size: input dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        x = self.dropout(x)
        y_1 = self.linear_1(x)
        return y_1    

class LateFusion(nn.Module):
    """Class implementing a late fusion model. All dimensions (as elsewhere in this file)
    are set to match the dimensions of CMU-MOSEI tensors for different modalities."""
    def __init__(self, audio=True, video=True, hidden_audio=32, hidden_video = 32,
                 hidden_text=64, audio_shape = 74, video_shape = 35, text_shape = 768,
                 post_fusion_dim = 32, drop_audio = 0.2, drop_video=0.2, drop_text = 0.2,
                 drop_post_fusion=0.2, out_dim=1):
        super(LateFusion, self).__init__()
        self.audio_in = audio_shape
        self.video_in = video_shape
        self.text_in = text_shape

        self.audio_hidden = hidden_audio
        self.video_hidden = hidden_video
        self.text_hidden = hidden_text
        self.text_out= hidden_text
        self.post_fusion_dim = post_fusion_dim
        self.out_dim = out_dim

        self.audio_prob = drop_audio
        self.video_prob = drop_video
        self.text_prob = drop_text
        self.post_fusion_prob = 0.2
    
        # define the pre-fusion subnetworks
        if audio:
            self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        if video:
            self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        if audio and video:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.audio_hidden+self.video_hidden+self.text_hidden, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        elif audio:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.audio_hidden+self.text_hidden, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        elif video:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.video_hidden+self.text_hidden, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        else:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.text_hidden, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
            
    
    def forward(self, audio_in, text_in, video_in, audio=True, video=True):
        """
        Args:
            audio_in (tensor): the audio input. This and the following models accept just 
                                audio and video inputs that were previously reduced to 
                                sentence level (e.g. by means of averaging), 
                                therefore having shape [batch_size, features_dimension]
            text_in (tensor): the text input
            video_in (tensor): the video input (see audio_in for details)
            audio (boolean): whether the audio modality is present or not
            video (boolean): whether the video modality is present or not"""
        if audio:
            audio_h = self.audio_subnet(audio_in)
        if video:
            video_h = self.video_subnet(video_in)
        text_h = self.text_subnet(text_in)
        if audio and video:
            combined = torch.cat((audio_h, text_h, video_h), dim=1)
        elif audio:
            combined = torch.cat((audio_h, text_h), dim=1)
        elif video:
            combined = torch.cat((text_h, video_h), dim=1)
        else:
            combined = text_h
        output = self.dense(combined)
        return (output,)
    
class EarlyFusion(nn.Module):
    """Class implementing early fusion architecture for CMU-MOSEI"""
    def __init__(self, audio= True, video= True, drop_in = 0.2, hidden_dim = 32,
                 audio_shape = 74, video_shape = 35, text_shape = 768, out_dim=1):
        super(EarlyFusion, self).__init__()
        self.audio_in = audio_shape
        self.video_in = video_shape
        self.text_in = text_shape
        self.post_fusion_dim = hidden_dim
        self.out_dim = out_dim
        self.post_fusion_prob = 0.2
        if audio and video:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.audio_in+self.video_in+self.text_in, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        elif audio:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.audio_in+self.text_in, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        elif video:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.video_in+self.text_in, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
        else:
            self.dense = nn.Sequential(nn.Dropout(self.post_fusion_prob),
            nn.Linear(self.text_in, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            nn.ReLU(),
            nn.Linear(self.post_fusion_dim, out_dim))
            
    def forward(self, audio_in, text_in, video_in, audio=True, video=True):
        """For details, see the comments to the forward function of LateFusion model"""
        if audio and video:
            combined = torch.cat((audio_in, text_in, video_in), dim=1)
        elif audio:
            combined = torch.cat((audio_in, text_in), dim=1)
        elif video:
            combined = torch.cat((text_in, video_in), dim=1)
        else:
            combined = text_in
        output = self.dense(combined)
        return (output,)
    
# Below code is taken and minimally adapted from https://github.com/Justin1904/TensorFusionNetworks
class TFN(nn.Module):
    def __init__(self):
        super(TFN, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.audio_in = 74
        self.video_in = 35
        self.text_in = 768

        self.audio_hidden = 32
        self.video_hidden = 32
        self.text_hidden = 128
        self.text_out= 768
        self.post_fusion_dim = 32

        self.audio_prob = 0.2
        self.video_prob = 0.2
        self.text_prob = 0.2
        self.post_fusion_prob = 0.2

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        
        
    
    def forward(self,c, intra, d, polarity=False, arousal = False):
        audio_h = self.audio_subnet(c)
        video_h = self.video_subnet(d)
        # text_h = self.text_subnet(intra)
        text_h = intra
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3 * self.output_range + self.output_shift
        
        outs = (output,)
        return outs
