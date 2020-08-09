# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:48:34 2020

@author: Iacopo
"""
import torch
import os
import pickle
import torch.optim as optim
import numpy as np
import random
from datasets import VideosSentenceDataset, VideosWordsDataset
import math
import io
from sklearn.preprocessing import normalize
from models import AttentionMultiModal, EarlyFusion, LateFusion, TFN
import pandas as pd
import argparse
import sys


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(
        description = 'Simple neural architectures and training functions for multimodal sentiment analysis on CMU-MOSEI')

parser.add_argument('--experiment_folder', '-folder', default='experiment', type=str,
                    help='Folder storing results and containing data for the current experiment.')

parser.add_argument('--data_dir', '-dir', default='data', type=str, 
                    help='Directory for training data.')

parser.add_argument('--AttentionModel', '-AM', action='store_true', default=False,
                    help="Use the attention model? If option is included then yes.")

parser.add_argument('--Transformer', '-trans', action='store_true', default=False,
                    help="Option for substituting the audio/video encoders in the attention based model (by default an LSTM) with a Transformer encoder")

parser.add_argument('--Convolution', '-conv', action='store_true', default=False,
                    help="Option for substituting the audio/video encoders in the attention based model (by default an LSTM) with a single convolutional layer having kernel=1")

parser.add_argument('--Raw', '-R', action='store_true', default=False,
                    help="Option for omitting the audio/video encoders in the attention based model (by default an LSTM), so that the attention mechanism is used directly on the provided features")

parser.add_argument('--ModifiedAttention', '-modat',action='store_true', default=False,
                    help="An experimental modification to the attention mechanism that makes the network use directly the attention scores instead of the attention weights in computing the contextual representation of the audio/video signals. This should theoretically has the effect of amplifying such modalities.")

parser.add_argument('--EarlyFusion', '-EF', action='store_true', default=False,
                    help="Use the Early Fusion model? If option is included then yes.")

parser.add_argument('--LateFusion', '-LF', action='store_true', default=False,
                    help="Use the Late Fusion model? If option is included then yes.")

parser.add_argument('--TensorFusion', '-TFN', action='store_true', default=False,
                    help="Use the Tensor Fusion model? If option is included then yes.")

parser.add_argument('--Normalise', '-norm', action='store_false', default=True,
                    help="Normalise audio/video tensors before inputting them in the models? The default is True, so if this option is included in the command line data will NOT be normalised (will turn to False). Normalisation is done to reduce every feature to the same scale and remove the effect of certain features (e.g. F0) as speaker's characteristic (e.g. gender) identifiers")

parser.add_argument('--training_file', '-trainf', default='df_MOSEI.tsv', type=str,
                    help="Name of the file (a tsv file or, anyway, a file that can be converted to a pandas dataframe) containing transcriptions, labels and ids for each training sentence.")

parser.add_argument('--validation_file', '-validf', default='df_valid_MOSEI.tsv', type=str,
                    help="Name of the file (a tsv file or, anyway, a file that can be converted to a pandas dataframe) containing transcriptions, labels and ids for each validation sentence.")

parser.add_argument('--test_file', '-testf', default='df_test_MOSEI.tsv', type=str,
                    help="Name of the file (a tsv file or, anyway, a file that can be converted to a pandas dataframe) containing transcriptions, labels and ids for each test sentence.")

parser.add_argument('--audio_file', '-af', default='COAVAREP_aligned_MOSEI.pkl', type=str,
                    help="Name of the audio file containing the extracted audio features: if changed, remember to change the audio shape option accordingly")

parser.add_argument('--video_file', '-vf', default='FACET_aligned_MOSEI.pkl', type=str,
                    help="Name of the video file containing the extracted video features: if changed, remember to change the video shape option accordingly")

parser.add_argument('--text_file', '-tf', default='BERT_MOSEI.pkl', type=str,
                    help="Name of the file containing the extracted text features (BERT's embeddings, currently'): if changed, remember to change the text shape option accordingly")

parser.add_argument('--exclude_audio', '-ea', action='store_true', default=False, 
                    help='Exclude audio modality? If option is included, then the modality is not included.')

parser.add_argument('--exclude_video', '-ev', action='store_true', default=False, 
                    help='Exclude video modality? If option is included, then the modality is not included.')

parser.add_argument('--audio_shape', '-ash', default=74, type=int,
                    help="Shape of the provided audio feature vectors (in CMU-MOSEI's provided COVAREP features this is 74, but the value can be changed so as to account for different features)")

parser.add_argument('--video_shape', '-vsh', default=35, type=int,
                    help="Shape of the provided video feature vectors (in CMU-MOSEI's provided FACET features this is 74, but the value can be changed so as to account for different features)")

parser.add_argument('--text_shape', '-tsh', default=768, type=int,
                    help="Shape of the provided text feature vectors (i.e. BERT's CLS embeddings)")

parser.add_argument('--output_dimension', '-out', default=1, type=int,
                    help="Dimension of the output: for sentiment analysis on CMU-MOSEI this is equal to 1, i.e. the sentiment label")

parser.add_argument('--hidden_audio_size', '-has', default=32, type=int, 
                    help='Number of hidden units in the audio encoder/subnetwork (same in each layer)')

parser.add_argument('--hidden_video_size', '-hvs', default=32, type=int, 
                    help='Number of hidden units in the video encoder/subnetwork (same in each layer)')

parser.add_argument('--hidden_text_size', '-hts', default=64, type=int, 
                    help='Number of hidden units in the text subnetwork (in use just in Late and Tensor Fusion)')

parser.add_argument('--hidden_size', '-hs', default=32, type=int, 
                    help='Number of hidden units in the fused, classification network (same in each layer)')

parser.add_argument('--num_layers', '-nl', default=1, type=int, 
                    help='Number of layers in the audio/video encoders of the attention based model. For now, just these encoders can have a variable number of layers, while for the others such an hyperparameter is fixed to what described in the accompanying paper to this code.')

parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, 
                    help='Learning rate')

parser.add_argument('--batch_size', '-bs', default=32, type=int, 
                    help='Mini batch size')

parser.add_argument('--num_epochs', '-ne', default=100, type=int, 
                    help='Number of training epochs')

parser.add_argument('--shuffle', '-sh', action='store_true', default=False, 
                    help='If included, training data are shuffled before input\
                    to the system.')

parser.add_argument('--bidirectional', '-bi', action='store_true', default=False, 
                    help='Create a bidirectional network? Default false.')

parser.add_argument('--early_stop', '-stop', default=5, type=int, 
                    help='Number of bad epochs after which to stop training (early stop)')

parser.add_argument('--dropout_in', '-di', default=0.0, type=float, 
                    help='Apply dropout to input of classification network: the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')

parser.add_argument('--dropout_out', '-do', default=0.0, type=float, 
                    help='Apply dropout to hidden layer(s): the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')
                    
parser.add_argument('--dropout_encoder_in', '-dei', default=0.0, type=float, 
                    help='Apply dropout directly to input features: the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')

parser.add_argument('--dropout_encoder_out', '-deo', default=0.0, type=float, 
                    help='Apply dropout to output of audio/video encoder: the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')

parser.add_argument('--optimizer', '-opt', default='Adam', type=str,
                    help='Optimizer to be used in train. Available options, at the moment, are: Adam or SGD')

parser.add_argument('--criterion', '-cr', default='MSE', type=str,
                    help="Loss function to be used in training. As the code currently support just the regression task on CMU-MOSEI' sentiment labels, available options are: MSE (mean squared error) or L1Loss (mean squared error)")

parser.add_argument('--clip', '-cl', default=4.0, type=float,
                    help='Define the value over which to clip the gradient for regolaring it.')

parser.add_argument('--save_model', '-save', action='store_true', default=False,
                    help='If included, the learned model will be saved in the saved model directory under the name model.pt')

parser.add_argument('--seed_value', default = 56, type=int,
                    help='The seed value determining the random initialisation of the parameters')

parser.add_argument('--From_checkpoint', '-checkpoint', action='store_true', default=False,
                    help="If the option is included, then starts training from a saved checkpoint.")

parser.add_argument('--save_outputs', '-save_out', action="store_true", default=False,
                    help="This option is used just during testing and, if included, makes the program save the output of the used metrics.")

args= parser.parse_args()

folder = args.experiment_folder
data_dir = args.data_dir
AttentionModel = args.AttentionModel
if AttentionModel:
    transformer = args.Transformer
    convolnet = args.Convolution
    raw = args.Raw
early_fusion = args.EarlyFusion
late_fusion = args.LateFusion
tensor_fusion = args.TensorFusion
if not AttentionModel and not early_fusion and not late_fusion and not TFN:
    raise Exception('At least one type of model must be selected by changing the relative voice in the hyperparameters file to TRUE!')
if AttentionModel ^ early_fusion ^ late_fusion ^ tensor_fusion:
    pass
else:
    print("WARNING: More than one model option has been selected")
    if AttentionModel:
        print('Attention model will be used')
    elif early_fusion:
        print('Early Fusion model will be used')
    elif late_fusion:
        print('Late Fusion model will be used')
    
if args.exclude_audio:
    audio = False
else:
    audio = True
if args.exclude_video:
    video = False
else:
    video = True
audio_shape = args.audio_shape
video_shape = args.video_shape
text_shape = args.text_shape
output_dimension = args.output_dimension
hidden_encoder_audio = args.hidden_audio_size
hidden_encoder_video = args.hidden_video_size
text_projection = args.hidden_text_size
hidden_units = args.hidden_size
NUM_LAYERS = args.num_layers
learning_rate = args.learning_rate
drop_in = args.dropout_in
drop_out = args.dropout_out
drop_enc_in = args.dropout_encoder_in
drop_enc_out = args.dropout_encoder_out
batch_size = args.batch_size
epochs = args.num_epochs
shuffle = args.shuffle
bidirectional = args.bidirectional
seed_value = args.seed_value
early_stop = args.early_stop
modified_attention = args.ModifiedAttention
clipping_value = args.clip
normalise = args.Normalise



# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
df = pd.read_csv(os.path.join(data_dir, args.training_file), delimiter=',', header=0, names=['source','sentence', 'polarity', 'hap', 'sad', 'ang', 'surpr', 'disg', 'fear'])
df_valid = pd.read_csv(os.path.join(data_dir, args.validation_file), delimiter=',', header=0, names=['source','sentence', 'polarity', 'hap', 'sad', 'ang', 'surpr', 'disg', 'fear'])

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# Set the seed value all over the place to make this reproducible.
seed_val = seed_value
random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

np.random.seed(seed_val)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Loading the data
with open(os.path.join(data_dir, args.text_file), 'rb') as f:
    if device.type=='cpu':
        feats = CPU_Unpickler(f).load()
    else:
        feats = pickle.load(f)
    
with open(os.path.join(data_dir, args.audio_file), 'rb') as f:
    COAVAREP = pickle.load(f)
    
with open(os.path.join(data_dir, args.video_file), 'rb') as f:
    FACET = pickle.load(f)

# Checking for alignment and corrupted values in the various modalities
audio_realigned_train = []
video_realigned_train = []
index_remove_train = []
for ind,s in enumerate(df.source.values):
    try:
        audio_realigned_train.append(COAVAREP[s]['features'])
        video_realigned_train.append(FACET[s]['features'])
    except KeyError:
        index_remove_train.append(ind)
        print('Training Sentence number {} was removed as not all modalities are present'.format(str(ind)))
        
audio_realigned_valid = []
video_realigned_valid = []
index_remove_valid = []
for ind,s in enumerate(df_valid.source.values):
    try:
        audio_realigned_valid.append(COAVAREP[s]['features'])
        video_realigned_valid.append(FACET[s]['features'])
    except KeyError:
        index_remove_valid.append(ind)
        print('Validation Sentence number {} was removed as not all modalities are present'.format(str(ind)))
        

# feats dictionary, containing BERT's embeddings, has all training, validation and test embeddings concatenated in sequential order
data = feats['Data'][:len(df)].tolist()
data_valid = feats['Data'][len(df):len(df)+len(df_valid)].tolist()

lab = df.polarity.values.tolist()
lab_valid = df_valid.polarity.values.tolist()

corrupted_audio = []

# Audio files present -inf values in some frame, that would make the whole training fail.
# Here I substitute such values in a verbose way, so as to have more control on them.
for step,batch in enumerate(audio_realigned_train):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_audio.append([step,ind])
            for inind,l in enumerate(value):
                if math.isinf(l) or math.isnan(l):
                    audio_realigned_train[step][ind][inind]=0
                    print("Audio Feature number {} in Frame {} of training sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))
                
for step,batch in enumerate(audio_realigned_valid):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_audio.append([step,ind])
            for inind,l in enumerate(value):
                if math.isinf(l) or math.isnan(l):
                    audio_realigned_valid[step][ind][inind]=0
                    print("Audio Feature number {} in Frame {} of validation sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))

                    
print('Corrupted audios: {}'.format(str(len(corrupted_audio))))

corrupted_video = []

# Video do not present -inf values currently, but in order to double check
# here I apply the same procedure as for the audio files.
for step,batch in enumerate(video_realigned_train):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_video.append([step,ind])
            for inind,l in enumerate(value):
                if math.isinf(l) or math.isnan(l):
                    video_realigned_train[step][ind][inind]=0
                    print("Video Feature number {} in Frame {} of training sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))
                
for step,batch in enumerate(video_realigned_valid):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_video.append([step,ind])
            if math.isinf(l) or math.isnan(l):
                    video_realigned_valid[step][ind][inind]=0
                    print("Video Feature number {} in Frame {} of validation sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))

                    
print('Corrupted video: {}'.format(str(len(corrupted_video))))

# If attention model and features need normalisation, then apply normalisation
# if not attention model and features need normalisation, apply both mean pooling and normalisation
# else (for non-attention models without normalisation) apply just mean pooling
if normalise and AttentionModel:
    audio_realigned_train = [normalize(el,norm='l2',axis=0) for el in audio_realigned_train]
    audio_realigned_valid = [normalize(el,norm='l2',axis=0) for el in audio_realigned_valid]
    video_realigned_train = [normalize(el,norm='l2',axis=0) for el in video_realigned_train]
    video_realigned_valid = [normalize(el,norm='l2',axis=0) for el in video_realigned_valid]
elif normalise:
    audio_realigned_train = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in audio_realigned_train]
    audio_realigned_valid = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in audio_realigned_valid]
    video_realigned_train = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in video_realigned_train]
    video_realigned_valid = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in video_realigned_valid]
elif not AttentionModel:
    audio_realigned_train = [np.mean(el, axis= 0) for el in audio_realigned_train]
    audio_realigned_valid = [np.mean(el, axis= 0) for el in audio_realigned_valid]
    video_realigned_train = [np.mean(el, axis= 0) for el in video_realigned_train]
    video_realigned_valid = [np.mean(el, axis= 0) for el in video_realigned_valid]

if index_remove_train:
    for ind in index_remove_train:
        data.pop(ind)
        lab.pop(ind)
if index_remove_valid:
    for ind in index_remove_valid:
        data_valid.pop(ind)
        lab_valid.pop(ind)

BATCH_SIZE = batch_size
shuffle = shuffle

# Different dataset classes are used whether the audio/video tensors are one per sentence
# or more (as in attention model). In the latter case the dataset applies padding
if not AttentionModel:
    train_dataset = (audio_realigned_train, lab, data, video_realigned_train)

    val_dataset = (audio_realigned_valid, lab_valid, data_valid, video_realigned_valid)
    
    training_2_batch = VideosSentenceDataset(train_dataset)
    dataset = torch.utils.data.DataLoader(dataset=training_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = training_2_batch.collater)
    
    valid_2_batch = VideosSentenceDataset(val_dataset)
    valid_dataset = torch.utils.data.DataLoader(dataset=valid_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = valid_2_batch.collater)
else:
    train_dataset = (audio_realigned_train, lab, data, video_realigned_train)

    val_dataset = (audio_realigned_valid, lab_valid, data_valid, video_realigned_valid)
    
    training_2_batch = VideosWordsDataset(train_dataset)
    dataset = torch.utils.data.DataLoader(dataset=training_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = training_2_batch.collater)
    
    valid_2_batch = VideosWordsDataset(val_dataset)
    valid_dataset = torch.utils.data.DataLoader(dataset=valid_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = valid_2_batch.collater)

# Instantiate the chosen model with the provided options and hyperparameters
if early_fusion:
    model = EarlyFusion(audio=audio, video=video, 
                        hidden_dim=hidden_units, 
                        drop_in = drop_in,
                        audio_shape = audio_shape,
                        video_shape = video_shape,
                        text_shape = text_shape,
                        out_dim = output_dimension)
elif late_fusion:
    """Instantiating late fusion model. In this context, the dropout values
    defined above are re-defined as follow:
        drop_enc_in = dropout applied to the audio features at the input level
        drop_enc_out = dropout applied to the video features at the input level
        drop_out = dropout applied to the text features at the input level
        drop_in = dropout applied to the concatenated audio, video and text features as obtained by the respective subnetworks"""
    model = LateFusion(audio=audio, video=video,
                       hidden_audio = hidden_encoder_audio,
                       hidden_video = hidden_encoder_video, 
                       hidden_text = text_projection,
                       audio_shape = audio_shape,
                       video_shape = video_shape,
                       text_shape = text_shape,
                       post_fusion_dim = hidden_units,
                       drop_audio = drop_enc_in,
                       drop_video = drop_enc_out,
                       drop_text = drop_out,
                       drop_post_fusion = drop_in,
                       out_dim = output_dimension)
elif tensor_fusion:
    """The present tensor fusion network implementation does not allow for changes 
    in the hyperparameters at this stage. The hyperparameters that are currently available
    reflect the implementation from the original paper and can be consulted in the model
    implementation itself."""
    model = TFN()
else:
    model = AttentionMultiModal(audio=audio, video=video, 
                       transformer=transformer, 
                       convolutional=convolnet, 
                       raw=raw, 
                       drop_in = drop_in,
                       drop_out = drop_out,
                       drop_enc_in = drop_enc_in,
                       drop_enc_out = drop_enc_out, 
                       hidden_encoder_audio = hidden_encoder_audio, 
                       hidden_encoder_video = hidden_encoder_video, 
                       bidirectional = bidirectional, # This option can be applied just if LSTM encoders are chosen
                       text_shape = text_shape,
                       audio_shape = audio_shape,
                       video_shape = video_shape,
                       attn_project = hidden_units,
                       nlayers = NUM_LAYERS, # this option controls the number of layers in the audio/video encoders 
                       out_dim = output_dimension) 

# If cuda is used, move the model to GPU. Also, if the transformer is used as an encoder
# and cuda is used, at this stage move the positional embeddings to GPU as well
if torch.cuda.is_available():
    model.cuda()
    if transformer:
      model.audio.pos_encoder.pe = model.audio.pos_encoder.pe.to(device)
      model.video.pos_encoder.pe = model.video.pos_encoder.pe.to(device)

if args.From_checkpoint:
    model.load_state_dict(torch.load(os.path.join(folder,'checkpoint','checkpoint.bin')))

if args.criterion=='MSE':
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.L1Loss()

if args.optimizer=='Adam':
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

epochs = epochs

bad_epoch = 0

import time
import datetime
from models import create_mask

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

training = []
valid = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(dataset):
        # Progress update every 40 batches.
        if step % 20 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataset), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU (if cuda is available) using the 
        # `to` method.
        if torch.cuda.is_available():
            b_audio = batch['src_tokens'].to(device).float()
            b_labels = batch['tgt_tokens'].to(device).float().squeeze() # delete the additional batch dimension with squeeze()
            b_text = batch['text'].to(device).float()
            b_video = batch['video'].to(device).float()
            b_len = batch['src_lengths'].to(device) # the single sentence lengths are used in the attention model (in other architectures is provided but never used)
            src_mask = create_mask(b_audio, b_len).to(device) # the mask is used in the attention mechanism (in other architectures is provided but never used)
        else:
            b_audio = batch['src_tokens'].float()
            b_labels = batch['tgt_tokens'].float().squeeze() # delete the additional batch dimension with squeeze()
            b_text = batch['text'].float()
            b_video = batch['video'].float()
            b_len = batch['src_lengths'] # the single sentence lengths are used in the attention model (in other architectures is provided but never used)
            src_mask = create_mask(b_audio, b_len) # the mask is used in the attention mechanism (in other architectures is provided but never used)
        
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()


        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        
        if AttentionModel:
            output = model(b_audio, b_text, b_video, audio=audio, video=video, mask=src_mask, line_len=b_len, modified_attention=modified_attention)  # attention model forward pass, the modified attention is an additional option that amplify modalities other than text
        
        else:
            output = model(b_audio, b_text, b_video, audio=audio, video=video) # for Early, Late and Tensor Fusion, no additional inputs/options are needed in the forward pass
        
        
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        
        loss = criterion(output[0].view(-1), b_labels.view(-1))
        
            
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to the clipping value.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        
        optimizer.step()
        

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(dataset)
    training.append(avg_train_loss)           
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    

    # Evaluate data for one epoch
    for step, batch in enumerate(valid_dataset):
    # for step, batch in enumerate(validation_dataloader):
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU (if cuda is used) using 
        # the `to` method.
        if torch.cuda.is_available():
            b_audio = batch['src_tokens'].to(device).float()
            b_labels = batch['tgt_tokens'].to(device).float().squeeze() # delete the additional batch dimension with squeeze()
            b_text = batch['text'].to(device).float()
            b_video = batch['video'].to(device).float()
            b_len = batch['src_lengths'].to(device) # the single sentence lengths are used in the attention model (in other architectures is provided but never used)
            src_mask = create_mask(b_audio, b_len).to(device) # the mask is used in the attention mechanism (in other architectures is provided but never used)
        else:
            b_audio = batch['src_tokens'].float()
            b_labels = batch['tgt_tokens'].float().squeeze() # delete the additional batch dimension with squeeze()
            b_text = batch['text'].float()
            b_video = batch['video'].float()
            b_len = batch['src_lengths'] # the single sentence lengths are used in the attention model (in other architectures is provided but never used)
            src_mask = create_mask(b_audio, b_len) # the mask is used in the attention mechanism (in other architectures is provided but never used)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            if AttentionModel:
                output = model(b_audio, b_text, b_video, mask=src_mask, line_len=b_len, modified_attention=modified_attention)  # attention model forward pass, the modified attention is an additional option that amplify modalities other than text
        
            else:
                output = model(b_audio, b_text, b_video, audio=audio, video=video) # for Early, Late and Tensor Fusion, no additional inputs/options are needed in the forward pass
        
            loss = criterion(output[0].view(-1), b_labels.view(-1))
            
        total_eval_loss += loss.item()

        

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(valid_dataset)
    valid.append(avg_val_loss)
    
    if epoch_i>0:
        if avg_val_loss <= best_val:
          torch.save(model.state_dict(),os.path.join(folder,'checkpoint','checkpoint.bin'))
          best_val = avg_val_loss
          bad_epoch = 0
        else:
            bad_epoch +=1
    else:
        best_val = avg_val_loss
        torch.save(model.state_dict(),os.path.join(folder,'checkpoint','checkpoint.bin'))
    
    
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
    if bad_epoch==early_stop:
        print('No improvements over the last {} epochs: early stop.'.format(str(early_stop)))
        break

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

if args.save_model:
    """Model is saved in the standard results folder, created with the prepare_workspace function"""
    model.load_state_dict(torch.load(os.path.join(folder,'checkpoint','checkpoint.bin')))
    torch.save(model.state_dict(), os.path.join(folder, 'saved_model','best_model.bin'))