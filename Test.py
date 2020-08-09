# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:47:39 2020

@author: Iacopo
"""

import torch
import os
import pickle
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
df_test = pd.read_csv(os.path.join(data_dir, args.test_file), delimiter=',', header=0, names=['source','sentence', 'polarity', 'hap', 'sad', 'ang', 'surpr', 'disg', 'fear'])


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# Set the seed value all over the place to make this reproducible.
seed_val = seed_value
random.seed(seed_val)

torch.manual_seed(seed_val)

# torch.cuda.manual_seed_all(seed_val)

np.random.seed(seed_val)

with open(os.path.join(data_dir, args.text_file), 'rb') as f:
    if device.type=='cpu':
        feats = CPU_Unpickler(f).load()
    else:
        feats = pickle.load(f)

    
with open(os.path.join(data_dir, args.audio_file), 'rb') as f:
    COAVAREP = pickle.load(f)
    
with open(os.path.join(data_dir, args.video_file), 'rb') as f:
    FACET = pickle.load(f)

audio_realigned_test = []
video_realigned_test = []
index_remove_test = []
for ind,s in enumerate(df_test.source.values):
    try:
        audio_realigned_test.append(COAVAREP[s]['features'])
        video_realigned_test.append(FACET[s]['features'])
    except KeyError:
        index_remove_test.append(ind)
        print('Test Sentence number {} was removed as not all modalities are present'.format(str(ind)))
        



data = feats['Data'][len(df)+len(df_valid):].tolist()
test_label = df_test.polarity.values.tolist()


corrupted_audio = []
for step,batch in enumerate(audio_realigned_test):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_audio.append([step,ind])
            for inind, l in enumerate(value):
                if math.isinf(l) or math.isnan(l):
                    audio_realigned_test[step][ind][inind]=0
                    print("Audio Feature number {} in Frame {} of sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))
                    
print('Corrupted audios: {}'.format(str(len(corrupted_audio))))

corrupted_video = []
for step,batch in enumerate(video_realigned_test):
    for ind, value in enumerate(batch):
        if math.isinf(np.mean(value)):
            corrupted_video.append([step,ind])
            for inind, l in enumerate(value):
                if math.isinf(l) or math.isnan(l):
                    video_realigned_test[step][ind][inind]=0
                    print("Video Feature number {} in Frame {} of sentence {} contains nan or infinite value: substituting it with 0".format(str(inind), str(ind), str(step)))
                    
print('Corrupted video: {}'.format(str(len(corrupted_video))))



if normalise and AttentionModel:
    audio_realigned_test = [normalize(el,norm='l2',axis=0) for el in audio_realigned_test]
    video_realigned_test = [normalize(el,norm='l2',axis=0) for el in video_realigned_test]

elif normalise:
    audio_realigned_test = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in audio_realigned_test]
    video_realigned_test = [np.mean(normalize(el,norm='l2',axis=0), axis= 0) for el in video_realigned_test]
elif not AttentionModel:
    audio_realigned_test = [np.mean(el, axis= 0) for el in audio_realigned_test]
    video_realigned_test = [np.mean(el, axis= 0) for el in video_realigned_test]


if index_remove_test:
    for ind in index_remove_test:
        data.pop(ind)
        test_label.pop(ind)    

    
BATCH_SIZE = batch_size
shuffle = shuffle

# Different dataset classes are used whether the audio/video tensors are one per sentence
# or more (as in attention model). In the latter case the dataset applies padding
if not AttentionModel:
    test_dataset = (audio_realigned_test, test_label, data, video_realigned_test)

    test_2_batch = VideosSentenceDataset(test_dataset)
    dataset = torch.utils.data.DataLoader(dataset=test_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = test_2_batch.collater)
    
    
else:
    test_dataset = (audio_realigned_test, test_label, data, video_realigned_test)
    
    test_2_batch = VideosWordsDataset(test_dataset)
    dataset = torch.utils.data.DataLoader(dataset=test_2_batch,
                                          batch_size = BATCH_SIZE,
                                          collate_fn = test_2_batch.collater)
    
    
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

assert os.path.exists(os.path.join(folder,'saved_model','best_model.bin')), "Couldn't find any saved model to load: check that the trained model's parameters are correctly saved in the saved_model folder"
try:
    model.load_state_dict(torch.load(os.path.join(folder,'saved_model','best_model.bin')))
except:
    model.load_state_dict(torch.load(os.path.join(folder,'saved_model','best_model.bin'), map_location=torch.device('cpu')))

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



print('Predicting labels for {:,} test sentences...'.format(len(test_label)))
# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels, pred_pol, true_pol = [], [], [], []

for batch in dataset:
    
    t0 = time.time()
    
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
    
    with torch.no_grad():


        if AttentionModel:
            output = model(b_audio, b_text, b_video, audio=audio, video=video, mask=src_mask, line_len=b_len, modified_attention=modified_attention)  # attention model forward pass, the modified attention is an additional option that amplify modalities other than text
        
        else:
            output = model(b_audio, b_text, b_video, audio=audio, video=video) # for Early, Late and Tensor Fusion, no additional inputs/options are needed in the forward pass
        
  
  
    # Move logits and labels to CPU
    logits = output[0].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    pred_pol.append(torch.tensor([int(log>=0) for log in logits]))
    true_polar_b = torch.tensor([int(log>=0) for log in b_labels])
    true_pol.append(true_polar_b)
  
print('    DONE.')

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#  Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_predictions = flat_predictions.flatten()

flat_polarities = np.concatenate(pred_pol, axis=0)
flat_polarities = flat_polarities.flatten()

true_polarities = np.concatenate(true_pol, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the MCC
cc = pearsonr(flat_true_labels, flat_predictions)[0]
MAE = mean_absolute_error(flat_true_labels, flat_predictions)
acc = accuracy_score(flat_polarities, true_polarities)
f1 = f1_score(true_polarities, flat_polarities)

print('Total CC: %.3f' % cc)
print('Total MAE: %.3f' % MAE)
print('Total Acc: %.4f' % acc)
print('Total F1: %.4f' % f1)

new_true_polarities = true_polarities.tolist()
new_pred_polarities = flat_polarities.tolist()
counter = 0
for ind,el in enumerate(true_polarities):
    if flat_true_labels[ind]==0:
        new_true_polarities.pop(ind-counter)
        new_pred_polarities.pop(ind-counter)
        counter+=1
        
acc_new = accuracy_score(new_pred_polarities, new_true_polarities)
f1_new = f1_score(new_true_polarities, new_pred_polarities)

print('Accuracy without neutral samples: %.4f' % acc_new)
print('F1 without neutral samples: %.4f' % f1_new)

pred_polarities7 = np.digitize(flat_predictions, [-3,-2,-1,0,1,2,3])
true_polarities7 = np.digitize(flat_true_labels,[-3,-2,-1,0,1,2,3])

acc_7 = accuracy_score(pred_polarities7, true_polarities7)
f1_7 = f1_score(pred_polarities7, true_polarities7,average='weighted')

print('Accuracy (7 classes): %.4f' % acc_7)
print('F1 (7 classes): %.4f' % f1_7)


print('Total number of learnable parameters: %.1f' % sum(p.numel() for p in model.parameters() if p.requires_grad))

if args.save_outputs:
    out_df = pd.DataFrame({'metrics':['MAE', 'CC', 'binary accuracy (with neutral labels)', 'F1 (with neutral labels)',
                                       'binary accuracy (without neutral labels)', 'F1 (without neutral labels)', 
                                       '7-class accuracy', '7-class weighted F1'],
                            'values':[MAE, cc, acc, f1, acc_new, f1_new, acc_7, f1_7]})
    
    out_df.to_csv(os.path.join(folder, 'outputs','outputs.csv'))