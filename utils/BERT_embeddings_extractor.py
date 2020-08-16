import argparse

parser = argparse.ArgumentParser(description='Extractor for BERT embeddings from CMU-MOSEI.')
parser.add_argument('--emotion', '-e', action='store_true', default=False,
                    help='If option is included, then extract embeddings fine-tuned on emotion labels (instead of sentiment)')
parser.add_argument('--max_length', '-max' type='int', default=130,
                    help='maximum sentence length (if any sentence is longer than the maximum length, then it is chopped up to the nth token as per this value.)')
                    
parser.add_argument('--use_pca', '-pca', action='store_true', default=False,
                    help='use pca to reduce the dimension of the output embeddings from BERT before saving them.')

args = parser.parse_args()
emotion = args.emotion
use_pca = args.use_pca

import torch

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

import os
os.chdir("../data") # change directory to the data folder containing the dataset's transcriptions

import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("df_MOSEI.tsv", delimiter=',', header=0, names=['source','sentence', 'level', 'happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear'])
df_valid = pd.read_csv("df_valid_MOSEI.tsv", delimiter=',', header=0, names=['source','sentence', 'level', 'happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear'])
df_test = pd.read_csv("df_test_MOSEI.tsv", delimiter=',', header=0, names=['source','sentence', 'level', 'happiness', 'sadness', 'anger', 'surprise', 'disgust', 'fear'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
print(df.sample(10))

# Get the lists of sentences and their labels.
sentences = df.sentence.values
if emotion:
  level = df.iloc[:,3:].values
  for index, k in enumerate(level):
    level[index] = [int(log>0) for log in k]
else:
  level = df.level.values
sents_valid = df_valid.sentence.values
if emotion:
  level_valid = df_valid.iloc[:,3:].values
  for index, k in enumerate(level_valid):
    level_valid[index] = [int(log>0) for log in k]
  else:
    level_valid = df_valid.level.values

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 0

# For every sentence...
for index, sent in enumerate(sentences):

    # sentences[index] = re.sub("(^|[\s])sp([\s]|$)"," =sp ", sent).strip()

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sentences[index], add_special_tokens=True)
    
    

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
    

print('Max sentence length: ', max_len)
max_len = args.max_length

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
counter = 0
# For every sentence...
for sent in sentences:
    counter+=1
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
level = torch.tensor(level)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])


# Same for validation set
input_ids_valid = []
attention_masks_valid = []

# For every sentence...
for sent in sents_valid:
    # sent = re.sub("(^|[\s])sp([\s]|$)"," =sp ", sent).strip()
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids_valid.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks_valid.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids_valid = torch.cat(input_ids_valid, dim=0)
attention_masks_valid = torch.cat(attention_masks_valid, dim=0)
level_valid = torch.tensor(level_valid)

# Print sentence 0, now as a list of IDs.
print('Original: ', sents_valid[0])
print('Token IDs:', input_ids_valid[0])

val = len(input_ids_valid)
from torch.utils.data import TensorDataset

# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(input_ids, attention_masks, level)
val_dataset = TensorDataset(input_ids_valid, attention_masks_valid, level_valid)

print('{:>5,} training samples'.format(counter))
print('{:>5,} validation samples'.format(val))

from torch.utils.data import DataLoader, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = SequentialSampler(train_dataset), # Select batches sequentially (so embeddings will appear in the same order as transcriptions)
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
            )

import numpy as np
import random

# Set the seed value all over the place to make this reproducible.
seed_val = args.seed_value

random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)

np.random.seed(seed_val)

torch.backends.cudnn.deterministic = True

# Pre-built classifier with BERT--> it is a very simple architecture and can be simply copied from the source code to have more control over it

from transformers import BertPreTrainedModel, BertModel, AdamW, BertConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss

class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, self.config.num_labels)
        self.task_1 = nn.Linear(768, 2) # Here two additional output layers are added so that is possible to fine-tune BERT embeddings with a multi-task framework (predicting polarity and arousal)
        self.task_2 = nn.Linear(768, 4)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                training=True, labeltask1 = None, labeltask2 = None, fine_tune=False,
                loss_weights = pos_w):
        
        if fine_tune:
          with torch.no_grad():

            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)

            pooled_output = outputs[1]
        else:
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)

            pooled_output = outputs[1]
            last_hidden = outputs[0]


        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,pooled_output) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression (sentiment analysis)
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Multi-label fine-tuning for emotions (emotion recognition)
                loss_fct = BCEWithLogitsLoss(torch.tensor(loss_weights).to(device))
                loss = loss_fct(logits, labels)
            
            # If additional polarity (and strength) are input in the model, train BERT with multitask objective
            
            if isinstance(labeltask1,torch.Tensor):
                loss_fn = CrossEntropyLoss()
                logitstask1 = self.task_1(pooled_output)
                losstask1 = loss_fn(logitstask1.view(-1, 2), labeltask1.view(-1))
                if isinstance(labeltask2,torch.Tensor):
                    logitstask2 = self.task_2(pooled_output)
                    losstask2 = loss_fn(logitstask2.view(-1, 4), labeltask2.view(-1))
                    loss = loss + 0.5*losstask1 + 0.5*losstask2
                else:
                  loss = loss+ 0.5*losstask1
                
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

labs_number = 6 if emotions else 1 # Number of labels over which to fine-tune is 6 if fine-tuning on emotions and 1 if fine-tuning on sentiment

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 6, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
if torch.cuda.is_available():
  model.cuda()
  
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import time
import datetime

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
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).float()

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

        loss, logits, _, _ = model(b_input_ids, attention_mask= b_input_mask,
                             labels=b_labels) # use it for pre-built classifier

        
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
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
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            

            loss, logits, _, _ = model(b_input_ids, attention_mask=b_input_mask,
                                 labels=b_labels) # use it with pre-built model

            
        # Accumulate the validation loss.
        
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        
        
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    # Report the final accuracy for this validation run.
    if epoch_i>0:
      if avg_val_loss >= best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(),'finetuned_model.bin')
    else:
        best_loss = avg_val_loss
        torch.save(model.state_dict(),'finetuned_model.bin')

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
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# 2nd Part: use the fine-tuned BERT to extract relevant features 
from transformers import BertPreTrainedModel, BertModel, AdamW, BertConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
feature_extractor = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = labs_number, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = True, # Whether the model returns all hidden-states.
)

feature_extractor.load_state_dict(torch.load("finetuned_model.bin"))
if torch.cuda.is_available():
  feature_extractor.cuda()

epochs = 1

import time
import datetime
words = []
feature_matrix = []
labels = []

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
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).float()

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        feature_extractor.zero_grad()


        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.

        with torch.no_grad():
            logits, CLS, hid = feature_extractor(b_input_ids, attention_mask=b_input_mask)
        words.append(hid[0])
        feature_matrix.append(CLS)
        labels.append(b_labels)
        # polarities.append(b_polarity)
        # arousals.append(b_arousal)

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device).float()
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits, CLS, hid = feature_extractor(b_input_ids, attention_mask=b_input_mask)
        words.append(hid[0]) 
        feature_matrix.append(CLS)
        labels.append(b_labels)
        
# Create sentence and label lists
sentences = df_test.sentence.values
if emotions:
  labels_test = df_test.iloc[:,3:].values
else:
  labels_test = df_test.level.values



# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels_test = torch.tensor(labels_test)


# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels_test)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
# Put model in evaluation mode
feature_extractor.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  
  # Unpack the inputs from our dataloader
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits,out, hid = feature_extractor(b_input_ids, attention_mask=b_input_mask)
      words.append(hid[0])
      feature_matrix.append(out)
      labels.append(b_labels)
      # polarities.append(polarity)
      # arousals.append(arousal)

  

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

#  Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_predictions = flat_predictions.flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

flat_polarities = [int(log>=0) for log in flat_predictions]
true_polarities = [int(log>=0) for log in flat_true_labels]

# Calculate the MCC
cc = pearsonr(flat_true_labels, flat_predictions)[0]
MAE = mean_absolute_error(flat_true_labels, flat_predictions)
acc = accuracy_score(flat_polarities, true_polarities)
f1 = f1_score(true_polarities, flat_polarities)
print('Total CC from simple BERT: %.3f' % cc) # --> Double Check the results from the feature extractors correspond to previous ones
print('Total MAE from simple BERT: %.3f' % MAE)
print('Total Acc from simple BERT: %.3f' % acc)
print('Total F1 from simple BERT: %.3f' % f1)

import pickle

if use_pca:
  from sklearn.decomposition import PCA # --> an option to reduce the dimensionality of the extracted features
  DATA = torch.cat(feature_matrix, dim=0).cpu()
  last_hiddens = torch.cat([words[k] for k in range(len(words))],dim=0).cpu()
  transformer = PCA(n_components=32)
  Data_transform = transformer.fit_transform(DATA)
  labels = [lab.cpu().float() for lab in labels]
  lab_tot = torch.cat(labels,dim=0)
  tot_arousals = torch.cat(arousals, dim=0)
  tot_polarities = torch.cat(polarities, dim=0)

  pca_dict = {'Data':Data_transform, 'level':lab_tot, 'polarities':tot_polarities, 'arousals':tot_arousals}
  
  with open('reduced_features.pkl', 'wb') as f:
      pickle.dump(pca_dict, f)

else:
  Bert_dict = {'Data':DATA, 'words': last_hiddens,'level':lab_tot, 'polarities':tot_polarities, 'arousals':tot_arousals}
  
  with open('BERT_features.pkl', 'wb') as f:
    pickle.dump(Bert_dict, f)

