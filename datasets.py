# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:02:35 2020

@author: Iacopo
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class VideosSentenceDataset(Dataset):
    def __init__(self, lines):
        self.src_dataset = [line for line in lines[0]]
        self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])
        self.text = [line for line in lines[2]]
        self.tgt_dataset = [line for line in lines[1]]
        self.video = [line for line in lines[3]]
        
    
    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'source': torch.tensor(self.src_dataset[index]),
            'target': self.tgt_dataset[index],
            'text': torch.tensor(self.text[index]).float(),
            'video': torch.tensor(self.video[index])
        }
        
    def __len__(self):
        return len(self.src_dataset)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, continuous=False):
            if len(values[0].shape)<2:
              return torch.stack(values)
            else:
              max_length = max(v.size(0) for v in values)
              result = torch.zeros((len(values),max_length, values[0].shape[1]))
              for i, v in enumerate(values):
                  result[i, :len(v)] = v
              return result

        id = torch.tensor([s['id'] for s in samples])
        # src_tokens = merge([s['source'] for s in samples])
        src_tokens = merge([s['source'].float() for s in samples])
        tgt_tokens = torch.tensor([s['target'] for s in samples])
        text = torch.stack([s['text'] for s in samples])
        src_lengths = torch.LongTensor([s['source'].shape[0] for s in samples])
        video = merge([s['video'].float() for s in samples])

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'text': text,
            'video': video
        }
    
class VideosWordsDataset(Dataset):
    def __init__(self, lines):
        self.src_dataset = [line for line in lines[0]]
        self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])
        self.text = [line for line in lines[2]]
        self.tgt_dataset = [line for line in lines[1]]
        self.video = [line for line in lines[3]]
        
    
    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'source': torch.tensor(self.src_dataset[index]),
            'target': self.tgt_dataset[index],
            'text': torch.tensor(self.text[index]).float(),
            'video': torch.tensor(self.video[index])
        }
        
    def __len__(self):
        return len(self.src_dataset)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, continuous=False):
            max_length = max(v.size(0) for v in values)
            result = torch.zeros((len(values),max_length, values[0].shape[1]))
            for i, v in enumerate(values):
                result[i, :len(v)] = v
            return result

        id = torch.tensor([s['id'] for s in samples])
        # src_tokens = merge([s['source'] for s in samples])
        src_tokens = merge([s['source'] for s in samples])
        tgt_tokens = torch.tensor([s['target'] for s in samples])
        text = torch.stack([s['text'] for s in samples])
        src_lengths = torch.LongTensor([s['source'].shape[0] for s in samples])
        video = merge([s['video'] for s in samples])

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'text': text,
            'video': video
        }