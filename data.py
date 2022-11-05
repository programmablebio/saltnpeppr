import esm
import random

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import pytorch_lightning as pl

from glob import glob
import numpy as np

import time
import pickle
import pandas as pd
from os import walk
from torch.utils.data import Dataset

class protein_dataset(Dataset):
    # pandas dataframe -> dataset

    def __init__(self, path):

        df = pd.read_csv(path)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

class dataset_collator:
    # processes the pandas dataframe row into model inputs and targets

    def __init__(self):

        alphabet = esm.Alphabet.from_architecture('ESM-1b')
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, raw_batch):

        batch = {}
        proteins = []
        peptides = []
        msa_ids = []
        peptide_classes = []
        peptide_scores = []

        _, _, pep_token = self.batch_converter([(0, raw_batch[0]['peptide_derived_sequence'])])

        batch = {}
        batch['peptide_source'] = pep_token

        energy_list = raw_batch[0]['mean_score'][1:-1].replace('\n', '').split(' ')
        energy_list = ' '.join(energy_list).split()
        energy_list = [float(escore) for escore in energy_list]
        energy_list = [float(escore) for escore in energy_list]
        energy_list = [0 if ele > 0 else ele for ele in energy_list]

        batch['peptide_scores'] = (torch.FloatTensor(energy_list) <= -1).unsqueeze(dim=1)
        batch['peptide_scores'] = batch['peptide_scores'].type(torch.LongTensor)
        batch['peptide_scores_value'] = (torch.FloatTensor(energy_list)/100.0).unsqueeze(dim=1)
        batch['id'] = raw_batch[0]['new_id']

        return batch

class ProteinPeptideDataModule(pl.LightningDataModule):

    def __init__(self,
                dataset_path):
        super().__init__()

        self.dataset_path = dataset_path

    def setup(self, stage):
        self.test_ds = protein_dataset(self.dataset_path+'testing_dataset_mean.csv')
        self.val_ds = protein_dataset(self.dataset_path+'validation_dataset_mean.csv')
        self.train_ds = protein_dataset(self.dataset_path+'training_dataset_mean.csv')

        self.collator = dataset_collator()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = 1, collate_fn=self.collator, num_workers=8, drop_last=False, pin_memory = True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = 1, collate_fn=self.collator, num_workers=8, drop_last=False, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size = 1, collate_fn=self.collator, num_workers=8, drop_last=False, pin_memory = True)
