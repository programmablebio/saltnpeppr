import torch
from model import SALTnPEPPR
from data import ProteinPeptideDataModule

import pytorch_lightning as pl
import os
from Bio import SeqIO
import pickle
import json
import esm
import itertools
import string
import pandas as pd
from torch.nn import LogSoftmax

binder_name = 'ACE2'
target_name = 'RBD_short'

data_path = '/peptides/protein_peptide/data'
binder_fasta = '/fastas/'+target_name+'/'+binder_name+'.fasta'
target_MSA = '/hhr_outs/'+target_name+'.msa'
model_path ='path_to_model_weights'

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename):
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence):
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename, nseq):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

# binder FASTA
_, binder_seq = read_sequence(data_path+binder_fasta)

binder_aa_list = []
for i in range(len(binder_seq)):
    binder_aa_list.append(binder_seq[i])

alphabet = esm.Alphabet.from_architecture('ESM-1b')
batch_converter = alphabet.get_batch_converter()
binder_tuple = [(1, binder_seq)]
batch_labels, batch_strs, batch_tokens = batch_converter(binder_tuple)

### model
model = SALTnPEPPR.load_from_checkpoint(model_path)

model.eval()
scores = model(batch_tokens, None)
scores = torch.exp(softmax(scores))

npscores = scores[:,1].detach().numpy()
df = pd.DataFrame({'AA':binder_aa_list, 'Binder Prob': npscores, 'Source': binder_name,'Target':target_name})

df.to_csv('./'+target_name+'_'+binder_name+'_prediction.csv')
