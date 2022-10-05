import randomname

import torch
from model import SALTnPEPPR
from data import ProteinPeptideDataModule
import os
import pytorch_lightning as pl
import argparse
import pickle
from pytorch_lightning.loggers import WandbLogger
import json

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

parser = argparse.ArgumentParser(description='Train SALTnPEPPR')

##dataloader args:
parser.add_argument('--dataset_path', default='/peptides/protein_peptide/data/')

#training args:
parser.add_argument('--lr', dest="lr", default=5e-6, type=float)
parser.add_argument('--grad_accum', dest="grad_accum", default=16, type=float)
parser.add_argument('--n_gpus', dest='n_gpus', type=int, default=[0])
parser.add_argument('--precision', dest='precision', default=16)
parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=150)

parser.add_argument('--beta1', dest='beta1', type=float, default=0.9)
parser.add_argument('--beta2', dest='beta2', type=float, default=0.98)

parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--early_stopping_patience', dest='early_stopping_patience', type=int, default=50)

parser.add_argument('--anomaly_detection', action='store_true', default=False)

#model args:
parser.add_argument('--random_seed', dest='random_seed', type=int, default=42)
parser.add_argument('--model_id', dest='model_id', default=None)
parser.add_argument('--mlp_dim', dest="mlp_dim", type=int, default=5000)
parser.add_argument('--mlp_layers', dest="mlp_layers", type=int, default=4)

parser.add_argument('--dropout', dest="dropout", type=float, default=0.2)
parser.add_argument('--learn_temperature', dest='learn_temperature', action='store_true')

#saving args
parser.add_argument('--resume_checkpoint', dest='resume_checkpoint', default=None)
parser.add_argument('--save_top_k', dest='save_top_k', default=5, type=int)

args, _ = parser.parse_known_args()
config = vars(args)

def random_name():
    return randomname.get_name(
        adj=('speed', 'emotions', 'temperature', 'weather', 'character', 'algorithms', 'geometry', 'complexity', 'physics', 'shape', 'taste', 'colors', 'size', 'appearance'),
        noun=('astronomy', 'infrastructure', 'chemistry', 'physics', 'geometry', 'coding', 'architecture', 'metals', 'apex_predators')
    )

##name model:
if args.model_id == None:
    config['model_id'] = random_name()
else:
    config['model_id'] = args.model_id
    

datamodule = ProteinPeptideDataModule(dataset_path=args.dataset_path)  

root_dir = os.path.join("/peptides/protein_peptide/new_models/esm_based/", config['model_id'])
os.makedirs(root_dir, exist_ok=True)

with open(os.path.join(root_dir, "config.pkl"), "wb") as f:
    pickle.dump(config, f)

if args.early_stopping:
    callbacks = [EarlyStopping(monitor="peptiderive_val_loss", mode="min", verbose=True, patience=args.early_stopping_patience, min_delta=1e-1)]
else:
    callbacks = []
callbacks.append(ModelCheckpoint(monitor="peptiderive_val_loss", save_top_k=args.save_top_k, dirpath=os.path.join(root_dir, "top_models")))



wandb_logger = WandbLogger(project="hotspotter", name=args.model_id, log_model = False)

trainer = pl.Trainer(
                    logger=wandb_logger,
                    max_epochs=args.max_epochs,
                    precision=args.precision,
                    gpus=args.n_gpus,
                    enable_checkpointing=True,
                    callbacks=callbacks,
                    default_root_dir=root_dir,
                    detect_anomaly=args.anomaly_detection,
                    log_every_n_steps=1,
                    val_check_interval=10000,
                    gradient_clip_val=0.1,
                    accumulate_grad_batches = args.grad_accum)

model = SALTnPEPPR(config)

wandb_logger.log_hyperparams(config)
wandb_logger.watch

wandb_logger.log_text(key="hyperparameters", columns=["key", "value"], data=[[str(k), str(v)] for k, v in config.items()])

trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_checkpoint)