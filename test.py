import torch
from model import SALTnPEPPR
from data import ProteinPeptideDataModule
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

root_dir = os.path.join("logging_path")
os.makedirs(root_dir, exist_ok=True)

wandb_logger = WandbLogger(project="test-hotspotter", log_model = False)

trainer = pl.Trainer(
                    logger=wandb_logger,
                    max_epochs=1000,
                    precision=16,
                    gpus=[0],
                    enable_checkpointing=False,
                    default_root_dir=root_dir,
                    log_every_n_steps=1,
                    check_val_every_n_epoch=1,
                    gradient_clip_val=0.1)

model = SALTnPEPPR.load_from_checkpoint('PATH_HERE:modelweights')

model.eval()

datamodule = ProteinPeptideDataModule()

trainer.test(model, datamodule=datamodule)
