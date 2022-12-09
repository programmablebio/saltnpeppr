import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import esm
import numpy as np
from torchmetrics.functional import auroc
import torchmetrics

import random
import pickle
from torch.nn import LogSoftmax

# helper functions
def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp,device=tmp.device)
    ranks[tmp] = torch.arange(x.size(0),device=tmp.device)
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


class MLP(pl.LightningModule):
    def __init__(self, input_dim, embedding_dim, num_layers, dropout=0, output_relu=False, bias=False):
        super().__init__()

        layers_list = [nn.Linear(input_dim, embedding_dim, bias=bias)]
        for i in range(num_layers - 1):
            # relu for previous layer gets added first
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(p=dropout))
            layers_list.append(nn.Linear(embedding_dim, embedding_dim, bias=bias))

        if output_relu:
            layers_list.append(nn.ReLU())

        self.layers = nn.Sequential(*layers_list)

    def forward(self, input_embedding, padding_mask=None):
        embedding = input_embedding

        embedding = self.layers(embedding)
        return embedding

class SALTnPEPPR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigmoid = nn.Sigmoid()
        self.val_AUC = torchmetrics.AUROC(task="binary")

        self.esm_transformer, _ = esm.pretrained.esm2_t33_650M_UR50D() # load pretrained ESM2

        counter = 0
        for name, param in self.esm_transformer.named_parameters(): # unfreeze ESM2
            counter += 1
            if counter < 466: # last 3 transformer layers unfreezed for finetuning
                param.requires_grad = False

        self.reader_MLP = MLP(
            input_dim=1280, # Embedding dimension
            embedding_dim=self.config['mlp_dim'],
            num_layers=self.config['mlp_layers'],
            dropout=self.config['dropout'],
            output_relu=False,
            bias = True
        )
        self.infer_mlp = MLP(
            input_dim=self.config['mlp_dim'],
            embedding_dim=2,
            num_layers=1,
            dropout=self.config['dropout'],
            output_relu=False,
            bias = True
        )
        self.save_hyperparameters()

        self.loss = torch.nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, protein_input, protein_padding_mask=None, batch_size = None, return_loss = True):

        outputs = self.esm_transformer(protein_input,repr_layers=[33],return_contacts=False)
        representations = outputs['representations'][33] # last layer

        mlp_out = self.reader_MLP(representations) # pass each token to MLP
        outs = self.infer_mlp(mlp_out)

        return outs[0][1:-1].squeeze(dim=-1)

    def training_step(self,batch,batch_idx):
        scores = self(
            batch['peptide_source'],
        )

        bin_scores = batch['peptide_scores']

        loss = self.loss(scores.unsqueeze(dim=-1), bin_scores) # loss calculated as batch across amino acid positions

        self.log("train_loss", loss, sync_dist=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        scores = self(
            batch['peptide_source'],
        )

        true_scores = batch['peptide_scores_value']
        bin_scores = batch['peptide_scores']

        squeezed_true = scores.squeeze(dim=-1)
        loss = self.loss(scores.unsqueeze(dim=-1), bin_scores)
        self.log("peptiderive_val_loss", loss, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
        self.val_AUC.update(scores.cpu().detach(), batch['peptide_scores'].squeeze(dim=-1).cpu().detach())
        self.log("validation_AUROC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        scores = self(
            batch['peptide_source'],
        )
        true_scores = batch['peptide_scores_value']
        bin_scores = batch['peptide_scores']

        self.val_AUC.update(scores.cpu().detach(), batch['peptide_scores'].squeeze(dim=-1).cpu().detach())
        self.log("test_AUROC", self.val_AUC, on_step=False, on_epoch=True, prog_bar=True)

        squeezed_score = scores.squeeze(dim=-1)
        squeezed_true = true_scores.squeeze(dim=-1)

        loss = self.loss(scores.unsqueeze(dim=-1), bin_scores)

        if scores.size(0) > 64:
            topbinder_scores = scores[:,1]
            spearman = spearman_correlation(topbinder_scores, -1*squeezed_true)
            self.log("entry_peptiderive_test_spearman", spearman, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            top_1_ind = torch.topk(topbinder_scores, 1, largest=True)[1]
            top_1_true = squeezed_true[top_1_ind]
            self.log("entry_test_top_1_%<-15", top_1_true.le(-0.15).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_1_%<-10", top_1_true.le(-0.1).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_1_%<-07", top_1_true.le(-0.07).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_1_%<-03", top_1_true.le(-0.03).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_1_%<-0", top_1_true.lt(0.00).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)


            top_6_ind = torch.topk(topbinder_scores, 6, largest=True)[1]
            top_6_true = squeezed_true[top_6_ind]
            self.log("entry_test_top_6_any<-15", top_6_true.le(-0.15).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_any<-10", top_6_true.le(-0.1).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_any<-07", top_6_true.le(-0.07).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_any<-03", top_6_true.le(-0.03).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_any<-0", top_6_true.lt(0.00).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            top_6_ind = torch.topk(topbinder_scores, 6, largest=True)[1]
            top_6_true = squeezed_true[top_6_ind]
            self.log("entry_test_top_6_%<-15", top_6_true.le(-0.15).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_%<-10", top_6_true.le(-0.1).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_%<-07", top_6_true.le(-0.07).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_%<-03", top_6_true.le(-0.03).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_6_%<-0", top_6_true.lt(0.00).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            random_integer = random.randint(0,scores.size(0)-1)
            random_true = squeezed_true[random_integer]
            self.log("entry_test_random1_%<-15", random_true.le(-0.15).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random1_%<-10", random_true.le(-0.1).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random1_%<-07", random_true.le(-0.07).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random1_%<-03", random_true.le(-0.03).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random1_%<-0", random_true.lt(0.0).any(), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            rand_ind = list(range(0, scores.size(0)))
            random.shuffle(rand_ind)
            rand_ind = rand_ind[0:7]
            random_true = squeezed_true[rand_ind]
            self.log("entry_test_random6_%<-15", random_true.le(-0.15).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random6_%<-10", random_true.le(-0.1).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random6_%<-07", random_true.le(-0.07).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random6_%<-03", random_true.le(-0.03).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_random6_%<-0", random_true.lt(0.0).sum()/6.0, sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            rand_ind = rand_ind[0:3]

            self.log("entry_random_top_3_worst_ind", torch.min(torch.FloatTensor(rand_ind))/scores.size(0), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_random_top_3_avg_ind", torch.mean(torch.FloatTensor(rand_ind))/scores.size(0), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

            top_3_ind = torch.topk(topbinder_scores, 3, largest=True)[1]

            top_3_true_ind = torch.isin(squeezed_true.argsort(descending=True), top_3_ind)
            top_3_true_ind = top_3_true_ind.nonzero().double()

            self.log("entry_test_top_3_worst_ind", torch.min(top_3_true_ind)/scores.size(0), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)
            self.log("entry_test_top_3_avg_ind", torch.mean(top_3_true_ind)/scores.size(0), sync_dist=True, prog_bar=True, batch_size=scores.size(0), add_dataloader_idx=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        return optimizer
