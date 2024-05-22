from torch import optim, nn
import lightning as L
from torch.nn import Conv1d, AdaptiveAvgPool1d, Dropout, ReLU, Linear, BatchNorm1d, Flatten
import torch
import numpy as np
from scipy import signal
import torchvision
from  lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss


class BarlowTwins(L.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = BarlowTwinsLoss(gather_distributed=True)        

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        return x
    

    def training_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        loss = self.criterion(z1, z2)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1, x2 = batch[0], batch[1]
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        loss = self.criterion(z1, z2)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # print("PREDICT STEP", batch[0].shape)
        return self.backbone(batch[0]).flatten(start_dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer