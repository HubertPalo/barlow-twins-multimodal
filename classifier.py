from torch import optim, nn
import lightning as L
import numpy as np


class SSLClassifier(L.LightningModule):
    def __init__(
            self,
            backbone,
            # prediction_head,
            freeze_backbone=True
            ):
        super().__init__()
        self.backbone = backbone
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        self.prediction_head = nn.Sequential(
            nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.BatchNorm1d(64),
            nn.Linear(64, 6),
            nn.Softmax(dim=1)
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.prediction_head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log("test_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        x = self(x)
        x = np.argmax(x.cpu(), axis=1)
        return [x, batch[1]]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer