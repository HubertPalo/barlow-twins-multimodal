from torch import optim, nn
import lightning as L


class SSLClassifier(L.LightningModule):
    def __init__(
            self,
            backbone,
            prediction_head,
            freeze_backbone=True
            ):
        super().__init__()
        self.backbone = backbone
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        self.prediction_head = prediction_head
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        # print("FORWARD1", x.shape)
        x = self.backbone(x).flatten(start_dim=1)
        # print("FORWARD2", x.shape)
        x = self.prediction_head(x)
        # print("FORWARD3", x.shape)
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
        return self(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer