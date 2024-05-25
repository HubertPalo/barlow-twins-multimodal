from data_helper import timeserie2image, read_files, preprocess_data
import numpy as np
import pandas as pd
from dataset import UCIHARDataset
from transform import Transform, ResizeTransform
import torch
from barlowtwins import BarlowTwins
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
import datetime
import os
from classifier import SSLClassifier
from torch import nn, set_float32_matmul_precision
from torchvision.transforms import ToPILImage, Resize

set_float32_matmul_precision('medium')
np.random.seed(42)

def main(args):
    prediction_head = nn.Sequential(
        # From (512,1,1) to (512)
        # nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 6)
    )
    
    model = BarlowTwins.load_from_checkpoint(f'{args.dirpath}/BT-PRETEXT-{args.filename}.ckpt')
    classifier = SSLClassifier.load_from_checkpoint(
        f'{args.dirpath}/BT-DOWNSTREAM-{args.filename}.ckpt',
        # backbone=model.backbone,
        # prediction_head=prediction_head, freeze_backbone=True
        )
    classifier = SSLClassifier(backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)

    model = BarlowTwins.load_from_checkpoint(f'{model_folder}/model.ckpt')
    classifier = SSLClassifier(backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)


np.random.seed(42)

train_data, train_y, validation_data, validation_y, test_data, test_y = read_files()
test_x = preprocess_data(test_data)

test_dataset = UCIHARDataset(test_x, test_y, transform=ResizeTransform(), output_num=1)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

classifier = SSLClassifier.load_from_checkpoint(f'{model_folder}TEST/model.ckpt', backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)
# classifier = SSLClassifier(backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)

trainer = Trainer(accelerator="gpu", devices=[0])
# trainer.fit(model=classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

predictions = trainer.predict(model=classifier, dataloaders=test_dataloader, return_predictions=True)
y_orig = []
y_pred = []
for pred_list in predictions:
    y_orig.extend(pred_list[1].tolist())
    y_pred.extend(pred_list[0].tolist())

from torchmetrics.classification import MulticlassConfusionMatrix
target = torch.tensor(y_orig)
preds = torch.tensor(y_pred)
metric = MulticlassConfusionMatrix(num_classes=6)
print(metric(preds, target))


# print(torch.stack(predictions).shape)
# y_orig = [val[1] for val in predictions]
