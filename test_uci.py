from data_helper import timeserie2image, read_files, download_uci_dataset, extract_uci_dataset
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
from torch import nn


torch.set_float32_matmul_precision('medium')


prediction_head = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 6)
)
current_date = '2021-09-29_15-00-00'
model = BarlowTwins.load_from_checkpoint(f'{current_date}/model.ckpt')
classifier = SSLClassifier(backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)


if not os.path.exists('data/UCIHAR/dataset/UCI HAR Dataset'):
    download_uci_dataset()
    extract_uci_dataset()


np.random.seed(42)

train_data, train_y, validation_data, validation_y, test_data, test_y = read_files()

train_x = []
for i in range(train_data.shape[0]):
    signal = train_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    train_x.append(image)
train_x = torch.tensor(np.array(train_x))

val_x = []
for i in range(validation_data.shape[0]):
    signal = validation_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    val_x.append(image)
val_x = torch.tensor(np.array(val_x))

test_x = []
for i in range(test_data.shape[0]):
    signal = test_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    test_x.append(image)
test_x = torch.tensor(np.array(test_x))

train_dataset = UCIHARDataset(train_x, train_y, transform=ResizeTransform())
val_dataset = UCIHARDataset(val_x, validation_y, transform=ResizeTransform())
test_dataset = UCIHARDataset(test_x, test_y, transform=ResizeTransform())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


early_stopping = EarlyStopping('val_loss', patience=100, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # 'min' mode means the checkpoint will be saved when the monitored quantity decreases
    save_top_k=1,        # Save the best model
    dirpath=current_date+'TEST',  # Directory to save the checkpoints
    filename='model',  # Filename format
)
trainer = Trainer(limit_train_batches=1.0, max_epochs=100000, callbacks=[early_stopping, checkpoint_callback], accelerator="gpu", devices=[0])
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)