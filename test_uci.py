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
from torchvision.transforms import ToPILImage, Resize


torch.set_float32_matmul_precision('medium')


prediction_head = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 6)
)
model_folder = 'barlowtwins_training'
model = BarlowTwins.load_from_checkpoint(f'{model_folder}/model.ckpt')
classifier = SSLClassifier(backbone=model.backbone, prediction_head=prediction_head, freeze_backbone=True)


np.random.seed(42)

train_data, train_y, validation_data, validation_y, test_data, test_y = read_files()

train_x = []
for i in range(train_data.shape[0]):
    signal = train_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    image = torch.tensor(image)
    image = ToPILImage()(image)
    image = Resize((224, 224))(image)
    train_x.append(image)
# train_x = torch.tensor(np.array(train_x))

val_x = []
for i in range(validation_data.shape[0]):
    signal = validation_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    image = torch.tensor(image)
    image = ToPILImage()(image)
    image = Resize((224, 224))(image)
    val_x.append(image)
# val_x = torch.tensor(np.array(val_x))

test_x = []
for i in range(test_data.shape[0]):
    signal = test_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    image = torch.tensor(image)
    image = ToPILImage()(image)
    image = Resize((224, 224))(image)
    test_x.append(image)
# test_x = torch.tensor(np.array(test_x))

train_dataset = UCIHARDataset(train_x, train_y, transform=ResizeTransform(), output_num=1)
val_dataset = UCIHARDataset(val_x, validation_y, transform=ResizeTransform(), output_num=1)
test_dataset = UCIHARDataset(test_x, test_y, transform=ResizeTransform(), output_num=1)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


early_stopping = EarlyStopping('val_loss', patience=100, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # 'min' mode means the checkpoint will be saved when the monitored quantity decreases
    save_top_k=1,        # Save the best model
    dirpath=model_folder+'TEST',  # Directory to save the checkpoints
    filename='model',  # Filename format
)
trainer = Trainer(limit_train_batches=1.0, max_epochs=10, callbacks=[early_stopping, checkpoint_callback], accelerator="gpu", devices=[0])
trainer.fit(model=classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)