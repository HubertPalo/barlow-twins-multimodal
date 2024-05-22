from data_helper import timeserie2image, read_files
import numpy as np
import pandas as pd
from dataset import UCIHARDataset
from transform import Transform
import torch
from barlowtwins import BarlowTwins
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
import datetime
from torchvision.transforms import ToPILImage, Resize


torch.set_float32_matmul_precision('medium')

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
    # image = ToTensor()(image)
    train_x.append(image)
# train_x = torch.tensor(np.array(train_x))
# train_x = torch.stack(train_x)

validation_x = []
for i in range(validation_data.shape[0]):
    signal = validation_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    image = torch.tensor(image)
    image = ToPILImage()(image)
    image = Resize((224, 224))(image)
    # image = ToTensor()(image)
    validation_x.append(image)
# validation_x = torch.tensor(np.array(validation_x))
# validation_x = torch.stack(validation_x)

test_x = []
for i in range(test_data.shape[0]):
    signal = test_data.iloc[i,:].values.reshape(9, -1)
    image = timeserie2image(signal)
    image = np.array([image, image, image])
    image = torch.tensor(image)
    image = ToPILImage()(image)
    image = Resize((224, 224))(image)
    # image = ToTensor()(image)
    test_x.append(image)
# test_x = torch.tensor(np.array(test_x))
# test_x = torch.stack(test_x)

train_dataset = UCIHARDataset(train_x, train_y, transform=Transform())
val_dataset = UCIHARDataset(validation_x, validation_y, transform=Transform())
test_dataset = UCIHARDataset(test_x, test_y, transform=Transform())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

bt_model = BarlowTwins()
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_date = f'barlowtwins_training'
early_stopping = EarlyStopping('val_loss', patience=100, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # 'min' mode means the checkpoint will be saved when the monitored quantity decreases
    save_top_k=1,        # Save the best model
    dirpath=current_date,  # Directory to save the checkpoints
    filename='model',  # Filename format
)
trainer = Trainer(limit_train_batches=1.0, max_epochs=5, callbacks=[early_stopping, checkpoint_callback], accelerator="gpu", devices=[0])
trainer.fit(model=bt_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)