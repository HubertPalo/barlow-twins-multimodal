from data_helper import read_files, preprocess_data
import numpy as np
from dataset import UCIHARDataset
from transform import ResizeTransform
from torch import set_float32_matmul_precision
from barlowtwins import BarlowTwins
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
import os
from classifier import SSLClassifier
from torch.utils.data import DataLoader
import argparse

set_float32_matmul_precision('medium')
np.random.seed(42)

def main(args):
    os.makedirs(f'{args.exp_folder}/{args.dirpath}', exist_ok=True)
    model = BarlowTwins.load_from_checkpoint(f'{args.exp_folder}/{args.dirpath}/BT-PRETEXT-{args.filename}.ckpt')
    classifier = SSLClassifier(backbone=model.backbone, freeze_backbone=args.freeze_backbone)

    train_data, train_y, validation_data, validation_y, _, _ = read_files()
    train_x = preprocess_data(train_data)
    validation_x = preprocess_data(validation_data)

    train_dataset = UCIHARDataset(train_x, train_y, transform=ResizeTransform(), output_num=1)
    val_dataset = UCIHARDataset(validation_x, validation_y, transform=ResizeTransform(), output_num=1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    prefix = 'FROZEN' if args.freeze_backbone else 'FINETUNING'

    early_stopping = EarlyStopping('val_loss', patience=args.patience, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=f'{args.exp_folder}/{args.dirpath}', filename=f'BT-DOWNSTREAM-{prefix}-{args.filename}')
    trainer = Trainer(limit_train_batches=1.0, max_epochs=args.max_epochs, callbacks=[early_stopping, checkpoint_callback], accelerator="gpu", devices=[0])
    trainer.fit(model=classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Barlow Twins model')
    parser.add_argument('--exp-folder', type=str, default='experiments', help='Experiment folder', required=False)
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size', required=False)
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping', required=False)
    parser.add_argument('--max-epochs', type=int, default=100000, help='Maximum number of epochs', required=False)
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloader', required=False)
    parser.add_argument('--filename', type=str, default='model', help='Checkpoint file name', required=False)
    parser.add_argument('--freeze-backbone', type=bool, default=True, help='Freeze the backbone', required=False)
    parser.add_argument('--dirpath', type=str, help='Directory to save the checkpoints', required=True)
    args = parser.parse_args()
    main(args)