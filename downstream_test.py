from data_helper import timeserie2image, read_files, preprocess_data
import numpy as np
# import pandas as pd
from dataset import UCIHARDataset
from transform import Transform, ResizeTransform
import torch
from barlowtwins import BarlowTwins
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
import argparse
from torch.utils.data import DataLoader
from classifier import SSLClassifier
from torch import nn, set_float32_matmul_precision
# from torchvision.transforms import ToPILImage, Resize
# from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


set_float32_matmul_precision('medium')
np.random.seed(42)

def main(args):
    prefix = 'FROZEN' if args.freeze_backbone else 'FINETUNING'
    # model = BarlowTwins.load_from_checkpoint(f'{args.exp_folder}/{args.dirpath}/BT-PRETEXT-{args.filename}.ckpt')
    classifier = SSLClassifier.load_from_checkpoint(
        f'{args.exp_folder}/{args.dirpath}/BT-DOWNSTREAM-{prefix}-{args.filename}.ckpt',
        freeze_backbone=True
        )
    _, _, _, _, test_data, test_y = read_files()
    test_x = preprocess_data(test_data)
    test_dataset = UCIHARDataset(test_x, test_y, transform=ResizeTransform(), output_num=1)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    trainer = Trainer(accelerator="gpu", devices=[0])
    predictions = trainer.predict(model=classifier, dataloaders=test_dataloader, return_predictions=True)
    y_orig = []
    y_pred = []
    for pred_list in predictions:
        for i in range(len(pred_list[1])):
            print('ORIG', pred_list[1][i], 'PRED', pred_list[0][i])
        y_orig.extend(pred_list[1].tolist())
        y_pred.extend(pred_list[0].tolist())

    target = torch.tensor(y_orig)
    preds = torch.tensor(y_pred)
    
    print('Accuracy:', accuracy_score(target, preds))
    print('F1:', f1_score(target, preds, average='weighted'))
    print('Precision:', precision_score(target, preds, average='weighted'))
    print('Recall:', recall_score(target, preds, average='weighted'))
    print('Confusion Matrix:\n', confusion_matrix(target, preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Barlow Twins model')
    parser.add_argument('--exp-folder', type=str, default='experiments', help='Experiment folder', required=False)
    # parser.add_argument('--batch-size', type=int, default=256, help='Batch size', required=False)
    # parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping', required=False)
    # parser.add_argument('--max-epochs', type=int, default=100000, help='Maximum number of epochs', required=False)
    # parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloader', required=False)
    parser.add_argument('--filename', type=str, default='model', help='Checkpoint file name', required=False)
    parser.add_argument('--freeze-backbone', type=bool, default=True, help='Freeze the backbone', required=False)
    parser.add_argument('--dirpath', type=str, help='Directory to save the checkpoints', required=True)
    args = parser.parse_args()
    main(args)