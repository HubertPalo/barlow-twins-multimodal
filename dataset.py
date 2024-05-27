from torch.utils.data import Dataset


class UCIHARDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None, output_num=2) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.output_num = output_num
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        sample = self.x[index]
        label = self.y.iloc[index,0]
        # Reduce the label by 1 to make it 0-based
        # print('LABEL BEFORE:', label)
        label = label - 1
        # print('LABEL AFTER:', label)
        if self.output_num == 1:
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, label
        sample1, sample2 = self.transform(sample)
        return sample1, sample2, label
