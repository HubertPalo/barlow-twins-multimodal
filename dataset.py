from torch.utils.data import Dataset


class UCIHARDataset(Dataset):
    def __init__(self, x, y, transform, output_num=2) -> None:
        super().__init__()
        self.x = x.to('cuda')
        self.y = y.to('cuda')
        self.transform = transform
        self.output_num = output_num
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        sample = self.x[index]
        # print(sample.shape)
        # print(self.y.shape)
        label = self.y.iloc[index,0]
        sample1, sample2 = self.transform(sample)
        return sample1, sample2, label
