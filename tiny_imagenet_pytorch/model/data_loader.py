from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob

def fetch_label_map(unique_labels_path):
    with open(unique_labels_path, 'r') as f:
        label_map = {label.strip(): i for i, label in enumerate(f.readlines())}

    return label_map

class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, transform, label_map):
        try:
            self.filenames = glob(data_dir + '/*/*/*.JPEG')
            assert len(self.filenames) != 0
        except AssertionError:
            # it's the validation format
            self.filenames = glob(data_dir + '/*/*.JPEG')
            self.labels = [filename.split('/')[-2] for filename in self.filenames]
        else:
            self.labels = [filename.split('/')[-1].split('_')[0] for filename in self.filenames]

        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = image.convert('RGB')
        image = self.transform(image)
        label = self.label_map[self.labels[idx]]
        return image, label

def fetch_dataloader(mode, data_dir, label_map, batch_size=32, num_workers=4, pin_memory=True):

    if mode == 'train':
        train_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_dataloader = DataLoader(
            TinyImageNetDataset(data_dir, train_transformer, label_map),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_dataloader
    elif mode == 'eval':
        eval_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        eval_dataloader = DataLoader(
            TinyImageNetDataset(data_dir, eval_transformer, label_map),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return eval_dataloader

    return None

