import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, masks_dir, image_list_file, use_covid=True, mask_dir = None):
        mapping = { 'COVID': 0,
                    'Non-COVID': 1,
                    'Normal': 2}
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        image_names = []
        labels = []
        mask_names = []
        Segs = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = mapping[items[1]]
                if label == 2 and use_covid is False:
                    continue
                if mask_dir is not None:
                    mask_name = os.path.join(
                        mask_dir, os.path.splitext(image_name)[0] + '_xslor.png')
                    mask_names.append(mask_name)
                image_name = os.path.join(data_dir, image_name) ###
                image_names.append(image_name)
                mask = os.path.join(masks_dir, image_name)  ###
                Segs.append(mask)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.mask_names = mask_names
        self.Segs = Segs
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        masks = self.Segs[index]
        masks = Image.open(masks).convert('RGB')
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new('RGB', image.size), mask)
        if self.transform is not None:
            image = self.transform(image)

        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(masks.size)
            masks = Image.composite(masks, Image.new('RGB', masks.size), mask)
        if self.transform is not None:
            masks = self.transform(masks)
        return image_name, image, masks, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)







