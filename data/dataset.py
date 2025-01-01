import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .prepare_validation import prepare_validation_data

def get_transforms():
    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transformation, val_transformation

def get_dataloaders(config):
    # Prepare validation data if needed
    if config.prepare_validation:
        prepare_validation_data(config.val_folder, config.val_labels_file)
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=config.train_folder,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=config.val_folder,
        transform=val_transform
    )
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 