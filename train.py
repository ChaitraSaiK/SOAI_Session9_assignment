import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from data.dataset import get_dataloaders
from models.resnet import create_model
from utils.training import train_one_epoch, evaluate, save_checkpoint, load_checkpoint

def main():
    # Initialize configuration
    config = Config()
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create model, loss function, optimizer and scheduler
    model = create_model(config.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter('runs/' + config.name)
    
    # Load checkpoint if resuming training
    start_epoch = 0
    if config.resume_training:
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, config)
    
    # Initial evaluation
    evaluate(val_loader, model, loss_fn, epoch=0, writer=writer, calc_acc5=True)
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print(f"******** EPOCH: {epoch}")
        
        train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, writer)
        save_checkpoint(model, optimizer, lr_scheduler, epoch, config)
        
        lr_scheduler.step()
        evaluate(val_loader, model, loss_fn, epoch + 1, writer, calc_acc5=True)

if __name__ == "__main__":
    main() 