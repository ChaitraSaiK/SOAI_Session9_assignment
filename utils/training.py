import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")
            writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + batch)

def evaluate(val_loader, model, loss_fn, epoch, writer, calc_acc5=True):
    model.eval()
    size = len(val_loader.dataset)
    num_batches = len(val_loader)
    test_loss, correct1, correct5 = 0, 0, 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # Top-1 accuracy
            correct1 += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            # Top-5 accuracy
            if calc_acc5:
                _, top5 = pred.topk(5, 1, True, True)
                correct5 += torch.eq(top5, y.view(-1, 1).expand_as(top5)).sum().item()

    test_loss /= num_batches
    correct1 /= size
    writer.add_scalar('Loss/val', test_loss, epoch)
    writer.add_scalar('Accuracy/val', 100*correct1, epoch)
    
    print(f"Test Error: \n Accuracy: {(100*correct1):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    if calc_acc5:
        correct5 /= size
        writer.add_scalar('Accuracy5/val', 100*correct5, epoch)
        print(f"Test Error: \n Accuracy-5: {(100*correct5):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def save_checkpoint(model, optimizer, lr_scheduler, epoch, config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "params": config
    }
    
    checkpoint_dir = os.path.join("checkpoints", config.name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_{epoch}.pth"))
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint.pth"))

def load_checkpoint(model, optimizer, lr_scheduler, config):
    checkpoint_path = os.path.join("checkpoints", config.name, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        return 0
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    assert config == checkpoint["params"]
    
    return checkpoint["epoch"] + 1 