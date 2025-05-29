import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import time 

sys.stdout.reconfigure(line_buffering=True)

def setup_ddp():
    # Get SLURM environment variables with defaults
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')

    # Explicitly set environment variables for PyTorch
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize the process group
    dist.init_process_group(backend='gloo', init_method='env://')
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} initialized")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def get_dataloaders(data_dir, rank, world_size, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def train_one_epoch(model, dataloader, criterion, optimizer, device, rank):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if rank == 0 and batch_idx % 50 == 0:
            print(f"[Train] Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, rank):
    print(f"Rank {rank} is evaluating...", flush=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100. * correct / total
    if rank == 0:
        print(f"[Validation] Accuracy: {accuracy:.2f}%")
    
    return accuracy

def save_model(model, optimizer, epoch, accuracy, save_dir, filename):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'model_name': 'resnet50',
        'args': {
            'num_classes': model.module.fc.out_features,
        }
    }
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

def load_model(args, model, optimizer, load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint.get('accuracy', 0.0)
    
    if 'args' in checkpoint:
        saved_num_classes = checkpoint['args'].get('num_classes', args.num_classes)
        current_num_classes = model.module.fc.out_features if hasattr(model, 'module') else model.fc.out_features
        
        if saved_num_classes != current_num_classes:
            print(f"Warning: Model was saved with {saved_num_classes} classes, "
                  f"but current model has {current_num_classes} classes")
    else:
        print("Note: Loading checkpoint from older version (no model config saved)")
    
    print(f"Model loaded from {load_path}")
    print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")
    
    return epoch, accuracy

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def train(args):
    rank, local_rank, world_size = setup_ddp()
    device = get_device()

    train_loader, val_loader = get_dataloaders(args.data_dir, rank, world_size, args)

    model = models.resnet50(num_classes=args.num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])

    print("Model initialized.", flush=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), 
                        lr=args.learning_rate,
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)

    best_accuracy = 0.0
    start_epoch = 0

    if args.resume:
        if rank == 0:
            print(f"Resuming training from {args.resume}")
        start_epoch, best_accuracy = load_model(args, model, optimizer, args.resume)
        start_epoch += 1
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, best accuracy: {best_accuracy:.2f}%")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, rank)
        
        accuracy = evaluate(model, val_loader, device, rank)

        # Print epoch summary
        if rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"[Epoch {epoch}] Summary - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
        
        
        if rank == 0 and accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, optimizer, epoch, accuracy, args.save_dir, f"best_model_{epoch}_epochs.pth")
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        if rank == 0 and (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, best_accuracy, args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    if rank == 0:
        save_model(model, optimizer, args.epochs-1, best_accuracy, args.save_dir, "final_model.pth")
        print(f"Final model saved to {args.save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./saved_resnet_model', 
                        help='Directory to save the trained model')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
