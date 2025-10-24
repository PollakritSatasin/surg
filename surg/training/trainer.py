"""
Trainer class for instance segmentation models.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Dict


class Trainer:
    """
    Trainer for instance segmentation models.
    
    Args:
        model: The segmentation model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Configuration dictionary with training parameters
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Default configuration
        self.config = {
            'batch_size': 2,
            'num_workers': 4,
            'learning_rate': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'num_epochs': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_dir': './checkpoints',
            'log_dir': './logs',
            'save_interval': 1,
            'print_freq': 10,
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = torch.device(self.config['device'])
        self.model.model.to(self.device)
        
        # Setup optimizer
        params = [p for p in self.model.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.model.train()
        
        # Create data loader
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=self.train_dataset.collate_fn
        )
        
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
        
        for i, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            # Update metrics
            loss_value = losses.item()
            epoch_loss += loss_value
            
            # Logging
            if i % self.config['print_freq'] == 0:
                progress_bar.set_postfix({'loss': loss_value})
                
                # Log to TensorBoard
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
                self.writer.add_scalar('train/total_loss', loss_value, self.global_step)
            
            self.global_step += 1
        
        avg_loss = epoch_loss / len(data_loader)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataset is None:
            return {}
        
        self.model.model.eval()
        
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=self.val_dataset.collate_fn
        )
        
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc='Validation'):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_val_loss = val_loss / len(data_loader)
        return {'val_loss': avg_val_loss}
    
    def train(self):
        """
        Run the full training loop.
        """
        print(f"Starting training for {self.config['num_epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
            
            # Validate
            if self.val_dataset:
                val_metrics = self.validate()
                print(f'Epoch {epoch}: Val Loss = {val_metrics["val_loss"]:.4f}')
                self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
        
        print("Training complete!")
        self.writer.close()
    
    def save_checkpoint(self, epoch: int):
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint_path = os.path.join(
            self.config['save_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
