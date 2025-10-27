# coding=utf-8
"""
Base Neural Network Class - OPTIMIZED VERSION
Enhanced with better training strategies and optimization techniques
"""

import abc
import torch
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class EarlyStopping:
    """OPTIMIZED: Early stopping utility"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class NeuralNetwork(nn.Module):
    """
    OPTIMIZED Base Neural Network Class
    
    Improvements:
    - Better optimizer with learning rate scheduling
    - Mixed precision training support
    - Enhanced early stopping
    - Better evaluation metrics
    - Gradient accumulation support
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.best_f1 = 0
        self.patience = 0
        self.init_clip_max_norm = 1.0  # Set default gradient clipping
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # OPTIMIZED: Training state
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

    @abc.abstractmethod
    def forward(self):
        pass

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):
        """
        OPTIMIZED training loop with:
        - Learning rate scheduling
        - Mixed precision training
        - Better early stopping
        - Gradient accumulation
        """
        if torch.cuda.is_available():
            self.cuda()
        
        batch_size = self.config['batch_size']
        
        # OPTIMIZED: Better optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 1e-3),
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # OPTIMIZED: Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True,
            min_lr=1e-6
        )
        
        # OPTIMIZED: Cosine annealing with warm restarts (alternative)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=5,
        #     T_mult=2,
        #     eta_min=1e-6
        # )

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=4, pin_memory=True)

        # OPTIMIZED: Label smoothing for better generalization
        if self.config.get('label_smoothing', 0) > 0:
            loss_func = LabelSmoothingCrossEntropy(
                smoothing=self.config['label_smoothing']
            )
        else:
            loss_func = nn.CrossEntropyLoss()
        
        # OPTIMIZED: Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 5),
            verbose=True
        )
        
        # OPTIMIZED: Mixed precision training
        use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        if use_amp:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        
        # OPTIMIZED: Gradient accumulation
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for epoch in range(self.config['epochs']):
            print("\n" + "="*60)
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print("="*60)
            
            self.train()
            avg_loss = 0
            avg_acc = 0
            num_batches = 0
            
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
            
            for i, data in enumerate(pbar):
                batch_x_tid, batch_x_text, batch_y = (
                    item.to(self.device, non_blocking=True) for item in data
                )

                # OPTIMIZED: Mixed precision forward pass
                if use_amp:
                    with autocast():
                        logit = self.forward(batch_x_tid, batch_x_text)
                        loss = loss_func(logit, batch_y)
                        loss = loss / accumulation_steps
                else:
                    logit = self.forward(batch_x_tid, batch_x_text)
                    loss = loss_func(logit, batch_y)
                    loss = loss / accumulation_steps

                # OPTIMIZED: Mixed precision backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # OPTIMIZED: Gradient accumulation
                if (i + 1) % accumulation_steps == 0:
                    if self.init_clip_max_norm is not None:
                        if use_amp:
                            scaler.unscale_(self.optimizer)
                        utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
                    
                    if use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()

                # Calculate accuracy
                corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100.0 * corrects / len(batch_y)

                avg_loss += loss.item() * accumulation_steps
                avg_acc += accuracy.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'acc': f'{accuracy:.2f}%'
                })

            # Calculate epoch metrics
            epoch_loss = avg_loss / num_batches
            epoch_acc = avg_acc / num_batches
            
            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['train_acc'].append(epoch_acc)
            
            print(f"\nTraining - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.2f}%")

            # OPTIMIZED: Validation
            val_loss, val_acc, val_f1 = self.evaluate(
                X_dev_tid, X_dev, y_dev, loss_func
            )
            
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            print(f"Validation - Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")

            # OPTIMIZED: Learning rate scheduling
            self.scheduler.step(val_f1)  # Step based on F1 score
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")

            # OPTIMIZED: Early stopping check
            early_stopping(val_loss, self, self.config['save_path'])
            
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # Load best model
        print("\nLoading best model...")
        self.load_state_dict(torch.load(self.config['save_path']))
        
        # Final evaluation
        print("\nFinal Evaluation on Dev Set:")
        self.evaluate_detailed(X_dev_tid, X_dev, y_dev)

    def evaluate(self, X_dev_tid, X_dev, y_dev, loss_func=None):
        """
        OPTIMIZED evaluation with loss calculation
        """
        self.eval()
        
        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        
        X_dev_tid = torch.LongTensor(X_dev_tid).to(self.device)
        X_dev = torch.LongTensor(X_dev).to(self.device)
        y_dev_tensor = torch.LongTensor(y_dev).to(self.device)

        dataset = TensorDataset(X_dev_tid, X_dev, y_dev_tensor)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x_tid, batch_x_text, batch_y in dataloader:
                logits = self.forward(batch_x_tid, batch_x_text)
                loss = loss_func(logits, batch_y)
                total_loss += loss.item()
                
                predicted = torch.max(logits, dim=1)[1]
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1

    def evaluate_detailed(self, X_dev_tid, X_dev, y_dev):
        """
        OPTIMIZED: Detailed evaluation with classification report
        """
        y_pred = self.predict(X_dev_tid, X_dev)
        acc = accuracy_score(y_dev, y_pred)
        f1 = f1_score(y_dev, y_pred, average='macro')

        if acc > self.best_acc or (acc == self.best_acc and f1 > self.best_f1):
            self.best_acc = acc
            self.best_f1 = f1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print("\n" + "="*60)
            print("NEW BEST MODEL!")
            print("="*60)
            print(classification_report(y_dev, y_pred, 
                                       target_names=self.config['target_names'], 
                                       digits=5))
            print(f"Accuracy: {acc:.5f}")
            print(f"F1-Score: {f1:.5f}")
            print("="*60)
        else:
            self.patience += 1

    def predict(self, X_test_tid, X_test):
        """
        OPTIMIZED prediction with batch processing
        """
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).to(self.device)
        X_test = torch.LongTensor(X_test).to(self.device)

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=False)

        with torch.no_grad():
            for batch_x_tid, batch_x_text in dataloader:
                logits = self.forward(batch_x_tid, batch_x_text)
                predicted = torch.max(logits, dim=1)[1]
                y_pred.extend(predicted.cpu().numpy().tolist())
        
        return y_pred

    def save_training_history(self, filepath):
        """OPTIMIZED: Save training history"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=4)


class LabelSmoothingCrossEntropy(nn.Module):
    """OPTIMIZED: Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss