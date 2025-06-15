import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.runner.utils import progress_bar
import time

from src.build.registry import MODULES, OPTIMIZERS, DATASETS, EVALUATORS
from src.build.registry import build_module


class Runner:
    def __init__(self, model: nn.Module | dict,
                 dataloader_cfg: dict,
                 dataset: nn.Module | dict,
                 optim: nn.Module | dict,
                 work_dir: str = None,
                 device: str = 'cpu', 
                 seed: int = None,
                 patience: int = 15,
                 abort_condition: float = 0.05,
                 direction: str = 'min',
                 metric: str = 'val_loss',
                 save_best: bool = True,
                 **kwargs):
        """
        Initializes the Runner with model, data, and optimizer configurations.
        
        Args:
            model (nn.Module | dict): Model configuration or instance.
            dataloader_cfg (dict): Configuration for DataLoader.
            dataset (nn.Module | dict): Dataset configuration or instance.
            optim (nn.Module | dict): Optimizer configuration or instance.
            work_dir (str, optional): Directory to save model and logs. Defaults to None.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.model_cfg = model
        self.dataloader_cfg = dataloader_cfg
        self.dataset_cfg = dataset
        self.optim_cfg = optim
        
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"Using seed: {seed}")
        else:
            seed = torch.initial_seed() & 0xFFFFFFFF #Make the seed a 32-bit integer to work with numpy and torch
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"No seed provided, using initial seed: {seed}")

        self.device = torch.device(device)

        self.dataloader_cfg = dataloader_cfg
        self.model = build_module(self.model_cfg, MODULES).to(self.device)
        self.model.print_param_summary()
        self.criterion = self.model.criterion


        self.optim = build_module(optim, OPTIMIZERS, params=self.model.parameters())
        datasets = build_module(self.dataset_cfg, DATASETS)  # Build dataset from config
        self.train_data = datasets.train_data
        self.val_data = datasets.val_data
        self.test_data = datasets.test_data

        self.evaluator = build_module(dict(type='BaseEvaluator', dataset=self.val_data), EVALUATORS)
        self.batch_size = self.dataloader_cfg['batch_size']

        self.patience = patience
        self.abort_condition = abort_condition
        self.direction = direction
        self.metric = metric
        self.save_best = save_best
        self.best_model_path = None
       
        if work_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.save_dir = os.path.join(work_dir, f"run_{timestamp}")
            os.makedirs(self.save_dir)
            self.save_cfg()

        

        self.history = {'train_loss': [], 'val_loss': [], 'f1': [], 'mean_ap': [], 'ap_per_class': []}

        
    def save_cfg(self, filename='config.yaml'):
        """
        Saves the configuration of the model, optimizer, and dataset to a YAML file.
        Args:
            filename (str): Name of the file to save the configuration.
        """
        import yaml
        cfg = {
            'model': self.model.describe(),
            'dataloader_cfg': self.dataloader_cfg,
            'dataset': self.dataset_cfg,
            'optim': self.optim,
            'work_dir': self.save_dir,
            'device': str(self.device),
            'seed': torch.initial_seed(),
        }
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def save_model(self, filename='ckpt.pth'):
        """
        Saves the model state, optimizer state, and training history.
        Args:
            filename (str): Name of the file to save the model.
        """
        torch.save({'metrics': self.history}, os.path.join(self.save_dir, 'metrics.pth'))
        torch.save({'optmizer_state': self.optim.state_dict()}, os.path.join(self.save_dir, 'optimizer.pth'))
        self.model.save_checkpoint(path=os.path.join(self.save_dir, filename))
        print(f"Model saved to {os.path.join(self.save_dir, filename)}")
        return os.path.join(self.save_dir, filename)


    def run(self,
            mode: str = 'train',
            val_interval: int= 10,
            log_interval: int= 10,
            epochs: int = 100,
            start_epoch: int = 1):
        """
        Main entry point for running the model.
        Args:
            mode (str): 'train', 'validation', or 'test'.
            val_interval (int): Validation interval in epochs.
            log_interval (int): Logging interval in batches.
            epochs (int): Total number of epochs to train.
            start_epoch (int): Starting epoch for training.
        Returns:
            dict: History of training/validation metrics.
        """
        if mode == 'train':
            return self._train_loop(start_epoch, epochs, val_interval, log_interval)

        elif mode == 'validation':
            return self.evaluate(self.val_data, batch_size=1)

        elif mode == 'test':
            return self.predict(self.test_data, batch_size=self.batch_size, loss=False)

        else:
            raise ValueError("Mode must be 'train', 'validation', or 'test'.")
        
        

    def _train_loop(self, start, epochs, val_interval, log_interval):
        """Main training loop.
        Args:
            start (int): Starting epoch.
            epochs (int): Total number of epochs to train.
            val_interval (int): Validation interval in epochs.
            log_interval (int): Logging interval in batches.
        """
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)


        for epoch in range(start, epochs):
            train_loss = self._train_epoch(train_loader, epoch=epoch, total_epochs=epochs, log_interval=log_interval)
            print("") # for a new line after progress bar at the end of the training epoch
            self.history['train_loss'].append(train_loss)

            if epoch % val_interval == 0:
                evals = self.evaluate(self.val_data, epoch=epoch, batch_size=1)

                if self.save_dir and self.save_best:
                    if self.direction == 'min':
                        cond = evals[self.metric] < min(self.history[self.metric], default=float('inf'))	
                    elif self.direction == 'max':
                        cond = evals[self.metric] > max(self.history[self.metric], default=float('-inf'))
                    else:
                        raise ValueError("direction must be 'min' or 'max'")

                    if cond:
                        path = self.save_model(filename='best_model.pth')
                        self.best_model_path = path
                        print(f"Best model saved at epoch {epoch} with {self.metric} {evals[self.metric]:.4f}")

                # append eval results to history
                self.history['val_loss'].append(evals['val_loss'])
                self.history['f1'].append(evals['f1'])
                self.history['mean_ap'].append(evals['mean_ap'])
                self.history['ap_per_class'].append(evals['ap_per_class'])
            
            self.plot_metrics() # plot metrics after each epoch
            if self._check_abort(): break
            
        return self.history
    
    def _check_abort(self) -> bool:
        values = self.history.get(self.metric, [])
        # need at least patience+1 points (so you have a baseline + window)
        if len(values) < self.patience + 1:
            return False

        # take the last (patience+1) values
        window = values[-(self.patience + 1):]
        baseline, *rest = window  # baseline is the oldest in that window

        if self.direction == 'min':
            # did we ever drop by at least delta?
            if min(rest) <= baseline - self.abort_condition:
                return False  # there was a decent improvement
            print(f"Early stopping: no drop ≥{self.abort_condition} in last {self.patience} epochs of {self.metric}.")
            return True

        elif self.direction == 'max':
            # did we ever rise by at least delta?
            if max(rest) >= baseline + self.abort_condition:
                return False
            print(f"Early stopping: no rise ≥{self.abort_condition} in last {self.patience} epochs of {self.metric}.")
            return True

        else:
            raise ValueError("direction must be 'min' or 'max'")

    def _train_epoch(self, loader, epoch, total_epochs, log_interval=10):
        """
        Runs a single training epoch.
        Args:
            loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
            total_epochs (int): Total number of epochs.
            log_interval (int): Interval for logging progress.
        """
        self.model.train() # Set model to training mode
        total_loss = 0.
        total_batches = len(loader)

        for i, batch in enumerate(loader):
            imgs = batch['image'].to(self.device) # Convert images to device
            labels = batch['label'].to(self.device) # Convert labels to device

            self.optim.zero_grad() # Zero the gradients
            logits = self.model(imgs) # Forward pass
            loss = self.criterion(logits, labels) #! currently unsure whether i will implement a model-specific loss
            loss.backward() # Backward pass
            self.optim.step() # Update weights


            if i % log_interval == 0:
                progress_bar(
                    epoch=epoch, 
                    total_epochs=total_epochs,
                    iteration=i+1,
                    total_iterations=total_batches,
                    vars={
                        'loss': loss.item(),
                        'lr': self.optim.param_groups[0]['lr'],
                    },)                

            total_loss += loss.item()
        return total_loss / len(loader)
    
    def predict(self, dataset, batch_size=1, loss=True, log_interval=50):
        """
        Runs model inference on a dataset.
        Args:
            dataset (nn.Module | dict): Dataset configuration or instance.
            batch_size (int): Batch size for inference.
        Returns:
            np.ndarray: Predicted labels.
        """
        """
        Runs model inference on a DataLoader.
        Returns:
            y_true (np.ndarray): True labels shape (N,).
            y_pred (np.ndarray): Predicted labels shape (N,).
            y_scores (np.ndarray): Predicted probabilities shape (N, num_classes).
        """
        self.model.eval()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        total_iterations = len(loader)
        if loss:
            return_dict = {'y_pred': [], 'y_scores': [],'y_true': [] ,'val_loss': [], 'mean_loss': 0.0}
        else: 
            return_dict = {'y_pred': [], 'y_scores': []}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                imgs = batch['image'].to(self.device)
                
                logits = self.model(imgs)
                
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                return_dict['y_pred'].extend(preds.cpu().numpy())
                return_dict['y_scores'].extend(probs.cpu().numpy())
                if loss:
                    labels = batch['label'].to(self.device)
                    return_dict['y_true'].extend(labels.cpu().numpy())

                    val_loss = self.model.criterion(logits, labels)
                    return_dict['val_loss'].append(val_loss.item())

                if i % log_interval == 0 or i == total_iterations - 1:
                    mean_loss = float(sum(return_dict['val_loss']) / len(return_dict['val_loss'])) if loss else None
                    return_dict['mean_loss'] = mean_loss
                    vars = {
                        'val_loss': mean_loss,
                    } if loss else None
                    progress_bar(
                        iteration=i + 1,
                        total_iterations=total_iterations,
                        prefix="Evaluating",
                        postfix=f" {((i+1)/total_iterations)*100:.2f}%",
                        vars=vars if loss else None,
                        style="arrow"
                    )
                

        return return_dict

    def evaluate(self, dataset, epoch=None, batch_size=1, log_interval=50):
        prediction = self.predict(dataset, batch_size=batch_size, loss=True, log_interval=log_interval) # Run inference on the dataset

        y_true = np.array(prediction['y_true']) # True labels
        y_pred = np.array(prediction['y_pred']) # Predicted labels
        y_scores = np.array(prediction['y_scores']) # Predicted probabilities
        val_loss = float(prediction['mean_loss']) # Mean validation loss


        f1 = self.evaluator.compute_f1(y_true, y_pred) # compute F1 score
        mean_ap, ap_per_class = self.evaluator.compute_map(y_true, y_scores) # compute mean Average Precision (mAP)

        # Print confusion matrix if save_dir is set
        if self.save_dir and epoch is not None:
            fig = self.evaluator.plot_confusion_matrix(y_true, y_pred)
            filename = f'ep_{epoch}_cmatrix.png'
            fig.savefig(os.path.join(self.save_dir, filename))
            plt.close(fig)
        
        print(f" --> F1: {f1:.4f} | mAP: {mean_ap:.4f}") 

        return {
            'val_loss': val_loss,
            'f1': f1,
            'mean_ap': mean_ap,
            'ap_per_class': ap_per_class
        }
    
    def describe(self):
        """
        Returns a description of the model, optim, and dataset.
        """
        return {
            'model': self.model.describe(),
            'optim_cfg': self.optim_cfg,
            'dataloader_cfg': self.dataloader_cfg,
            'dataset_cfg': self.dataset_cfg,
            'device': str(self.device),
        }
    

    def plot_metrics(self):
        """
        Plots training and validation metrics and saves the plot.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot losses
        fig = plt.figure(figsize=(12, 6))
        plt.grid(True)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', color='blue')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'loss.png'))
            plt.close()
        else:
            plt.show()
        # Clear the current figure to avoid overlap
        plt.close(fig)

        
        # Plot F1 score
        fig = plt.figure(figsize=(12, 6))
        plt.grid(True)
        plt.plot(epochs, self.history['f1'], label='F1 Score', color='green')
        plt.title('F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'f1.png'))
        plt.close(fig)


        # Plot mAP
        fig = plt.figure(figsize=(12, 6))
        plt.grid(True)
        plt.plot(epochs, self.history['mean_ap'], label='mAP', color='red')
        plt.title('Mean Average Precision (mAP)')
        plt.xlabel('Epochs')
        plt.ylabel('mAP')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'map.png'))
        plt.close(fig)