import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from src.runner.utils import progress_bar

from src.build.registry import EVALUATORS

@EVALUATORS.register_module()
class BaseEvaluator:
    def __init__(self, model, dataset, batch_size=1, device='cpu'):
        """
        Args:
            model (torch.nn.Module): Trained model for inference.
            device (str or torch.device): 'cpu' or 'cuda'.
            num_classes (int): Number of target classes.
            class_names (list[str], optional): Names of classes for plots.
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names if hasattr(dataset, 'class_names') else [str(i) for i in range(self.num_classes)]
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def predict(self, loss = False, log_interval=50):
        """
        Runs model inference on a DataLoader.
        Returns:
            y_true (np.ndarray): True labels shape (N,).
            y_pred (np.ndarray): Predicted labels shape (N,).
            y_scores (np.ndarray): Predicted probabilities shape (N, num_classes).
        """
        self.model.eval()
        y_true, y_pred, y_scores = [], [], []
        total_iterations = len(self.dataloader)
        if loss:
            hist = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                imgs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.model(imgs)
                
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                if loss:
                    val_loss = self.model.criterion(logits, labels)
                    hist.append(val_loss.item())
                    vars = {"val_loss": float(sum(hist) / len(hist))}

                if i % log_interval == 0 or i == total_iterations - 1:
                    progress_bar(
                        iteration=i + 1,
                        total_iterations=total_iterations,
                        prefix="Evaluating",
                        postfix=f" {((i+1)/total_iterations)*100:.2f}%",
                        vars=vars if loss else None,
                        style="arrow"
                    )

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

        return np.array(y_true), np.array(y_pred), np.array(y_scores)

    def compute_f1(self, y_true, y_pred, average='macro'):
        """Compute F1 score."""
        return f1_score(y_true, y_pred, average=average)

    def compute_map(self, y_true, y_scores):
        """
        Compute mean Average Precision (mAP) for multiclass.
        Returns:
            mean_ap (float): Mean of per-class AP.
            ap_per_class (np.ndarray): AP for each class.
        """
        # Binarize labels for one-vs-rest
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        ap_per_class = average_precision_score(y_true_bin, y_scores, average=None)
        mean_ap = np.mean(ap_per_class)
        return mean_ap, ap_per_class

    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        """
        Plots the confusion matrix.
        Args:
            y_true (array): True labels.
            y_pred (array): Predicted labels.
            normalize (bool): Whether to normalize by row sums.
        Returns:
            fig: Matplotlib figure object.
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Setup labels
        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Annotate cells
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        return fig
