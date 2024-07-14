from typing import Dict, Any
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ImageClassificationBase(nn.Module):
    def training_step(self, batch) -> torch.Tensor:
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch) -> Dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs) -> Dict[str, float | int | bool]:
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result) -> None:
        print(f"""Epoch [{epoch + 1}], {f"last_lr: {result['lrs'][-1]:.5f}," if 'lrs' in result else ''}, train_loss:"""
              f" {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

    @staticmethod
    def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Get max value and its index from output tensor
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class OcclusionDetector(ImageClassificationBase):
    def __init__(self, num_classes: int):
        from pathlib import Path
        import warnings
        warnings.filterwarnings('ignore')

        super().__init__()

        # Use a pretrained model
        # downloading weights from this model when it was trained on ImageNet dataset
        # self.network = resnet34(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
        self.network = resnet34()
        self.network.load_state_dict(torch.load(Path(__file__).parent / 'pretrained' / "resnet34-b627a593.pth"))
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        self.early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)
        self.history = None
        self.TRAIN_ID = None

    def forward(self, xb):
        return self.network(xb)

    @staticmethod
    @torch.no_grad()
    def evaluate_validation(model, val_loader) -> Dict[str, Any]:
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(self, epochs: int, max_lr: float, train_loader,
            val_loader, weight_decay=0, grad_clip=None,
            opt_func=torch.optim.SGD) -> list[dict[str, Any]]:

        from os import makedirs
        from time import time

        torch.cuda.empty_cache()
        history = []
        best_val_acc = 0
        self.TRAIN_ID = str(int(time()))
        models_dir = join(".training_checkpoints", self.TRAIN_ID)
        makedirs(models_dir, exist_ok=True)

        # Set up custom optimizer with weight decay
        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training Phase
            self.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(optimizer.param_groups[0]['lr'])
                sched.step()
            # Validation phase
            result = self.evaluate_validation(self, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            self.early_stopping(result['train_loss'], result['val_loss'])
            self.epoch_end(epoch, result)
            history.append(result)
            if round(result['val_acc'], 2) > best_val_acc:
                best_val_acc = result['val_acc']
                torch.save(self.state_dict(),
                           join(models_dir, f"occlusion_classifier_{self.TRAIN_ID}epoch{epoch + 1}.pt"))
            if self.early_stopping.early_stop:
                self.history = history
                return history
        self.history = history
        return history

    def plot_history(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        n_epochs = len(self.history)

        train_losses = [res['train_loss'] for res in self.history]
        val_losses = [res['val_loss'] for res in self.history]
        val_accuracy = [res['val_acc'] for res in self.history]

        ax.plot(list(range(1, n_epochs + 1)), train_losses, label='Train Loss', color='blue')
        ax.plot(list(range(1, n_epochs + 1)), val_losses, label='Validation Loss', color='red')
        ax.plot(list(range(1, n_epochs + 1)), val_accuracy, label='Validation Accuracy', color='green')

        # Customize the plot
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss | Accuracy')
        ax.set_title('Training and Validation Metrics')
        ax.set_yscale('log')
        ax.legend()

        # Display the plot
        plt.grid(True, alpha=0.5)
        plt.savefig(f"training_metrics_{self.TRAIN_ID}.svg")
        plt.show()


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
