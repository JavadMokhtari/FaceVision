from typing import Dict, Optional
from os.path import join
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

from facevision.models.inception_resnet_v1 import InceptionResnetV1
from facevision.occlusion.utils import EarlyStopping


class OcclusionClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, model_path: Optional[str] = None):
        from pathlib import Path
        import warnings
        warnings.filterwarnings('ignore')

        super().__init__()

        root = Path(__file__).parent.parent.parent
        RESNET_WEIGHTS_PATH = root / "assets/weights/resnet50-11ad3fa6.pth"
        INCEPTIONRESNET_WEIGHTS_PATH = root / "assets/weights/20180402-114759-vggface2.pt"

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Load pre-trained ResNet50
        self.resnet = resnet50()
        self.resnet.load_state_dict(torch.load(RESNET_WEIGHTS_PATH))

        # Freeze pre-trained layers
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # Loading InceptionResNetV1 model
        self.inception_resnet = InceptionResnetV1(
            weights=INCEPTIONRESNET_WEIGHTS_PATH, dropout_prob=0.5, device=self.device)

        # Freeze pre-trained layers
        # for param in self.inception_resnet.parameters():
        #     param.requires_grad = False

        # Adding a fully connected layer to ResNet50 with size 512
        self.resnet_fx = nn.Linear(self.resnet.fc.out_features, out_features=256)

        # Adding a fully connected layer to model with size 512
        self.inceptionResnet_fx = nn.Linear(self.inception_resnet.last_bn.num_features, 256)

        # Define 3 last fully connected layers
        self.lin1 = nn.Linear(512, 256)
        self.lin2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, num_classes)

        if pretrained:
            self.load_state_dict(model_path)

        self.early_stopping = EarlyStopping(tolerance=5, min_delta=0.2)
        self.history = {'train': {'loss': [], 'acc': []},
                        'val': {'loss': [], 'acc': []}}

        self.to(torch.device(self.device))

    def forward(self, xb):
        resnet_out = self.resnet(xb)
        resnet_out = self.resnet_fx(resnet_out)

        inception_out = self.inception_resnet(xb)
        inception_out = self.inceptionResnet_fx(inception_out)

        in_features = torch.cat((resnet_out, inception_out), dim=1)
        out = self.lin1(in_features)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.out(out)
        out = F.softmax(out, dim=1)
        return out

    def fit(self, dataloaders: Dict, criterion, optimizer, scheduler, num_epochs: int = 100):
        """
        Support function for model training.
  
        Args:
          self: Model to be trained
          dataloaders: Dictionary of dataloaders have keys 'train' and 'val'
          criterion: Optimization criterion (loss)
          optimizer: Optimizer to use for training
          scheduler: Instance of ``torch.optim.lr_scheduler``
          num_epochs: Number of epochs
        """
        from os import makedirs

        since = time()
        best_acc = 0.0
        makedirs(".training_checkpoints", exist_ok=True)
        MODEL_PATH = join(".training_checkpoints", f"occlusion_classifier_best_weights.pt")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                match phase:
                    case 'train':
                        self.train()  # Set model to training mode
                    case 'val':
                        self.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects / len(dataloaders[phase])

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                self.history[phase]['loss'].append(epoch_loss)
                self.history[phase]['acc'].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.state_dict(), MODEL_PATH)

            self.early_stopping(self.history['train']['loss'][-1], self.history['val']['loss'][-1])
            if self.early_stopping.stop_flag:
                break
            print()

        time_elapsed = time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.load_state_dict(MODEL_PATH)
        return self.history

    def plot_history(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x_range = list(range(1, len(self.history['train']['loss']) + 1))

        ax.plot(x_range, self.history['train']['loss'], label='Train Loss', color='blue')
        ax.plot(x_range, self.history['val']['loss'], label='Validation Loss', color='red')
        ax.plot(x_range, self.history['train']['acc'], label='Train Accuracy', color='purple')
        ax.plot(x_range, self.history['val']['acc'], label='Validation Accuracy', color='green')

        # Customize the plot
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss | Accuracy')
        ax.set_title('Training and Validation Metrics')
        ax.set_yscale('log')
        ax.legend()

        # Display the plot
        plt.grid(True, alpha=0.5)
        plt.savefig(f"training_metrics.svg")
        plt.show()
