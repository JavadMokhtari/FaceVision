import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from facevision.occlusion.classifier import OcclusionClassifier
from facevision.occlusion.utils import get_dataloaders


def train_classifier():
    DATASET_DIR = r"C:\Users\j.mokhtari\Downloads\datasets\glasses-and-coverings\occluded_faces"
    dataloaders = get_dataloaders(DATASET_DIR)

    classifier = OcclusionClassifier(num_classes=4, pretrained=False)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    classifier.fit(dataloaders, criterion, optimizer, exp_lr_scheduler)
    classifier.plot_history()


if __name__ == '__main__':
    train_classifier()
