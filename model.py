import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchmetrics


class ResNet18Classifier(pl.LightningModule):
    def __init__(
        self, num_classes, learning_rate=1e-4, weight_decay=0.0, base_model="resnet18"
    ):
        super(ResNet18Classifier, self).__init__()

        if base_model == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
        elif base_model == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError(
                f"Unknown base_model architecture: {base_model}. Should be resnet18 or resnet50."
            )

        # Replace the final fully connected layer to match the number of classes in the dataset
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

        self.num_classes = num_classes

        self.train_acc = torchmetrics.classification.Accuracy(
            task="binary", num_classes=num_classes
        )
        self.train_auc = torchmetrics.classification.AUROC(
            task="binary", num_classes=num_classes
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="binary", num_classes=num_classes
        )
        self.valid_auc = torchmetrics.classification.AUROC(
            task="binary", num_classes=num_classes
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="binary", num_classes=num_classes
        )
        self.test_auc = torchmetrics.classification.AUROC(
            task="binary", num_classes=num_classes
        )

        self.test_outputs = []
        self.test_labels = []

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        self.train_acc(logits, y)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)

        self.train_auc(logits, y)
        self.log("train/auc", self.train_auc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("valid/loss", loss, on_step=True, on_epoch=True)

        self.valid_acc(logits, y)
        self.log("valid/acc", self.valid_acc, on_step=True, on_epoch=True)

        self.valid_auc(logits, y)
        self.log("valid/auc", self.valid_auc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)

        logits = self(x)

        self.test_acc(logits, y)
        self.log("test/acc", self.test_acc)

        self.test_auc(logits, y)
        self.log("test/auc", self.test_auc)

        self.test_outputs.append(torch.argmax(logits, dim=1))
        self.test_labels.append(torch.argmax(y, dim=1))

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_outputs)
        all_labels = torch.cat(self.test_labels)

        test_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(
            device=self.device
        )
        bcm = test_confusion_matrix(all_preds, all_labels)

        print("Confusion Matrix:")
        print(bcm)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
