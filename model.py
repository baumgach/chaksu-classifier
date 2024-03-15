import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchmetrics
from sklearn import metrics
import numpy as np

class ResNetClassifier(pl.LightningModule):
    def __init__(
        self, num_classes, learning_rate=1e-4, weight_decay=0.0, base_model="resnet18"
    ):
        super(ResNetClassifier, self).__init__()

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

        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.train_auc = torchmetrics.classification.AUROC(task="binary")
        self.valid_acc = torchmetrics.classification.Accuracy(task="binary")
        self.valid_auc = torchmetrics.classification.AUROC(task="binary")

        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_auc = torchmetrics.classification.AUROC(task="binary")
        self.test_sensitivity = torchmetrics.classification.Recall(task="binary")
        self.test_specificity = torchmetrics.classification.Specificity(task="binary")

        self.test_outputs = []
        self.test_labels = []
        self.test_probas = []

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probas = F.softmax(logits, dim=1)[:, 1]

        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)

        self.train_acc(preds, y[:, 1])
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)

        self.train_auc(probas, y[:, 1])
        self.log("train/auc", self.train_auc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probas = F.softmax(logits, dim=1)[:, 1]

        loss = self.criterion(logits, y)
        self.log("valid/loss", loss, on_step=True, on_epoch=True)

        self.valid_acc(preds, y[:, 1])
        self.log("valid/acc", self.valid_acc, on_step=True, on_epoch=True)

        self.valid_auc(probas, y[:, 1])
        self.log("valid/auc", self.valid_auc, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.squeeze(), num_classes=self.num_classes).type(torch.float32)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probas = F.softmax(logits, dim=1)[:, 1]

        self.test_acc(preds, y[:, 1])
        self.log("test/acc", self.test_acc)

        self.test_auc(probas, y[:, 1])
        self.log("test/auc", self.test_auc)

        self.test_sensitivity(preds, y[:, 1])
        self.log("test/sensitivity", self.test_sensitivity)

        self.test_specificity(preds, y[:, 1])
        self.log("test/specificity", self.test_specificity)

        self.test_outputs.append(preds)
        self.test_labels.append(torch.argmax(y, dim=1))
        self.test_probas.append(probas)

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_outputs)
        all_labels = torch.cat(self.test_labels)
        all_probas = torch.cat(self.test_probas)

        test_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix().to(
            device=self.device
        )
        bcm = test_confusion_matrix(all_preds, all_labels)

        print("Confusion Matrix:")
        print(bcm)

        # Sanity check
        test_auroc = torchmetrics.classification.AUROC(task="binary").to(
            device=self.device
        )
        print("Sanity AUROC:")
        print(test_auroc(all_probas, all_labels))

        fpr, tpr, thresholds = metrics.roc_curve(
            all_labels.cpu(), all_probas.cpu(), pos_label=1
        )
        print("SKLEARN AUROC:", metrics.auc(fpr, tpr))

        # Calculate the Youden Index
        youden_index = tpr - fpr

        # Find the optimal threshold
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]

        print("Optimal threshold:", optimal_threshold)
        preds_threshold = (all_probas.cpu().detach().numpy() >= optimal_threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(all_labels.cpu(), preds_threshold).ravel()

        # Compute sensitivity and specificityß
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)



    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
