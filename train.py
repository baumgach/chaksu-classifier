import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import ResNetClassifier
import argparse

from chaksu import Chaksu_Classification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training model.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Base name for experiment.",
        required=True,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Strength of weight-decay term; 0 means no weight decay.",
        default=0.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for ADAM optimizer.",
        default=1e-4,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model architecture ('resnet18', 'resnet50').",
        default="resnet18",
    )
    parser.add_argument(
        "--use_rois",
        action="store_true",
        help="Use ROIs. If false uses the whole image.",
    )

    args = parser.parse_args()

    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            filename="best-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val/dice",
            filename="best-dice-{epoch}-{step}",
            mode="max",
        ),
    ]

    if args.use_rois:
        Chaksu = "/mnt/qb/work/baumgartner/bkc562/ResearchProject/Chaksu/Chaksu_ROI.h5"
    else:
        Chaksu = "/mnt/qb/work/baumgartner/bkc562/ResearchProject/Chaksu/Chaksu.h5"

    train_dataset = Chaksu_Classification(file_path=Chaksu, t="train")
    valid_dataset = Chaksu_Classification(file_path=Chaksu, t="val")

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    experiment_name = args.experiment_name + f"-{args.base_model}"
    experiment_name += f"-LR{str(args.learning_rate)}"
    experiment_name += f"-WD{str(args.weight_decay)}"
    if args.use_rois:
        experiment_name += "-rois"
    else:
        experiment_name += "-fullimgs"

    logger = TensorBoardLogger(
        save_dir="./runs", name=experiment_name, default_hp_metric=False
    )

    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="valid/auc_epoch",
            filename="best-auc-{epoch}-{step}",
            mode="max",
        ),
        ModelCheckpoint(
            monitor="valid/loss_epoch",
            filename="best-loss-{epoch}-{step}",
            mode="min",
        ),
    ]

    # Create the model and trainer
    model = ResNetClassifier(
        num_classes=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        base_model=args.base_model,
    )
    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=10,
        val_check_interval=0.25,
        logger=logger,
        callbacks=checkpoint_callbacks,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
