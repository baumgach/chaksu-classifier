from model import ResNet18Classifier
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from chaksu import Chaksu_Classification


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating trained models."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=dir_path,
        help="Path to directory containing the checkpoint files. The most recent checkpoint will be loaded.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_identifier",
        type=str,
        help="Filter by checkpoint identifier such aus 'auc' or 'loss'",
        default=None,
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

    checkpoint_dir = args.checkpoints_dir
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if args.checkpoint_identifier is not None:
        checkpoints = [f for f in checkpoints if args.checkpoint_identifier in f]

    # Sort the checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split("=")[1].split("-")[0]))

    # Get the most recent checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    if args.use_rois:
        Chaksu = (
            "/mnt/qb/work/baumgartner/bkc562/ResearchProject/Chaksu/Chaksu_ROI_test.h5"
        )
    else:
        Chaksu = "/mnt/qb/work/baumgartner/bkc562/ResearchProject/Chaksu/Chaksu_test.h5"

    test_dataset = Chaksu_Classification(file_path=Chaksu, t="test")
    test_loader = DataLoader(
        test_dataset, batch_size=2, drop_last=False, shuffle=False
    )

    model = ResNet18Classifier.load_from_checkpoint(
        num_classes=2,
        checkpoint_path=latest_checkpoint,
        base_model=args.base_model,
    )

    trainer = pl.Trainer()

    # test the model
    trainer.test(model, dataloaders=test_loader)
