import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
from typing import Optional

from torch.utils.data import DataLoader

from src.dataset import SimpleWallADE20KDataset
from src.transform import get_train_augmentations, get_preprocessing_transform, get_val_augmentations_single
from src import config


class WallModel(pl.LightningModule):
    train_dataset: Optional[SimpleWallADE20KDataset]
    val_dataset: Optional[SimpleWallADE20KDataset]

    def __init__(
            self,
            architecture: str,
            encoder_name: str,
            in_channels: int,
            out_classes: int,
            learning_rate: float = config.LEARNING_RATE,
            train_size: Optional[int] = None,
            val_size: Optional[int] = None,
            init_datasets: bool = False,
            encoder_depth: int = config.ENCODER_DEPTH,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.train_size = train_size
        self.val_size = val_size
        self.encoder_depth = encoder_depth

        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_depth=encoder_depth,
        )

        # Preprocessing parameters for image
        self.params = smp.encoders.get_preprocessing_params(encoder_name)

        if init_datasets:
            self.train_dataset = self._create_train_dataset()
            self.val_dataset = self._create_val_dataset()
        else:
            self.train_dataset = None
            self.val_dataset = None

        # Dice loss or binary cross-entropy
        # self.losses = [
        #     ("jaccard", 0.1, smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)),
        #     ("focal", 0.9, smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE)),
        # ]
        # self.losses = [
        #     # ('cross-entropy', 0.9, torch.nn.CrossEntropyLoss()),
        #     ('binary-cross-entropy', 0.9, torch.nn.BCEWithLogitsLoss()),
        #     # ('soft-cross-entropy', 0.9, smp.losses.SoftCrossEntropyLoss(mode)),
        #     ("jaccard", 0.1, smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)),
        # ]typ
        # self.losses = [
        #     ('binary-cross-entropy', 1.0, torch.nn.BCEWithLogitsLoss()),
        # ]
        # self.losses = [
        #     ('dice', 1.0, smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True))
        # ]
        
        
        # self.losses = [
        #     ("jaccard", 0.1, smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)),
        #     ('binary-cross-entropy', 0.9, torch.nn.BCEWithLogitsLoss()),
        # ]
        self.losses = [
            ('focal', 1.0, smp.losses.FocalLoss(mode=smp.losses.BINARY_MODE, ignore_index=-1))
        ]
        # self.losses = [
        #     ('soft-bce-with-logits', 1.0, smp.losses.SoftBCEWithLogitsLoss(ignore_index=-1))
        # ]
        

        self.stage_outputs = {
            "train": [],
            "val": []
        }
        self.best_metrics = {}

        self.optimizer = SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-4, momentum=0.9)
        self.scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2, optimizer=self.optimizer)

    def _create_train_dataset(self):
        return SimpleWallADE20KDataset(
            root=config.DATA_ROOT,
            mode='train',
            length=self.train_size,
            augmentation_fn=get_train_augmentations(mask_pad_val=-1.0),
            preprocessing_fn=get_preprocessing_transform(config.ENCODER),
        )

    def _create_val_dataset(self):
        return SimpleWallADE20KDataset(
            root=config.DATA_ROOT,
            mode='val',
            length=self.val_size,
            augmentation_fn=get_val_augmentations_single(mask_pad_val=-1.0),
            preprocessing_fn=get_preprocessing_transform(config.ENCODER)
        )

    def forward(self, image):
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def _shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width) : NCHW
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Assert NCHW
        assert mask.ndim == 4
        # assert mask.max() <= 1.0 and mask.min() >= 0.0

        logits_mask = self.forward(image)

        # Predict mask contains logits, and loss_fn param 'from_logits' is set to True
        # loss = self.loss_fn(logits_mask, mask)
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            ls_mask = loss(logits_mask, mask)
            total_loss += weight * ls_mask
            self.log(f'{stage}_{loss_name}_loss', ls_mask, prog_bar=True, on_step=False, on_epoch=True)

        # Lets compute metrics for some threshold
        # First convert mask values to probabilities, then apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative
        # and true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        # tp, fp, fn, tn = smp.metrics.get_stats(
        #     pred_mask.long(),
        #     mask.long(),
        #     mode="binary"
        # )
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode='multiclass',
            num_classes=2,
            ignore_index=-1,
        )

        self.log(f'{stage}_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)

        output = {
            "loss": total_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        self.stage_outputs[stage].append(output)

        return output

    def _shared_epoch_end(self, stage):
        outputs = self.stage_outputs[stage]

        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # Dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

        outputs.clear()

    def training_step(self, batch, batch_idx):
        # TODO: log learning rate as in https://github.com/ternaus/cloths_segmentation/blob/main/cloths_segmentation/train.py
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        # self.log('learning_rate', self.current_learning_rate(), prog_bar=False)
        return self._shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        return self._shared_epoch_end("val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self._shared_epoch_end("test")

    def configure_optimizers(self):
        # optimizer = Adam(self.parameters(), lr=self.learning_rate)
        # TODO: try this
        # optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.optimizer = SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-4, momentum=0.9)

        # scheduler = ExponentialLR(optimizer, gamma=0.95, verbose=True, )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(T_0=10, T_mult=2, optimizer=self.optimizer)
        # scheduler = OneCycleLR(optimizer, max_lr, epochs=epoch, steps_per_ephc=len(train_loader)
        return [self.optimizer], [self.scheduler]

    def current_learning_rate(self):
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            optimizer = optimizers[0]
        else:
            optimizer = optimizers
        return optimizer.param_groups[0]['lr']

    def train_dataloader(self):
        if self.train_dataset is None:
            self.train_dataset = self._create_train_dataset()
        n_cpu = os.cpu_count()
        return DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=n_cpu)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.val_dataset = self._create_val_dataset()
        n_cpu = os.cpu_count()
        # Single batch size because images are of different size
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)


if __name__ == '__main__':
    # Sample train run
    model = WallModel(
        architecture="DeepLabV3Plus",
        encoder_name="mobileone_s3",
        in_channels=3,
        out_classes=1,
    )
