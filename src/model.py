import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from torch.optim import Adam


class WallModel(pl.LightningModule):

    def __init__(self, architecture, encoder_name, in_channels, out_classes):
        super().__init__()

        self.model = smp.create_model(
            architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes
        )

        # Preprocessing parameters for image
        self.params = smp.encoders.get_preprocessing_params(encoder_name)

        # Per-channel NCHW
        self.register_buffer("std", torch.tensor(self.params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(self.params["mean"]).view(1, 3, 1, 1))

        # Dice loss or binary cross-entropy
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.stage_outputs = {
            "train": [],
            "val": []
        }

    def forward(self, image):
        image = (image - self.mean) / self.std
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
        assert mask.max() <= 1.0 and mask.min() >= 0.0

        logits_mask = self.forward(image)

        # Predict mask contains logits, and loss_fn param 'from_logits' is set to True
        loss = self.loss_fn(logits_mask, mask)

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
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),
            mask.long(),
            mode="binary"
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        output = {
            "loss": loss,
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
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
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
        optimizer = Adam(self.parameters(), lr=1e-4)
        # scheduler = ExponentialLR(optimizer, gamma=0.95, verbose=True, )
        # eturn [optimizer], [scheduler]
        return optimizer


if __name__ == '__main__':
    # Sample train run
    model = WallModel(
        architecture="DeepLabV3Plus",
        encoder_name="mobileone_s3",
        in_channels=3,
        out_classes=1,
    )
