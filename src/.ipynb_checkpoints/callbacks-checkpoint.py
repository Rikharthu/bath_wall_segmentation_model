from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import mlflow

class LearningRateLogging(Callback):

    def __init__(self, log_fn):
        self.log_fn = log_fn

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        lr = trainer.optimizers[0].param_groups[0]['lr']
        params = {
            'learning_rate': lr,
            'global_step': trainer.global_step
        }
        self.log_fn(params)

class MLFlowImageLogging(Callback):

    def __init__(self, dataset: Dataset, dataset_vis: Dataset, threshold=0.5, images_dir='images'):
        super().__init__()
        self.is_training = False
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.dataset_vis = dataset_vis
        self.threshold = threshold
        self.images_dir = images_dir

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.is_training = True

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.is_training = False

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.is_training = True

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.is_training:
            return

        epoch = pl_module.current_epoch
        eval_model = pl_module.eval()
        for idx, (x, _) in enumerate(iter(self.dataloader)):
            x = x.cuda()
            mask_pred = eval_model(x).cpu().detach()
            mask_pred = mask_pred.sigmoid().numpy().squeeze()
            mask_pred[mask_pred >= self.threshold] = 1.0
            mask_pred[mask_pred < self.threshold] = 0.0
            image, mask_gt = self.dataset_vis[idx]

            fig = plt.figure(figsize=(16, 5), frameon=False)

            plt.subplot(1, 3, 1, frameon=False)
            plt.axis('off')
            plt.title('image')
            plt.imshow(image)

            plt.subplot(1, 3, 2, frameon=False)
            plt.axis('off')
            plt.title('pred')
            plt.imshow(mask_pred)

            plt.subplot(1, 3, 3, frameon=False)
            plt.axis('off')
            plt.title('mask')
            plt.imshow(mask_gt)

            fig.tight_layout()

            image_filename = f'{idx}'.zfill(3)
            mlflow.log_figure(fig, f'{self.images_dir}/{epoch}/{image_filename}.jpg')
            plt.close()
