import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src import config


class ADE20KDataset(torch.utils.data.Dataset):

    def __init__(self, root, mode='all'):
        assert mode in {"train", "val", 'all'}

        self.root = root
        self.mode: str = mode

        # Load dataset index
        index_file = os.path.join(self.root, "index_ade20k.pkl")
        with open(index_file, "rb") as f:
            self.index: Dict[str, Any] = pickle.load(f)

        if not mode == 'all':
            if mode == 'train':
                predicate = lambda i: self._is_training_folder(self.index['folder'][i])
            else:
                predicate = lambda i: not self._is_training_folder(self.index['folder'][i])
            indices_to_keep = list(filter(predicate, range(0, len(self.index['filename']))))
            self._filter_keep_indices(indices_to_keep)

    def _filter_keep_indices(self, indices_to_keep):
        num_object_classes = len(self.index['objectPresence'])
        num_keep = len(indices_to_keep)

        # Update object presence
        # [C, N]
        updated_object_presence = np.zeros((num_object_classes, num_keep), dtype=np.uint8)
        for class_idx in range(0, num_object_classes):
            for updated_image_idx in range(0, num_keep):
                orig_image_idx = indices_to_keep[updated_image_idx]
                updated_object_presence[class_idx, updated_image_idx] = self.index['objectPresence'][
                    class_idx, orig_image_idx]
        self.index['objectPresence'] = updated_object_presence

        indices_to_keep = set(indices_to_keep)
        self.index['filename'] = [x for i, x in enumerate(self.index['filename']) if i in indices_to_keep]
        self.index['folder'] = [x for i, x in enumerate(self.index['folder']) if i in indices_to_keep]
        self.index['scene'] = [x for i, x in enumerate(self.index['scene']) if i in indices_to_keep]

    @staticmethod
    def _is_training_folder(folder: str) -> bool:
        split = folder.split('/')[3]
        return split == 'training'

    # TODO: some sample code return this as infinite or other value, stating that DataLoader maintain their own list, thus transormations are not effective
    def __len__(self):
        return len(self.index['filename'])

    def _read_image_at_pos(self, idx: int, seg=False) -> Image:
        filename: str = self.index['filename'][idx]
        if seg:
            filename = filename.replace(".jpg", "_seg.png")
        image_path = os.path.join(self.root, '..', self.index['folder'][idx], filename)
        return Image.open(image_path)

    def _convert_seg_image_to_mask(self, seg_image) -> np.array:
        """
        Decodes RGB segmentation image into classes mask; 0 corresponds to background.
        """
        mask = np.array(seg_image)
        r = mask[:, :, 0]
        g = mask[:, :, 1]
        mask = (r / 10).astype(np.int32) * 256 + (g.astype(np.int32))
        mask = mask.astype(np.float32)
        return mask

    def __getitem__(self, idx: int):
        image = self._read_image_at_pos(idx)
        image = np.array(image.convert('RGB'))

        mask = self._read_image_at_pos(idx, seg=True)
        mask = self._convert_seg_image_to_mask(mask)

        return image, mask


class WallADE20KDataset(ADE20KDataset):

    def __init__(self, root, mode='train', filter_scenes=True):
        super().__init__(root, mode)

        self.filter_scenes = filter_scenes

        # Delete items at indices where images do not contain any walls
        num_total_images = len(self.index['filename'])

        indices_to_keep = [i for i in range(0, num_total_images) if self._is_indoor_wall_sample(i)]

        self._filter_keep_indices(indices_to_keep)

    def _is_indoor_wall_sample(self, idx) -> bool:
        if self.index["objectPresence"][config.ADE20K_WALL_CLASS_IDX, idx] == 0:
            # There is no wall object present at this index
            return False

        if not self.filter_scenes:
            return True

        # Additional filtering based on scene whitelist
        scene = self.index['scene'][idx]
        return scene in config.WALL_SCENES

    def _convert_seg_image_to_mask(self, seg_image) -> np.array:
        # Remove all labels except 'wall' and set it to '1'
        mask = super()._convert_seg_image_to_mask(seg_image)
        mask[mask != config.ADE20K_WALL_CLASS_ID] *= 0
        mask[mask == config.ADE20K_WALL_CLASS_ID] = 1
        return mask


class SimpleWallADE20KDataset(WallADE20KDataset):
    # TODO: add transforms, RandomVerticalFlip, etc (see WallSegmentation paper)
    #   And this: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
    def __init__(
            self,
            root, mode='all',
            length: Optional[int] = None,
            filter_scenes: bool = True,
            image_size: Tuple[int, int] = config.INPUT_IMAGE_SIZE,
            augmentation_fn=None,
            preprocessing_fn=None
    ):
        super().__init__(root, mode=mode, filter_scenes=filter_scenes)

        self.image_size = image_size
        self.augmentation_fn = augmentation_fn
        self.preprocessing_fn = preprocessing_fn

        if length:
            length = min(self.__len__(), length)
            # TODO: pick random indices if `shuffle` parameter is passed True
            indices_to_keep = list(range(0, length))
            self._filter_keep_indices(indices_to_keep)

    def __getitem__(self, idx: int):
        image, mask = super().__getitem__(idx)

        # Resize image
        # TODO: consider resizing by preserving padding (make it optional)
        # image = np.array(Image.fromarray(image).resize(self.image_size, Image.BILINEAR))
        # mask = np.array(Image.fromarray(mask).resize(self.image_size, Image.NEAREST))

        if self.augmentation_fn is not None:
            sample = self.augmentation_fn(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        # TODO: wrap in albu lambda, as in car_segmentation.ipynb
        if self.preprocessing_fn is not None:
            sample = self.preprocessing_fn(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']

        # # Convert image from HWC to CHW
        # image = np.moveaxis(image, -1, 0)
        # mask = np.expand_dims(mask, 0)

        return image, mask
