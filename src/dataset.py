import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from torchvision import transforms as T


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
        num_classes = len(self.index['objectPresence'])
        num_keep = len(indices_to_keep)

        # Update object presence
        updated_object_presence = np.zeros((num_classes, num_keep), dtype=np.uint8)
        for class_idx in range(0, num_classes):
            for updated_image_idx in range(0, num_keep):
                orig_image_idx = indices_to_keep[updated_image_idx]
                updated_object_presence[class_idx, updated_image_idx] = self.index['objectPresence'][
                    class_idx, orig_image_idx]
        self.index['objectPresence'] = updated_object_presence

        indices_to_keep = set(indices_to_keep)
        self.index['filename'] = [x for i, x in enumerate(self.index['filename']) if i in indices_to_keep]
        self.index['folder'] = [x for i, x in enumerate(self.index['folder']) if i in indices_to_keep]

    @staticmethod
    def _is_training_folder(folder: str) -> bool:
        split = folder.split('/')[3]
        return split == 'training'

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
    ADE20K_WALL_CLASS_IDX = 2977
    # 0 is reserved for background
    ADE20K_WALL_CLASS_ID = ADE20K_WALL_CLASS_IDX + 1

    def __init__(self, root, mode="train"):
        super().__init__(root, mode)

        # Delete items at indices where images do not contain any walls
        indices_to_delete = []
        num_total_images = len(self.index['filename'])
        for i in range(0, num_total_images):
            if self.index["objectPresence"][self.ADE20K_WALL_CLASS_IDX, i] == 0:
                # There is no wall object present at this index
                indices_to_delete.append(i)
        indices_to_delete = set(indices_to_delete)

        self.index['filename'] = [x for i, x in enumerate(self.index['filename']) if i not in indices_to_delete]
        self.index['folder'] = [x for i, x in enumerate(self.index['folder']) if i not in indices_to_delete]

    def _convert_seg_image_to_mask(self, seg_image) -> np.array:
        # Remove all labels except 'wall' and set it to '1'
        mask = super()._convert_seg_image_to_mask(seg_image)
        mask[mask != self.ADE20K_WALL_CLASS_ID] *= 0
        mask[mask == self.ADE20K_WALL_CLASS_ID] = 1
        return mask


class SimpleWallADE20KDataset(WallADE20KDataset):
    IMAGE_SIZE = (512, 512)

    def __init__(self, root, mode="train"):
        super().__init__(root, mode)

    def __getitem__(self, idx: int):
        image, mask = super().__getitem__(idx)

        # Resize image
        image = np.array(Image.fromarray(image).resize(self.IMAGE_SIZE, Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize(self.IMAGE_SIZE, Image.NEAREST))

        # Convert image from HWC to CHW
        image = np.moveaxis(image, -1, 0)
        mask = np.expand_dims(mask, 0)

        return image, mask
