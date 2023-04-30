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

    def __init__(self, root, mode="train"):
        assert mode in {"train", "val"}

        self.root = root
        self.mode: str = mode

        # Load dataset index
        index_file = os.path.join(self.root, "index_ade20k.pkl")
        with open(index_file, "rb") as f:
            self.index: Dict[str, Any] = pickle.load(f)
        # TODO: filter index based on mode: 'train' or 'val'

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

    def __init__(self, root, mode="train"):
        super().__init__(root, mode)

        self.orig_wall_class_idx = 2977
        # 0 is reserved for background
        self.orig_wall_class_id = self.orig_wall_class_idx + 1

        # Delete items at indices where images do not contain any walls
        indices_to_delete = []
        num_total_images = len(self.index['filename'])
        for i in range(0, num_total_images):
            if self.index["objectPresence"][self.orig_wall_class_idx, i] == 0:
                # There is no wall object present at this index
                indices_to_delete.append(i)
        indices_to_delete = set(indices_to_delete)

        self.index['filename'] = [x for i, x in enumerate(self.index['filename']) if i not in indices_to_delete]
        self.index['folder'] = [x for i, x in enumerate(self.index['folder']) if i not in indices_to_delete]

    def _convert_seg_image_to_mask(self, seg_image) -> np.array:
        # Remove all labels except 'wall' and set it to '1'
        mask = super()._convert_seg_image_to_mask(seg_image)
        mask[mask != self.orig_wall_class_id] *= 0
        mask[mask == self.orig_wall_class_id] = 1
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
