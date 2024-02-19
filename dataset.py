import json
import pathlib

import numpy as np
import torch
import torch.utils.data

from PIL import Image, ImageDraw


class DroneImages(torch.utils.data.Dataset):
    def __init__(self, root: str = 'data'):
        self.root = pathlib.Path(root)
        self.parse_json(self.root / 'descriptor.json')

    def parse_json(self, path: pathlib.Path):
        """
        Reads and indexes the descriptor.json

        The images and corresponding annotations are stored in COCO JSON format. This helper function reads out the images paths and segmentation masks.
        """
        with open(path, 'r') as handle:
            content = json.load(handle)

        self.ids = [entry['id'] for entry in content['images']]
        self.images = {entry['id']: self.root / pathlib.Path(entry['file_name']).name for entry in content['images']}

        # add all annotations into a list for each image
        self.polys = {}
        self.bboxes = {}
        for entry in content['annotations']:
            image_id = entry['image_id']
            self.polys.setdefault(image_id, []).append(entry['segmentation'])
            self.bboxes.setdefault(image_id, []).append(entry['bbox'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a drone image and its corresponding segmentation mask.

        The drone image is a tensor with dimensions [H x W x C=5], where
            H - height of the image
            W - width of the image
            C - (R,G,B,T,H) - five channels being red, green and blue color channels, thermal and depth information

        The corresponding segmentation mask is binary with dimensions [H x W].
        """
        image_id = self.ids[index]

        # deserialize the image from disk
        x = np.load(self.images[image_id])

        polys = self.polys[image_id]
        bboxes = self.bboxes[image_id]
        masks = []
        # generate the segmentation mask on the fly
        for poly in polys:
            mask = Image.new('L', (x.shape[1], x.shape[0],), color=0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(poly[0], fill=1, outline=1)
            masks.append(np.array(mask))

        masks = torch.tensor(np.array(masks))

        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        # bounding boxes are given as [x, y, w, h] but rcnn expects [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        y = {
            'boxes': boxes,  # FloatTensor[N, 4]
            'labels': labels,  # Int64Tensor[N]
            'masks': masks,  # UIntTensor[N, H, W]
        }
        x = torch.tensor(x, dtype=torch.float).permute((2, 0, 1))
        x = x / 255.

        return x, y
