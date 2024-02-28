import json
import pathlib
import copy

import numpy as np
import torch
import torch.utils.data

import copy

from PIL import Image, ImageDraw

from monai.transforms import ResizeWithPadOrCrop
import torchvision
import random
from torchvision.transforms.v2.functional import horizontal_flip, crop, resize
from torchvision.transforms import v2

class DroneImages(torch.utils.data.Dataset):
    def __init__(self, root: str = 'data', predict: bool = False, in_channels: int = 5, return_dict_y: bool = True):
        self.root = pathlib.Path(root)
        self.predict = predict
        if self.predict:
            self.parse_json(self.root / 'descriptor.json')
            self.new_ids, self.new_images, self.new_polys, self.new_bboxes = self.ids, self.images, self.polys, self.bboxes
        else:
            self.parse_json(self.root / 'old_descriptor.json')
            self.old_ids, self.old_images, self.old_polys, self.old_bboxes = self.ids, self.images, self.polys, self.bboxes
            self.parse_json(self.root / 'new_descriptor.json')
            self.new_ids, self.new_images, self.new_polys, self.new_bboxes = self.ids, self.images, self.polys, self.bboxes
            
        assert in_channels in (2,3,5), f'in_channels can only have values of 2, 3 or 5. Given as {in_channels}'
        self.in_channels = in_channels
        self.return_dict_y = return_dict_y
        self.resizer = ResizeWithPadOrCrop(spatial_size = (2688, 3392))
        
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
        if index <= 322 and not self.predict:
            image_id = self.old_ids[index]
        else:
            image_id = self.new_ids[index]

        # deserialize the image from disk
        if index <= 322 and not self.predict:
            x = np.load(self.old_images[image_id])
        else:
            x = np.load(self.new_images[image_id])

        if index <= 322 and not self.predict:
            polys = self.old_polys[image_id]
            bboxes = self.old_bboxes[image_id]
            masks = []
        else:
            polys = self.new_polys[image_id]
            bboxes = self.new_bboxes[image_id]
            masks = []
            
        # generate the segmentation mask on the fly
        for poly in polys:
            mask = Image.new('L', (x.shape[1], x.shape[0],), color=0)
            draw = ImageDraw.Draw(mask)
            if index <= 322 and not self.predict:
                draw.polygon([i + ((ind + 1) % 2) * 60 for ind,i in zip(range(len(poly[0])), poly[0])], fill=1, outline=1)
            else:
                draw.polygon(poly[0], fill=1, outline=1)
                
            masks.append(np.array(mask))

        masks = torch.tensor(np.array(masks))
        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        
        # bounding boxes are given as [x, y, w, h] but rcnn expects [x1, y1, x2, y2]
        boxes[:, 0] += 60 if index <= 322 and not self.predict else 0
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        y = {
            'boxes': boxes,  # FloatTensor[N, 4]
            'labels': labels,  # Int64Tensor[N]
            'masks': masks,  # UIntTensor[N, H, W]
        }
        x = torch.tensor(x, dtype=torch.float).permute((2, 0, 1))
        x = x / 255.

        if self.in_channels==2:
            x = x[3:] # return only 3rd and 4th channel (exclude RGB and include only depth height)
        
        elif self.in_channels==3:
            dummy_img_rgb = torchvision.transforms.functional.rgb_to_grayscale(x[:3])
            dummy_img_rest = x[3:]
            x = torch.cat([dummy_img_rgb, dummy_img_rest])
        
        else:
            pass            
            
        if self.return_dict_y==False:
            y = self.resizer(y['masks'].sum(dim=0).clamp(0., 1.)[None, :, :])
            x = self.resizer(x)
        
        x_non_T = copy.deepcopy(x)
        y_non_T = copy.deepcopy(y)

        # horizontal flip
        if random.uniform(0, 1) <= 0.5:
            print('Apply horizontal flip')
            x = horizontal_flip(x)
            if 'masks' in y:
                y['masks'] = horizontal_flip(y['masks'].sum(dim=0).clamp(0., 1.)[None, :, :])
            else:
                y = horizontal_flip(y)

        H = 2680
        W = 3370

        # random crop
        if random.uniform(0, 1) <= 0.5:
            print('Apply random crop')
            i, j, h, w = v2.RandomCrop.get_params(
            x, output_size=(int(H / 2), int(W / 2)))
            x = crop(x, i, j, h, w)
            x = resize(x, [H, W])

            if 'masks' in y:
                y['masks'] = crop(y['masks'].sum(dim=0).clamp(0., 1.)[None, :, :], i, j, h, w)
                y['masks'] = resize(y['masks'].sum(dim=0).clamp(0., 1.)[None, :, :], [H, W], interpolation=0)
            else:
                y = crop(y, i, j, h, w)
                y = resize(y, [H, W])

        return x, y
