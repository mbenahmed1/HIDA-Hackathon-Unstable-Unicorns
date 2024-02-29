import numpy as np
import json
import pathlib

from custom_transforms import PolyToMaskTransform 
from monai.transforms import  LoadImaged
from monai.data import Dataset
import torch

from torchvision.transforms.functional import rgb_to_grayscale

from monai.transforms import (
    Compose,
    RandFlipd,
    LoadImaged,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    RandZoomd,
    ToTensord,
    Transposed,
    Lambdad
)



def parse_json(path: pathlib.Path, root: pathlib.Path):
    with open(path, 'r') as handle:
        content = json.load(handle)

    ids = [entry['id'] for entry in content['images']]
    images = {entry['id']: root / pathlib.Path(entry['file_name']).name for entry in content['images']}

    # add all annotations into a list for each image
    polys = {}
    
    for entry in content['annotations']:
        image_id = entry['image_id']
        polys.setdefault(image_id, []).append(entry['segmentation'])
        
    return ids, images, polys


def get_DroneImages_datalist(root: str, predict: bool = False):
    root = pathlib.Path(root)

    if predict:
        ids, images, polys = parse_json(root / 'descriptor.json', root)
        new_ids, new_images, new_polys = ids, images, polys
    else:
        ids, images, polys = parse_json(root / 'old_descriptor.json', root)
        old_ids, old_images, old_polys = ids, images, polys

        ids, images, polys = parse_json(root / 'new_descriptor.json', root)
        new_ids, new_images, new_polys = ids, images, polys
        
    datalist = []
    for index in range(len(new_ids)):
        d = {}
        if index <= 322 and not predict:
            image_id = old_ids[index]
        else:
            image_id = new_ids[index]

        # deserialize the image from disk
        if index <= 322 and not predict:
            d["image"] = old_images[image_id]
        
        else:
            d["image"] = new_images[image_id]

        if index <= 322 and not predict:
            op = old_polys[image_id]
            for ps in op:
                ps[0] = [i + ((ind + 1) % 2) * 60 for ind,i in zip(range(len(ps[0])), ps[0])]
            d["label"] = op
            
        else:
            d["label"] = new_polys[image_id]
        
        datalist.append(d)
        
    return datalist
    
    
    
    
def get_DroneImages_dataset(datalist, augmentation = False, in_channels = 2):
    assert in_channels in (2,3,5), f'in_channels can only have values of 2, 3 or 5. Given as {in_channels}'
    
    if in_channels == 2:
        CropChannels = Lambdad(keys="image", func=lambda x: x[-2:, :, :])
    elif in_channels == 3:
        CropChannels = Lambdad(keys="image", 
                               func=lambda x: torch.cat([rgb_to_grayscale(x[:3]), x[3:]]))
    else:
        CropChannels = Lambdad(keys="image", func=lambda x: x)
        
    if augmentation:
        transforms = Compose([
            LoadImaged(keys="image"),
            Transposed(keys="image", indices=[2, 0, 1]),
            ScaleIntensityRanged(
                keys=["image"], a_min=0., a_max=255., b_min=0., b_max=1., clip=True
            ),
            CropChannels,
            PolyToMaskTransform(keys="label", spatial_size=[3370, 2680]),
            Lambdad(keys="label", func=lambda x: x.sum(dim=0).clamp(0., 1.)[None, :, :]),
            RandZoomd(keys=["image", "label"], 
                      prob = 0.5, 
                      mode=["bilinear", "nearest"],
                      min_zoom = 1.0, max_zoom = 1.5
            ),
            RandFlipd(keys=["image", "label"], 
                      prob = 0.5, 
                      spatial_axis = 1
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(2688, 3392)),
            ToTensord(keys=["image", "label"])
        ]
        )
        
    else:
        transforms = Compose([
            LoadImaged(keys="image"),
            Transposed(keys="image", indices=[2, 0, 1]),
            ScaleIntensityRanged(
                keys=["image"], a_min=0., a_max=255., b_min=0., b_max=1., clip=True
            ),
            CropChannels,
            PolyToMaskTransform(keys="label", spatial_size=[3370, 2680]),
            Lambdad(keys="label", func=lambda x: x.sum(dim=0).clamp(0., 1.)[None, :, :]),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(2688, 3392)),
            ToTensord(keys=["image", "label"])
        ]
        )
        
    dataset = Dataset(datalist, transform=transforms)
    return dataset