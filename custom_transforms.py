from monai.transforms import MapTransform, LoadImaged
from PIL import Image, ImageDraw
import numpy as np
import torch
from collections.abc import Sequence

class PolyToMaskTransform(MapTransform):
    def __init__(self, keys: str, spatial_size: Sequence[str]) -> None:
        """
        Initialize the PolyToMaskTransform.

        Args:
            keys (str): Keys to extract data for transformation. Assumes data is under "label" key.
            spatial_size (Sequence): size (H x W) of the image.
        """
        super().__init__(keys)
        self.spatial_size = spatial_size

    def __call__(self, data):
        """
        Apply the polygon to mask transformation.

        Args:
            data (dict): The input data with "label" key containing polygons.

        Returns:
            dict: The updated data with masks replacing the original polygons.
        """
        # Ensure `data` is a dictionary and contains the specified key(s).
        #super().__call__(data)

        for key in self.keys:
            if key in data:
                polys = data[key]  # Extract polygons.
                masks = []  # Initialize a list to hold generated masks.
                
                for poly in polys:
                    # Create a mask for each polygon.
                    mask = Image.new('L', self.spatial_size, color=0)  # (H, W).
                    draw = ImageDraw.Draw(mask)
                        
                    draw.polygon(poly[0], fill=1, outline=1)
                        
                    masks.append(np.array(mask))
                
            
                masks_tensor = torch.tensor(np.stack(masks), dtype=torch.float32)
                data[key] = masks_tensor  # Update the input data with the generated masks.
            else:
                # Handle missing key if necessary.
                pass

        return data
