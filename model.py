from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from monai.networks.nets import UNet, SwinUNETR, FlexibleUNet

def MaskRCNN(in_channels=5, num_classes=2, image_mean=None, image_std=None, **kwargs):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225, 0.225, 0.225]

    model = maskrcnn_resnet50_fpn_v2(
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std
    )
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model

def UNet_model(in_channels=5, num_classes=2, **kwargs):
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(4, 8, 16, 32, 64, 256),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        dropout = 0.2
    )
    
    return model


def UNet_small_model(in_channels=5, num_classes=2, **kwargs):
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout = 0.5
    )
    
    return model

def SwinUNETR_model(in_channels=5, num_classes=2, **kwargs):
        model = SwinUNETR(
                    spatial_dims=2,
                    img_size=(2688, 3392),
                    in_channels=in_channels,
                    out_channels=num_classes,
                    feature_size=48,
                    depths=[2, 2, 2, 2],
                    num_heads=[3, 6, 12, 24],
                    drop_rate=0.2,
                    attn_drop_rate=0.1,
                    use_checkpoint=True,
                    use_v2=True
                )
        
        return model
    
def EfficientUNet_model(in_channels=2, num_classes=2, **kwargs):
        model = FlexibleUNet(pretrained=True, 
                             in_channels=in_channels, 
                             out_channels=num_classes, 
                             backbone="efficientnet-b1")
        
        return model
    
def EfficientUNet_small_model(in_channels=2, num_classes=2, **kwargs):
        model = FlexibleUNet(pretrained=True, 
                             in_channels=in_channels, 
                             out_channels=num_classes, 
                             backbone="efficientnet-b0")
        
        return model