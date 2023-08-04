from typing import List
import torchvision
from models.layers import MixingMaskAttentionBlock, PixelwiseLinear, UpMask
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid


class ChangeClassifier(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks:
        self._first_mix = PixelwiseLinear([3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(
            [
                PixelwiseLinear([24, 12, 6], [12, 6, 1]),
                PixelwiseLinear([32, 16, 8], [16, 8, 1]),
                PixelwiseLinear([56], [56]),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        features = self._encode(x)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, x) -> List[Tensor]:
        features = [self._first_mix(x)]
        for num, layer in enumerate(self._backbone):
            x = layer(x)
            if num != 0:
                features.append(self._mixing_mask[num - 1](x))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model
