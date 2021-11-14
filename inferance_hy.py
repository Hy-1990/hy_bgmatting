#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/14 21:24
# @Author  : 至尊宝
# @Site    : 
# @File    : inferance_hy.py
import argparse
import torch
import os

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, List, Tuple
import glob
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor
import torchvision
import numpy as np
import cv2
import uuid


# --------------- hy ---------------
class HomographicAlignment:
    """
    Apply homographic alignment on background to match with the source image.
    """

    def __init__(self):
        self.detector = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)

    def __call__(self, src, bgr):
        src = np.asarray(src)
        bgr = np.asarray(bgr)

        keypoints_src, descriptors_src = self.detector.detectAndCompute(src, None)
        keypoints_bgr, descriptors_bgr = self.detector.detectAndCompute(bgr, None)

        matches = self.matcher.match(descriptors_bgr, descriptors_src, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        num_good_matches = int(len(matches) * 0.15)
        matches = matches[:num_good_matches]

        points_src = np.zeros((len(matches), 2), dtype=np.float32)
        points_bgr = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points_src[i, :] = keypoints_src[match.trainIdx].pt
            points_bgr[i, :] = keypoints_bgr[match.queryIdx].pt

        H, _ = cv2.findHomography(points_bgr, points_src, cv2.RANSAC)

        h, w = src.shape[:2]
        bgr = cv2.warpPerspective(bgr, H, (w, h))
        msk = cv2.warpPerspective(np.ones((h, w)), H, (w, h))

        # For areas that is outside of the background,
        # We just copy pixels from the source.
        bgr[msk != 1] = src[msk != 1]

        src = Image.fromarray(src)
        bgr = Image.fromarray(bgr)

        return src, bgr


class Refiner(nn.Module):
    # For TorchScript export optimization.
    __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']

    def __init__(self,
                 mode: str,
                 sample_pixels: int,
                 threshold: float,
                 kernel_size: int = 3,
                 prevent_oversampling: bool = True,
                 patch_crop_method: str = 'unfold',
                 patch_replace_method: str = 'scatter_nd'):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']
        assert kernel_size in [1, 3]
        assert patch_crop_method in ['unfold', 'roi_align', 'gather']
        assert patch_replace_method in ['scatter_nd', 'scatter_element']

        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.prevent_oversampling = prevent_oversampling
        self.patch_crop_method = patch_crop_method
        self.patch_replace_method = patch_replace_method

        channels = [32, 24, 16, 12, 4]
        self.conv1 = nn.Conv2d(channels[0] + 6 + 4, channels[1], kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(channels[2] + 6, channels[3], kernel_size, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size, bias=True)
        self.relu = nn.ReLU(True)

    def forward(self,
                src: torch.Tensor,
                bgr: torch.Tensor,
                pha: torch.Tensor,
                fgr: torch.Tensor,
                err: torch.Tensor,
                hid: torch.Tensor):
        H_full, W_full = src.shape[2:]
        H_half, W_half = H_full // 2, W_full // 2
        H_quat, W_quat = H_full // 4, W_full // 4

        src_bgr = torch.cat([src, bgr], dim=1)

        if self.mode != 'full':
            err = F.interpolate(err, (H_quat, W_quat), mode='bilinear', align_corners=False)
            ref = self.select_refinement_regions(err)
            idx = torch.nonzero(ref.squeeze(1))
            idx = idx[:, 0], idx[:, 1], idx[:, 2]

            if idx[0].size(0) > 0:
                x = torch.cat([hid, pha, fgr], dim=1)
                x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
                x = self.crop_patch(x, idx, 2, 3 if self.kernel_size == 3 else 0)

                y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
                y = self.crop_patch(y, idx, 2, 3 if self.kernel_size == 3 else 0)

                x = self.conv1(torch.cat([x, y], dim=1))
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = F.interpolate(x, 8 if self.kernel_size == 3 else 4, mode='nearest')
                y = self.crop_patch(src_bgr, idx, 4, 2 if self.kernel_size == 3 else 0)

                x = self.conv3(torch.cat([x, y], dim=1))
                x = self.bn3(x)
                x = self.relu(x)
                x = self.conv4(x)

                out = torch.cat([pha, fgr], dim=1)
                out = F.interpolate(out, (H_full, W_full), mode='bilinear', align_corners=False)
                out = self.replace_patch(out, x, idx)
                pha = out[:, :1]
                fgr = out[:, 1:]
            else:
                pha = F.interpolate(pha, (H_full, W_full), mode='bilinear', align_corners=False)
                fgr = F.interpolate(fgr, (H_full, W_full), mode='bilinear', align_corners=False)
        else:
            x = torch.cat([hid, pha, fgr], dim=1)
            x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
            y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
            if self.kernel_size == 3:
                x = F.pad(x, (3, 3, 3, 3))
                y = F.pad(y, (3, 3, 3, 3))

            x = self.conv1(torch.cat([x, y], dim=1))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            if self.kernel_size == 3:
                x = F.interpolate(x, (H_full + 4, W_full + 4))
                y = F.pad(src_bgr, (2, 2, 2, 2))
            else:
                x = F.interpolate(x, (H_full, W_full), mode='nearest')
                y = src_bgr

            x = self.conv3(torch.cat([x, y], dim=1))
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)

            pha = x[:, :1]
            fgr = x[:, 1:]
            ref = torch.ones((src.size(0), 1, H_quat, W_quat), device=src.device, dtype=src.dtype)

        return pha, fgr, ref

    def select_refinement_regions(self, err: torch.Tensor):
        """
        Select refinement regions.
        Input:
            err: error map (B, 1, H, W)
        Output:
            ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
        """
        if self.mode == 'sampling':
            # Sampling mode.
            b, _, h, w = err.shape
            err = err.view(b, -1)
            idx = err.topk(self.sample_pixels // 16, dim=1, sorted=False).indices
            ref = torch.zeros_like(err)
            ref.scatter_(1, idx, 1.)
            if self.prevent_oversampling:
                ref.mul_(err.gt(0).float())
            ref = ref.view(b, 1, h, w)
        else:
            # Thresholding mode.
            ref = err.gt(self.threshold).float()
        return ref

    def crop_patch(self,
                   x: torch.Tensor,
                   idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   size: int,
                   padding: int):
        """
        Crops selected patches from image given indices.

        Inputs:
            x: image (B, C, H, W).
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            patch: (P, C, h, w), where h = w = size + 2 * padding.
        """
        if padding != 0:
            x = F.pad(x, (padding,) * 4)

        if self.patch_crop_method == 'unfold':
            # Use unfold. Best performance for PyTorch and TorchScript.
            return x.permute(0, 2, 3, 1) \
                .unfold(1, size + 2 * padding, size) \
                .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]
        elif self.patch_crop_method == 'roi_align':
            # Use roi_align. Best compatibility for ONNX.
            idx = idx[0].type_as(x), idx[1].type_as(x), idx[2].type_as(x)
            b = idx[0]
            x1 = idx[2] * size - 0.5
            y1 = idx[1] * size - 0.5
            x2 = idx[2] * size + size + 2 * padding - 0.5
            y2 = idx[1] * size + size + 2 * padding - 0.5
            boxes = torch.stack([b, x1, y1, x2, y2], dim=1)
            return torchvision.ops.roi_align(x, boxes, size + 2 * padding, sampling_ratio=1)
        else:
            # Use gather. Crops out patches pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size, padding)
            pat = torch.gather(x.view(-1), 0, idx_pix.view(-1))
            pat = pat.view(-1, x.size(1), size + 2 * padding, size + 2 * padding)
            return pat

    def replace_patch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Replaces patches back into image given index.

        Inputs:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.

        Output:
            image: (B, C, H, W), where patches at idx locations are replaced with y.
        """
        xB, xC, xH, xW = x.shape
        yB, yC, yH, yW = y.shape
        if self.patch_replace_method == 'scatter_nd':
            # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
            x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
            x[idx[0], idx[1], idx[2]] = y
            x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
            return x
        else:
            # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
            return x.view(-1).scatter_(0, idx_pix.view(-1), y.view(-1)).view(x.shape)

    def compute_pixel_indices(self,
                              x: torch.Tensor,
                              idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                              size: int,
                              padding: int):
        """
        Compute selected pixel indices in the tensor.
        Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
        Input:
            x: image: (B, C, H, W)
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
                 the element are indices pointing to the input x.view(-1).
        """
        B, C, H, W = x.shape
        S, P = size, padding
        O = S + 2 * P
        b, y, x = idx
        n = b.size(0)
        c = torch.arange(C)
        o = torch.arange(O)
        idx_pat = (c * H * W).view(C, 1, 1).expand([C, O, O]) + (o * W).view(1, O, 1).expand([C, O, O]) + o.view(1, 1,
                                                                                                                 O).expand(
            [C, O, O])
        idx_loc = b * W * H + y * W * S + x * S
        idx_pix = idx_loc.view(-1, 1, 1, 1).expand([n, C, O, O]) + idx_pat.view(1, C, O, O).expand([n, C, O, O])
        return idx_pix


def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
                                                        activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method
    additionally returns the feature maps at all resolutions for decoder's use.
    """

    def __init__(self, in_channels, norm_layer=None):
        super().__init__()

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

        # Remove last block
        self.features = self.features[:-1]

        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)

        # Delete classifier
        del self.classifier

    def forward(self, x):
        x0 = x  # 1/1
        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x  # 1/2
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x  # 1/4
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x  # 1/8
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0


class Decoder(nn.Module):

    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        return x


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ResNetEncoder(ResNet):
    layers = {
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }

    def __init__(self, in_channels, variant='resnet101', norm_layer=None):
        super().__init__(
            block=Bottleneck,
            layers=self.layers[variant],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=norm_layer)

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        # Delete fully-connected layer
        del self.avgpool
        del self.fc

    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0


class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """

    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x

    def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.

        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}

        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features


class MattingBase(Base):

    def __init__(self, backbone: str):
        super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))

    def forward(self, src, bgr):
        x = torch.cat([src, bgr], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = x[:, 0:1].clamp_(0., 1.)
        fgr = x[:, 1:4].add(src).clamp_(0., 1.)
        err = x[:, 4:5].clamp_(0., 1.)
        hid = x[:, 5:].relu_()
        return pha, fgr, err, hid


class MattingRefine(MattingBase):

    def __init__(self,
                 backbone: str,
                 backbone_scale: float = 1 / 4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.1,
                 refine_kernel_size: int = 3,
                 refine_prevent_oversampling: bool = True,
                 refine_patch_crop_method: str = 'unfold',
                 refine_patch_replace_method: str = 'scatter_nd'):
        assert backbone_scale <= 1 / 2, 'backbone_scale should not be greater than 1/2'
        super().__init__(backbone)
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_mode,
                               refine_sample_pixels,
                               refine_threshold,
                               refine_kernel_size,
                               refine_prevent_oversampling,
                               refine_patch_crop_method,
                               refine_patch_replace_method)

    def forward(self, src, bgr):
        assert src.size() == bgr.size(), 'src and bgr must have the same shape'
        assert src.size(2) // 4 * 4 == src.size(2) and src.size(3) // 4 * 4 == src.size(3), \
            'src and bgr must have width and height that are divisible by 4'

        # Downsample src and bgr for backbone
        src_sm = F.interpolate(src,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        bgr_sm = F.interpolate(bgr,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)

        # Base
        x = torch.cat([src_sm, bgr_sm], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha_sm = x[:, 0:1].clamp_(0., 1.)
        fgr_sm = x[:, 1:4]
        err_sm = x[:, 4:5].clamp_(0., 1.)
        hid_sm = x[:, 5:].relu_()

        # Refiner
        pha, fgr, ref_sm = self.refiner(src, bgr, pha_sm, fgr_sm, err_sm, hid_sm)

        # Clamp outputs
        pha = pha.clamp_(0., 1.)
        fgr = fgr.add_(src).clamp_(0., 1.)
        fgr_sm = src_sm.add_(fgr_sm).clamp_(0., 1.)

        return pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm


class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
        if self.transforms:
            img = self.transforms(img)

        return img


class NewImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = [root]
        print(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transforms:
            img = self.transforms(img)

        return img


class ZipDataset(Dataset):
    def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False):
        self.datasets = datasets
        self.transforms = transforms

        if assert_equal_length:
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        x = tuple(d[idx % len(d)] for d in self.datasets)
        print(x)
        if self.transforms:
            x = self.transforms(*x)
        return x


class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x


class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]


# --------------- Arguments ---------------

parser = argparse.ArgumentParser(description='hy-replace-background')

parser.add_argument('--model-type', type=str, required=False, choices=['mattingbase', 'mattingrefine'],
                    default='mattingrefine')
parser.add_argument('--model-backbone', type=str, required=False, choices=['resnet101', 'resnet50', 'mobilenetv2'],
                    default='resnet50')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=False, default='model/pytorch_resnet50.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('--output-dir', type=str, required=False, default='content/output')
parser.add_argument('--output-types', type=str, required=False, nargs='+',
                    choices=['com', 'pha', 'fgr', 'err', 'ref', 'new'],
                    default=['new'])
parser.add_argument('-y', action='store_true')


def handle(image_path: str, bgr_path: str, new_bg: str):
    parser.add_argument('--images-src', type=str, required=False, default=image_path)
    parser.add_argument('--images-bgr', type=str, required=False, default=bgr_path)
    args = parser.parse_args()

    assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
        'Only mattingbase and mattingrefine support err output'
    assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
        'Only mattingrefine support ref output'

    # --------------- Main ---------------

    device = torch.device(args.device)

    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    # Load images
    dataset = ZipDataset([
        NewImagesDataset(args.images_src),
        NewImagesDataset(args.images_bgr),
    ], assert_equal_length=True, transforms=PairCompose([
        HomographicAlignment() if args.preprocess_alignment else PairApply(nn.Identity()),
        PairApply(T.ToTensor())
    ]))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    # # Create output directory
    # if os.path.exists(args.output_dir):
    #     if args.y or input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         exit()

    for output_type in args.output_types:
        if os.path.exists(os.path.join(args.output_dir, output_type)) is False:
            os.makedirs(os.path.join(args.output_dir, output_type))

    # Worker function
    def writer(img, path):
        img = to_pil_image(img[0].cpu())
        img.save(path)

    # Worker function
    def writer_hy(img, new_bg, path):
        img = to_pil_image(img[0].cpu())
        img_size = img.size
        new_bg_img = Image.open(new_bg).convert('RGBA')
        new_bg_img.resize(img_size, Image.ANTIALIAS)
        out = Image.alpha_composite(new_bg_img, img)
        out.save(path)

    result_file_name = str(uuid.uuid4())

    # Conversion loop
    with torch.no_grad():
        for i, (src, bgr) in enumerate(tqdm(dataloader)):
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)

            if args.model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif args.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)

            pathname = dataset.datasets[0].filenames[i]
            pathname = os.path.relpath(pathname, args.images_src)
            pathname = os.path.splitext(pathname)[0]

            if 'new' in args.output_types:
                new = torch.cat([fgr * pha.ne(0), pha], dim=1)
                Thread(target=writer_hy,
                       args=(new, new_bg, os.path.join(args.output_dir, 'new', result_file_name + '.png'))).start()
            if 'com' in args.output_types:
                com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                Thread(target=writer, args=(com, os.path.join(args.output_dir, 'com', pathname + '.png'))).start()
            if 'pha' in args.output_types:
                Thread(target=writer, args=(pha, os.path.join(args.output_dir, 'pha', pathname + '.jpg'))).start()
            if 'fgr' in args.output_types:
                Thread(target=writer, args=(fgr, os.path.join(args.output_dir, 'fgr', pathname + '.jpg'))).start()
            if 'err' in args.output_types:
                err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
                Thread(target=writer, args=(err, os.path.join(args.output_dir, 'err', pathname + '.jpg'))).start()
            if 'ref' in args.output_types:
                ref = F.interpolate(ref, src.shape[2:], mode='nearest')
                Thread(target=writer, args=(ref, os.path.join(args.output_dir, 'ref', pathname + '.jpg'))).start()

    return os.path.join(args.output_dir, 'new', result_file_name + '.png')


if __name__ == '__main__':
    handle("data/img2.png", "data/bg.png", "data/newbg.jpg")
