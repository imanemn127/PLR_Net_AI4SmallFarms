"""
multi_task_head.py  —  PLR-Net / AI4SmallFarms
===============================================
Multi-task prediction head used by BsiNet_2.

With HEAD_SIZE = [[2]] the head produces a single (B, 2, H, W) output
that encodes the sub-pixel junction offset map (joff_x, joff_y).

The design mirrors the HiSup implementation:
  - one Conv3x3 branch per group in head_size
  - all branch outputs are concatenated along the channel dimension
"""

from torch import nn


class MultitaskHead(nn.Module):
    def __init__(self, in_channels: int, num_class: int, head_size: list):
        """
        Args:
            in_channels : number of input feature channels (64 in BsiNet_2)
            num_class   : total output channels = sum(sum(head_size, []))
            head_size   : list of lists, e.g. [[2]] means one group of 2 channels
        """
        super().__init__()

        # One Conv3x3 per group, each producing sum(group) output channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, sum(group), kernel_size=1),
            )
            for group in head_size
        ])

    def forward(self, x):
        # Concatenate all group outputs along the channel dimension
        import torch
        return torch.cat([head(x) for head in self.heads], dim=1)
