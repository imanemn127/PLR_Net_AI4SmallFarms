#!/usr/bin/env python3
"""
tif_to_png.py  —  PLR-Net / AI4SmallFarms
==========================================
Convert a Sentinel-2 GeoTIFF patch (uint16, 3 bands RGB) to an 8-bit PNG
that can be used for quick visual checks or as input to inference scripts
that expect a standard image file.

The uint16 values (range 0-10000) are clipped at a configurable ceiling
and then linearly rescaled to [0, 255].  The default ceiling (3000) gives
good contrast for Sentinel-2 surface reflectance over agricultural land.

Usage:
  # Convert one file
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/tif_to_png.py \
      data/train/patches_256/0_vietnam_00000_00000.tif \
      data/sample_test.png

  # Change the reflectance ceiling (e.g. 2500 for darker scenes)
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/tif_to_png.py \
      data/train/patches_256/0_vietnam_00000_00000.tif \
      data/sample_test.png --ceil 2500
"""

import argparse

import numpy as np
import rasterio
from PIL import Image


def tif_to_png(src_path: str, dst_path: str, ceil: int = 3000) -> None:
    with rasterio.open(src_path) as src:
        # Bands are stored as (B4=R, B3=G, B2=B) at indices 1,2,3
        data = src.read([1, 2, 3])   # shape (3, H, W), dtype uint16
        print(f"  Input  : {src_path}")
        print(f"  Shape  : {data.shape}  dtype={data.dtype}")
        print(f"  Range  : min={data.min()}  max={data.max()}")

    # Clip to reflectance ceiling, then scale to [0, 255]
    data_clipped = np.clip(data, 0, ceil).astype(np.float32) / ceil * 255.0
    data_uint8   = data_clipped.astype(np.uint8)

    # Rearrange from (3, H, W) to (H, W, 3) expected by PIL
    img = Image.fromarray(data_uint8.transpose(1, 2, 0), mode="RGB")
    img.save(dst_path)
    print(f"  Output : {dst_path}  size={img.size[0]}x{img.size[1]} px")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Sentinel-2 GeoTIFF patch to an 8-bit PNG"
    )
    parser.add_argument("src",  help="Input GeoTIFF (.tif)")
    parser.add_argument("dst",  help="Output PNG file (.png)")
    parser.add_argument(
        "--ceil", type=int, default=3000,
        help="Reflectance ceiling for uint16 → uint8 scaling (default: 3000)"
    )
    args = parser.parse_args()
    tif_to_png(args.src, args.dst, args.ceil)


if __name__ == "__main__":
    main()
