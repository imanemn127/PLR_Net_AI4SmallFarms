# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys
from urllib.parse import urlparse

import torch.hub

from PLRNet.utils.comm import is_main_process
from PLRNet.utils.comm import synchronize


def cache_url(url, model_dir=None, progress=True):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        filename = parts.path.replace("/", "_")
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        torch.hub.download_url_to_file(url, cached_file, progress=progress)
    synchronize()
    return cached_file
