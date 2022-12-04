# Copyright (C) 2022  Lopho <contact@lopho.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from itertools import zip_longest
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from concurrent.futures import CancelledError


def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

def batched(iterable, n):
    args = [ iter(iterable) ] * n
    return ( [ x for x in b if x is not None ] for b in zip_longest(*args, fillvalue = None) )

def seed_all(
        seed = 0,
        deterministic = True,
        warn_only = False,
        benchmark = False,
        keep_cublas_env = False
):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not keep_cublas_env:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only = warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
        if not keep_cublas_env and 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']
        torch.use_deterministic_algorithms(False)
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def load_vae_and_clip(model_name, device = 'cpu'):
    from diffusers.models.vae import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel
    from transformers.models.clip.tokenization_clip import CLIPTokenizer
    vae = AutoencoderKL.from_pretrained(model_name, subfolder = 'vae').eval().to(device)
    clip = CLIPTextModel.from_pretrained(model_name, subfolder = 'text_encoder').eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder = 'tokenizer')
    return vae, clip, tokenizer


class FutureMock:
    def __init__(self, data):
        self.data = data

    def result(self, *_):
        return self.data

    def done(self):
        return True

    def exception(self, *_):
        return None

    def add_done_callback(self, f):
        f(self.data)


class CatchPool:
    def __init__(self, pool, error_callback):
        self.pool = pool
        self.error_callback = error_callback

    def submit(self, *args, **kwargs):
        return CatchFuture(self, self.pool.submit(*args, **kwargs))

    def shutdown(self, wait = True, cancel_futures = False):
        return self.pool.shutdown(wait = wait, cancel_futures = cancel_futures)


class CatchFuture:
    def __init__(self, pool, future):
        self.pool = pool
        self.future = future

    def result(self, timeout = None):
        try:
            return self.future.result(timeout = timeout)
        except Exception as e:
            self.pool.error_callback()
            if not isinstance(e, CancelledError):
                import traceback
                print(traceback.format_exc())

    def add_done_callback(self, callback):
        def _callback(x):
            try:
                e = x.exception()
                if e is not None:
                    self.pool.error_callback()
                else:
                    callback(x.result())
            except CancelledError:
                pass
        self.future.add_done_callback(_callback)


class LazyDict(dict):
    def __init__(self, index, keys = [], index_keys = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = index
        self._keys = keys
        self._index_keys = index_keys

    def __getitem__(self, key):
        if key in self._keys:
            return super().__getitem__(key).result()
        elif key in self._index_keys:
            k = self._index_keys[key]
            if k is None:
                return super().__getitem__(key).result()[self._index]
            else:
                return super().__getitem__(key).result()[k][self._index]
        else:
            return super().__getitem__(key)

    def get(self, key, default = None):
        if key in self:
            return self.__getitem__(key)
        else:
            return default

    def evaluate(self):
        for k in self._keys:
            self.get(k, None)
        for k in self._index_keys:
            self.get(k, None)

