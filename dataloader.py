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
from functools import reduce
from io import BytesIO
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import signal
from tqdm.auto import tqdm
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(torch.float32).to(device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

def batcher(iterable, n):
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

def load_models(model_name, device = 'cpu'):
    from diffusers.models.vae import AutoencoderKL
    from transformers.models.clip.modeling_clip import CLIPTextModel
    from transformers.models.clip.tokenization_clip import CLIPTokenizer
    vae = AutoencoderKL.from_pretrained(model_name, subfolder = 'vae').eval().to(device)
    clip = CLIPTextModel.from_pretrained(model_name, subfolder = 'text_encoder').eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder = 'tokenizer')
    return vae, clip, tokenizer

def read_data_folder_encoded(path, quiet = False, progress = True):
    assert os.path.isdir(path)
    files = os.listdir(path)
    assert len(files) > 0
    if not quiet:
        tqdm.write(f"Reading data from {path}")
    entries = dict()
    for p in tqdm(
            files,
            disable = quiet or not progress,
            smoothing = 0.01,
            desc = 'Read',
            colour = '#cc9911',
            dynamic_ncols = True
    ):
        id, e = os.path.splitext(p)
        abspath = os.path.abspath(os.path.join(path, p))
        if not os.path.isfile(abspath):
            continue
        if e == '.zip':
            with zipfile.ZipFile(abspath, 'r') as zf:
                f = zf.open(id + '.pt')
                size = torch.load(f)['size']
        else:
            size = torch.load(abspath)['size']
        entries[id] = { 'id': id, 'data': abspath, 'image_size': size }
    return entries

def read_data_folder(path, quiet = False, progress = True):
    assert os.path.isdir(path), f"not a directory: {path}"
    files = os.listdir(path)
    assert len(files) > 0
    files_dict = dict()
    if not quiet:
        tqdm.write(f"Reading data from {path}")
    for p in tqdm(
            files,
            desc = 'Read',
            disable = quiet or not progress,
            smoothing = 0.01,
            colour = '#cc9911',
            dynamic_ncols = True
    ):
        abspath = os.path.abspath(os.path.join(path, p))
        if not os.path.isfile(abspath):
            continue
        id, e = os.path.splitext(p)
        if id not in files_dict:
            files_dict[id] = { 'id': id }
        if e == '.txt':
            files_dict[id]['text'] = abspath
        else:
            try:
                if e == '.zip':
                    with zipfile.ZipFile(abspath, 'r') as zf:
                        pimg = [
                                f for f in zf.filelist
                                if not f.filename.endswith('.pt') and not f.filename.endswith('.txt')
                        ][0]
                        ptxt = id + '.txt'
                        f = zf.open(pimg)
                        size = Image.open(f).size
                        files_dict[id]['image_size'] = size
                        files_dict[id]['image'] = pimg
                        files_dict[id]['text'] = ptxt
                        files_dict[id]['zip'] = abspath
                else:
                    size = Image.open(abspath).size
                    files_dict[id]['image_size'] = size
                    files_dict[id]['image'] = abspath
                    files_dict[id]['zip'] = None
            except Exception as e:
                tqdm.write(f"Skipping invalid file: {abspath}")
                continue
    entries = {}
    for id in files_dict:
        entry = files_dict[id]
        if 'text' in entry and 'image' in entry:
            entries[id] = entry
    return entries


##################################################################################################


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


class DataProcessor:
    def __init__(self,
            model_name,
            device = 'cpu',
            hf_online = True,
            max_image_size = 512,
            min_image_size = 256,
            alpha_color = (128, 128, 128),
            scale_algorithm = Image.Resampling.LANCZOS,
            clip_layer = 'last', # 'all', 'last', 'pooled', int(hidden layer index)
            parallel = True,
    ):
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.alpha_color = alpha_color
        self.scale_algorithm = scale_algorithm
        self.hf_online = hf_online
        self._model_name = model_name
        self.device = device
        if clip_layer.isdecimal():
            self.clip_layer = int(clip_layer)
        else:
            self.clip_layer = clip_layer
        self.vae, self.clip, self.tokenizer = None, None, None
        self.parallel = parallel
        if parallel:
            self.pool = ProcessPoolExecutor(initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_DFL))
            self.pool_thread = ThreadPoolExecutor()

    def _sigint_handler(self, *_):
        self.pool_thread.shutdown(wait = False, cancel_futures = True)
        self.pool.shutdown(wait = False, cancel_futures = True)
        raise KeyboardInterrupt

    def wait_for_done(self):
        if self.parallel:
            signal.signal(signal.SIGINT, self._sigint_handler)
            self.pool.shutdown(wait = True)
            self.pool_thread.shutdown(wait = True)
            self.pool_thread = ThreadPoolExecutor()
            self.pool = ProcessPoolExecutor(initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_DFL))

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, val):
        if val != self._model_name:
            self._model_name = val
            if self.vae is not None:
                self.load_model(False)
                self.load_model(True)

    def load_models(self, load = True):
        if load and self.vae is None and self._model_name is not None:
            self.vae, self.clip, self.tokenizer = load_models(
                    self._model_name,
                    device = self.device
            )
        else:
            self.vae, self.clip, self.tokenizer = None, None, None

    @staticmethod
    def _process_text(entry):
        if entry['zip'] is not None:
            with zipfile.ZipFile(entry['zip']) as zf:
                return zf.open(entry['text']).read().decode()
        else:
            with open(entry['text'], 'r') as f:
                return f.read()

    def process_text(self, batch):
        if not self.parallel:
            r = []
            for e in batch:
                x = self._process_text(e)
                r.append(FutureMock(x))
            return r
        r = []
        for e in batch:
            r.append(self.pool.submit(self._process_text, e))
        return r

    @staticmethod
    def _encode_text(batch, clip, tokenizer, device, clip_layer):
        with torch.inference_mode():
            x = tokenizer(
                    [ text.result() for text in batch ],
                    truncation = True,
                    return_overflowing_tokens = False,
                    padding = 'max_length',
                    return_tensors = 'pt'
            ).input_ids.to(device)
            y = clip(
                    input_ids = x,
                    output_hidden_states = isinstance(clip_layer, int) or clip_layer == 'all'
            )
            if clip_layer == 'last':
                encoded_texts = y.last_hidden_state
            elif clip_layer == 'pooled':
                encoded_texts = y.pooler_output[:, None, :]
            elif clip_layer == 'all':
                encoded_texts = y.hidden_states
            elif isinstance(clip_layer, int):
                encoded_texts = y.hidden_states[clip_layer]
        return encoded_texts

    def encode_text(self, batch):
        if not self.parallel:
            x = self._encode_text(batch, self.clip, self.tokenizer, self.device, self.clip_layer)
            return FutureMock(x)
        return self.pool_thread.submit(self._encode_text, batch, self.clip, self.tokenizer, self.device, self.clip_layer)

    @staticmethod
    def _fit_image_size_to_64(w, h, max_image_size, min_image_size):
        box = (0, 0, w, h)
        max_area = max_image_size ** 2
        scale = (max_area / (w * h)) ** 0.5
        w2 = round((w * scale) / 64) * 64
        h2 = round((h * scale) / 64) * 64
        if w2*h2 > max_area:
            w = int((w * scale) / 64) * 64
            h = int((h * scale) / 64) * 64
        else:
            w = w2
            h = h2
        # TODO option to squish or crop to fit min size
        # right now: squish
        # only happens at "insane" ratios, like 1:8+
        if w < min_image_size:
            w = min_image_size
            h = max_area // w
        elif h < min_image_size:
            h = min_image_size
            w = max_area // h
        return w, h, box

    def fit_image_size(self, w, h):
        return self._fit_image_size_to_64(w, h, self.max_image_size, self.min_image_size)

    @staticmethod
    def _process_image(entry, max_image_size, min_image_size, alpha_color, scale_algorithm):
        if entry['zip'] is not None:
            with zipfile.ZipFile(entry['zip']) as zf:
                image = Image.open(zf.open(entry['image']))
        else:
            image = Image.open(entry['image'])
        if image.mode == 'RGBA':
            bg = Image.new('RGBA', image.size, alpha_color)
            image = Image.alpha_composite(bg, image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        w, h, _ = DataProcessor._fit_image_size_to_64(
                *image.size,
                max_image_size,
                min_image_size
        )
        if image.size != (w, h):
            image = image.resize((w, h), resample = scale_algorithm)
        return image

    def process_image(self, batch):
        if not self.parallel:
            r = []
            for e in batch:
                x = self._process_image(e, self.max_image_size, self.min_image_size, self.alpha_color, self.scale_algorithm)
                r.append(FutureMock(x))
            return r
        r = []
        for e in batch:
            r.append(self.pool.submit(self._process_image, e, self.max_image_size, self.min_image_size, self.alpha_color, self.scale_algorithm))
        return r

    @staticmethod
    def _encode_image(batch, vae, device):
        with torch.inference_mode():
            batch = torch.stack([ pil_to_torch(image.result(), device) for image in batch ])
            latents = vae.encode(batch).latent_dist
        return { 'mean': latents.mean, 'std': latents.std }

    def encode_image(self, batch):
        if not self.parallel:
            x = self._encode_image(batch, self.max_image_size, self.min_image_size, self.alpha_color, self.scale_algorithm)
            return FutureMock(x)
        return self.pool_thread.submit(self._encode_image, batch, self.vae, self.device)

    def process_batch(self, size, batch, encode):
        scaled_images = self.process_image(batch)
        texts = self.process_text(batch)
        if encode:
            encoded_texts = self.encode_text(texts)
            latents = self.encode_image(scaled_images)
            results = []
            for i, e in enumerate(batch):
                r = LazyDict(
                        index = i,
                        keys = [ 'text', 'image' ],
                        index_keys = { 'latent': 'mean', 'latent_std': 'std', 'encoded_text': None }
                )
                r['id'] = e['id']
                r['encoded_text'] = encoded_texts
                r['latent'] = latents
                r['latent_std'] = latents
                r['text'] = texts[i]
                r['image'] = scaled_images[i]
                r['size'] = size
                results.append(r)
        else:
            results = []
            for i, e in enumerate(batch):
                r = LazyDict(index = i, keys = [ 'text', 'image' ])
                r['id'] = e['id']
                r['encoded_text'] = None
                r['latent'] = None
                r['latent_std'] = None
                r['text'] = texts[i]
                r['image'] = scaled_images[i]
                r['size'] = size
                results.append(r)
        return results

    @staticmethod
    def save_entry(
            entry,
            out_dir,
            save_image = True,
            save_text = True,
            save_encoded = True,
            make_zip = False,
            zip_algorithm = zipfile.ZIP_DEFLATED,
            image_format = 'png',
            image_quality = 100,
            image_compress = False
    ):
        files = []
        out_path = os.path.join(out_dir, entry['id'])
        if save_encoded and entry.get('latent', None) is not None:
            path = out_path + '.pt'
            buffer = BytesIO()
            torch.save({
                    'id': entry['id'],
                    'size': entry['size'],
                    'latent': entry['latent'].clone().cpu(),
                    'latent_std': entry['latent_std'].clone().cpu(),
                    'encoded_text': entry['encoded_text'].clone().cpu()
            }, buffer)
            files.append((path, buffer))
        if save_image and entry.get('image', None) is not None:
            path = out_path + '.' + image_format
            buffer = BytesIO()
            entry['image'].save(
                    buffer,
                    format = image_format,
                    optimize = image_compress,
                    quality = image_quality,
                    lossless = image_compress
            )
            files.append((path, buffer))
        if save_text and entry.get('text', None) is not None:
            path = out_path + '.txt'
            buffer = BytesIO()
            buffer.write(entry['text'].encode())
            files.append((path, buffer))
        if make_zip:
            with zipfile.ZipFile(
                    out_path + '.zip',
                    'w',
                    compression = zip_algorithm,
                    compresslevel = 9
            ) as zf:
                for f,d in files:
                    d.flush()
                    d.seek(0)
                    zf.writestr(os.path.basename(f), d.read())
        else:
            for f,d in files:
                d.flush()
                d.seek(0)
                with open(f, 'wb') as f:
                    f.write(d.read())
        return entry['id']

    def save_entries(self,
            entries,
            path,
            save_image = True,
            save_text = True,
            save_encoded = True,
            make_zip = False,
            zip_algorithm = zipfile.ZIP_DEFLATED,
            image_format = 'png',
            image_quality = 100,
            image_compress = False,
            callback = None
    ):
        os.makedirs(path, exist_ok = True)
        if not self.parallel:
            r = []
            for e in entries:
                x = self.save_entry(e, path, save_image, save_text, save_encoded, make_zip, zip_algorithm, image_format, image_quality, image_compress)
                r.append(FutureMock(x))
            return r
        r = []
        for e in entries:
            future = self.pool_thread.submit(self.save_entry, e, path, save_image, save_text, save_encoded, make_zip, zip_algorithm, image_format, image_quality, image_compress)
            if callback is not None:
                future.add_done_callback(callback)
            r.append(future)
        return r

    def __call__(self,
            dataset,
            processed = False,
            resume_from = None,
            batch_size = 8,
            encode = False,
            quiet = False,
            progress = True,
            lazy = False
    ):
        if not processed:
            return self.process_dataset(
                    dataset,
                    resume_from = resume_from,
                    batch_size = batch_size,
                    encode = encode,
                    quiet = quiet,
                    progress = progress,
                    lazy = lazy
            )
        else:
            return self.dataset(dataset, batch_size = batch_size, quiet = quiet, progress = progress)

    def dataset(self, dataset, batch_size = 8, quiet = False):
        if isinstance(dataset, str):
            dataset = read_data_folder_encoded(dataset, quiet)

    def process_dataset(self,
            dataset,
            resume_from = None,
            batch_size = 8,
            encode = False,
            quiet = False,
            progress = True,
            lazy = False
    ):
        if not self.parallel:
            lazy = False
        if isinstance(dataset, str):
            dataset = read_data_folder(dataset, quiet, progress)
        if resume_from is not None:
            if not quiet:
                tqdm.write(f"Resuming from {resume_from}")
            with open(resume_from, 'r') as f:
                done = f.read().splitlines()
            dataset = { k: {**v} for k,v in dataset.items() if k not in done }
        buckets = dict()
        for e in dataset.values():
            w, h, _ = self.fit_image_size(*e['image_size'])
            if (w, h) not in buckets:
                buckets[(w, h)] = []
            buckets[(w, h)].append(e)
        batched_buckets = [
                (k, [ x for x in batcher(y, batch_size) ]) for k, y in buckets.items()
        ]
        if encode:
            if not quiet:
                tqdm.write("Loading models ...")
            self.load_models()
        if not quiet:
            tqdm.write(f"Samples: {len(dataset)}")
            tqdm.write(f"Buckets: {len(buckets)}")
            tqdm.write(f"Batch size: {batch_size}")
            tqdm.write(f"\n    Bucket   | Batches | Samples")
            tqdm.write(   "---w--+---h--|---------|---------")
        if not lazy and self.parallel:
            signal.signal(signal.SIGINT, self._sigint_handler)
        pbar = tqdm(
                total = len(dataset),
                desc = 'Samples',
                smoothing = 0.01,
                disable = quiet or not progress,
                dynamic_ncols = True,
                colour = '#cc9911'
        )
        for size, bucket in batched_buckets:
            if not quiet:
                num_samples = reduce(lambda x,y: x + len(y), bucket, 0)
                tqdm.write(f" {size[0]:>4} | {size[1]:>4} | {len(bucket):>7} | {num_samples:>7}")
            for batch in bucket:
                r = self.process_batch(size, batch, encode)
                if not lazy:
                    for e in r:
                        e.evaluate()
                yield r
                pbar.update(len(batch))


##################################################################################################


def main(args):
    import time
    time_start = time.time()
    from argparse import ArgumentParser
    from threading import Lock
    scale_algorithms = {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'lanczos': Image.Resampling.LANCZOS
    }
    zip_algorithms = {
        'store': zipfile.ZIP_STORED,
        'deflate': zipfile.ZIP_DEFLATED,
        'lzma': zipfile.ZIP_LZMA,
        'bzip2': zipfile.ZIP_BZIP2
    }
    parser = ArgumentParser(
        epilog = "Copyright (C) 2022  Lopho <contact@lopho.org> | Licensed under the AGPLv3 <https://www.gnu.org/licenses/>",
    )
    def error(msg):
        parser.print_help()
        print(f'{sys.argv[0]}: error: {msg}')
        parser.exit(1)
    parser.error = error
    parser.add_argument(
            '-i',
            '--input_path',
            type = str,
            required = True,
            help = "path to a folder containing images and text"
    )
    parser.add_argument(
            '-o', '--output_path',
            type = str,
            help = "path to save processed data"
    )
    parser.add_argument(
            '--no_save_image',
            action = 'store_true',
            help = "don't save processed images"
    )
    parser.add_argument(
            '--no_save_text',
            action = 'store_true',
            help = "don't save processed text"
    )
    parser.add_argument(
            '--no_save_encoded',
            action = 'store_true',
            help = "don't save encoded images and text"
    )
    parser.add_argument(
            '--zip',
            action = 'store_true',
            help = "save data compressed as zip files"
    )
    parser.add_argument(
            '--zip_algorithm',
            choices = ['store', 'deflate', 'lzma', 'bzip2'],
            default = 'deflate',
            help = "compression algorithm used for zip files"
    )
    parser.add_argument(
            '--image_format',
            choices = ['png', 'jpeg', 'webp'],
            default = 'webp',
            help = "format to save images as"
    )
    parser.add_argument(
            '--image_quality',
            type = int,
            default = 100,
            help = "jpg, lossy webp: image quality | lossless webp: compression level"
    )
    parser.add_argument(
            '--image_compress',
            action = 'store_true',
            help = "webp, png: compress lossless"
    )
    parser.add_argument(
            '-m', '--model',
            type = str,
            default = 'stabilityai/stable-diffusion-2',
            help = "huggingface model used to encode images and text if encode = True"
    )
    parser.add_argument(
            '-e', '--encode',
            action = 'store_true',
            help = "encode images with VAE and text with CLIP"
    )
    parser.add_argument(
            '--clip_layer',
            type = str,
            default = 'last',
            help = "which CLIP hidden state to use: all, last, pooled, 0,1,2,3,4,5,... (hidden layer index)"
    )
    parser.add_argument(
            '-b', '--batch_size',
            type = int,
            default = os.cpu_count() * 2,
            help = "batch size, if not encoding, high values (like 512) get better throughput with little RAM overhead"
    )
    parser.add_argument(
            '-q', '--quiet',
            action = 'store_true',
            help = "suppress info messages"
    )
    parser.add_argument(
            '-s', '--scale_algorithm',
            choices = scale_algorithms.keys(),
            default = 'bilinear',
            help = "the scaling algorithm used to resize images"
    )
    parser.add_argument(
            '-d', '--device',
            type = str,
            default = 'cuda',
            help = "device used to run the VAE and CLIP for encoding"
    )
    parser.add_argument(
            '--min',
            type = int,
            default = 256,
            help = "minimum image side length, anything below will be stretched"
    )
    parser.add_argument(
            '--max',
            type = int,
            default = 768,
            help = "maximum total image size, MAX*MAX >= width*height"
    )
    parser.add_argument(
            '--alpha',
            type = int,
            nargs = 3,
            default = [ 128, 128, 128 ],
            help = "color used to replace alpha channel if present"
    )
    parser.add_argument(
            '--resume_from',
            type = str,
            help = "a file containing a list of file names without extension that will be skipped. processed data names will be appended."
    )
    parser.add_argument(
            '--not_parallel',
            action = 'store_true',
            help = "disable parallel processing"
    )
    parser.add_argument(
            '--lazy',
            action = 'store_true'
    )
    args = parser.parse_args(args)
    if args.not_parallel:
        args.lazy = False
    dataset = read_data_folder(args.input_path)
    if args.output_path is not None:
        progress_file = 'progress.txt'
        if args.resume_from is not None:
            progress_file = args.resume_from
            with open(progress_file, 'r') as f:
                done = f.read().splitlines()
            dataset = { k: {**v} for k,v in dataset.items() if k not in done }
        plock = Lock()
        def resume_callback(future):
            try:
                id = future.result()
            except:
                return
            with plock:
                with open(progress_file, 'a') as f:
                    print(id, file = f)
    else:
        resume_callback = lambda x: x
    if args.lazy:
        pbar = tqdm(total = len(dataset), desc = 'Samples', dynamic_ncols = True, smoothing = 0.01, colour = '#cc9911')
        plock = Lock()
        def pbar_progress(x):
            with plock:
                pbar.update(1)
            return x
        progress_callback = lambda x: resume_callback(pbar_progress(x))
    else:
        progress_callback = resume_callback
    seed_all(0)
    data_processor = DataProcessor(
            model_name = args.model,
            device = args.device,
            max_image_size = args.max,
            min_image_size = args.min,
            alpha_color = tuple(args.alpha),
            scale_algorithm = scale_algorithms[args.scale_algorithm],
            clip_layer = args.clip_layer,
            parallel = not args.not_parallel
    )
    for e in data_processor(
            dataset = dataset,
            resume_from = args.resume_from,
            batch_size = args.batch_size,
            encode = args.encode,
            quiet = args.quiet,
            progress = not args.lazy,
            lazy = args.lazy
    ):
        if args.output_path is not None:
            data_processor.save_entries(
                    e,
                    args.output_path,
                    save_image = not args.no_save_image,
                    save_text = not args.no_save_text,
                    save_encoded = not args.no_save_encoded,
                    make_zip = args.zip,
                    zip_algorithm = zip_algorithms[args.zip_algorithm],
                    image_format = args.image_format,
                    image_quality = args.image_quality,
                    image_compress = args.image_compress,
                    callback = progress_callback
            )
    data_processor.wait_for_done()
    if not args.quiet:
        tqdm.write(f"took {time.time() - time_start} seconds")

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

