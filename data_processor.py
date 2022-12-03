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
from functools import reduce
from io import BytesIO
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import signal
from tqdm.auto import tqdm
from PIL import Image
import torch

from utils import LazyDict, FutureMock, batched, pil_to_torch, load_vae_and_clip
from folder_parser import parse_folder_encoded, parse_folder


class DataProcessor:
    def __init__(self,
            model_name,
            device = 'cpu',
            hf_online = True,
            image_size = None,
            max_image_size = 512,
            min_image_size = 256,
            alpha_color = (128, 128, 128),
            scale_algorithm = Image.Resampling.LANCZOS,
            clip_layer = 'last', # 'all', 'last', 'pooled', int(hidden layer index)
            parallel = True,
    ):
        if image_size is not None:
            if isinstance(image_size, (tuple, list)):
                assert (image_size[0] // 64) == (image_size[0] / 64), 'fixed image size is not multiple of 64'
                assert (image_size[1] // 64) == (image_size[1] / 64), 'fixed image size is not multiple of 64'
            else:
                assert (image_size // 64) == (image_size / 64), 'fixed image size is not multiple of 64'
                image_size = (image_size, image_size)
        self.image_size = image_size
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
            self.vae, self.clip, self.tokenizer = load_vae_and_clip(
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
    def _fit_image_size_to_64(w, h, image_size, max_image_size, min_image_size):
        box = (0, 0, w, h)
        if image_size is not None:
            return *image_size, box
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
        return self._fit_image_size_to_64(w, h, self.image_size, self.max_image_size, self.min_image_size)

    @staticmethod
    def _process_image(entry, image_size, max_image_size, min_image_size, alpha_color, scale_algorithm):
        if entry['zip'] is not None:
            with zipfile.ZipFile(entry['zip']) as zf:
                image = Image.open(zf.open(entry['image']))
                image.load()
        else:
            image = Image.open(entry['image'])
        if image.mode == 'RGBA':
            bg = Image.new('RGBA', image.size, alpha_color)
            image = Image.alpha_composite(bg, image).convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        w, h, _ = DataProcessor._fit_image_size_to_64(
                *image.size,
                image_size,
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
                x = self._process_image(e, self.image_size, self.max_image_size, self.min_image_size, self.alpha_color, self.scale_algorithm)
                r.append(FutureMock(x))
            return r
        r = []
        for e in batch:
            r.append(self.pool.submit(self._process_image, e, self.image_size, self.max_image_size, self.min_image_size, self.alpha_color, self.scale_algorithm))
        return r

    @staticmethod
    def _encode_image(batch, vae, device):
        with torch.inference_mode():
            batch = torch.stack([ pil_to_torch(image.result(), device) for image in batch ])
            latents = vae.encode(batch).latent_dist
        return { 'mean': latents.mean, 'std': latents.std }

    def encode_image(self, batch):
        if not self.parallel:
            x = self._encode_image(batch, self.vae, self.device)
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
                    index_keys = { 'latent': 'mean', 'latent_std': 'std', 'encoded_text': None } if encode else {}
            )
            r['id'] = e['id']
            r['encoded_text'] = encoded_texts if encode else None
            r['latent'] = latents if encode else None
            r['latent_std'] = latents if encode else None
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
            dataset = parse_folder_encoded(dataset, quiet)

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
            dataset = parse_folder(dataset, quiet, progress)
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
                (k, [ x for x in batched(y, batch_size) ]) for k, y in buckets.items()
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

