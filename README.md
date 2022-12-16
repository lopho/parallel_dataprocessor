<!--
Copyright (C) 2022  Lopho <contact@lopho.org>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

# parallel_dataloader

## As a dataloader for training

### Examples
Resize images and create batched buckets.
> Note: the last batch of each bucket can be smaller than `batch_size`

> Note: if you call the DataProcessor with `lazy = True`, all batches will be returned almost immediatly. Batches are being processed in the background. When accessing data that is not done yet the call will block until the requested data is ready.
```py
from PIL import Image
from data_processor import DataProcessor
dp = DataProcessor(
        max_image_size = 512,
        min_image_size = 256,
        scale_algorithm = Image.Resampling.BILINEAR
)
for b in dp(dataset = '/path/to/dataset', batch_size = 16):
    # b is a list of dicts, with size of batch size
    # len(b) == batch_size
    # b = [
    #   {
    #       'id': filename without extension (String),
    #       'latent': None
    #       'latent_std': None,
    #       'encoded_text': None,
    #       'image': PIL.Image,
    #       'text': String,
    #   },
    #   {...},
    #   ...
    # ]
```
Resize and encode images with VAE, encode text with CLIP using penultimate layer
```py
from PIL import Image
from data_processor import DataProcessor
dp = DataProcessor(
        # for sd v2: model_name = 'stabilityai/stable-diffusion-2'
        model_name = 'CompVis/stable-diffusion-v1-4',
        encode = True,
        max_image_size = 512,
        min_image_size = 256,
        clip_layer = -2,
        scale_algorithm = Image.Resampling.LANCZOS
)
for b in dp(dataset = '/path/to/dataset', batch_size = 16):
    # b is a list of dicts, with size of batch size
    # len(b) == batch_size
    # b = [
    #   {
    #       'id': filename without extension (String),
    #       'latent': torch.Tensor
    #       'latent_std': torch.Tensor,
    #       'encoded_text': torch.Tensor,
    #       'image': PIL.Image,
    #       'text': String,
    #   },
    #   {...},
    #   ...
    # ]
```

## As a script for preprocessing
```sh
python main.py -i /input/folder -o /output/folder
```

### Examples
Resize images to multiple of 64 with a max area of 512x512, save as webp.\
Input folder needs images and texts. Texts with `.txt` extension and the same
name as the image name without extension.
```sh
python main.py -i /folder/with/txt_and_image -o /output/folder \
    --max 512 --image_format webp
```
Using a fixed image size of 768x384
```sh
python main.py -i /folder/with/txt_and_image -o /output/folder \
    --fixed_size 768 384 --image_format webp
```
Encode images and text using VAE and CLIP, save only encoded data as torch pickle, one file per sample.\
Using VAE and CLIP from stable diffusion v2.
```sh
python main.py -i /folder/with/txt_and_image -o /output/folder \
    --no_save_image --no_save_text --max 512 \
    --device cuda --encode \
    --model 'stabilityai/stable-diffusion-2'
```

### Usage
```
usage: main.py [-h] -i INPUT_PATH [-o OUTPUT_PATH] [--no_save_image] [--no_save_text] [--no_save_encoded] [--zip]
               [--zip_algorithm {store,deflate,lzma,bzip2}] [--image_format {png,jpeg,webp}]
               [--image_quality IMAGE_QUALITY] [--image_compress] [-m MODEL] [-e] [--clip_layer CLIP_LAYER]
               [-b BATCH_SIZE] [-q] [-s {nearest,box,bilinear,hamming,bicubic,lanczos}] [-d DEVICE] [--min MIN]
               [--max MAX] [--fixed_size FIXED_SIZE FIXED_SIZE] [--alpha ALPHA ALPHA ALPHA] [--resume_from RESUME_FROM]
               [--no_parallel] [--no_lazy] [--delete_original]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        path to a folder containing images and text
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to save processed data
  --no_save_image       don't save processed images
  --no_save_text        don't save processed text
  --no_save_encoded     don't save encoded images and text
  --zip                 save data compressed as zip files
  --zip_algorithm {store,deflate,lzma,bzip2}
                        compression algorithm used for zip files
  --image_format {png,jpeg,webp}
                        format to save images as
  --image_quality IMAGE_QUALITY
                        jpg, lossy webp: image quality | lossless webp: compression level
  --image_compress      webp, png: compress lossless
  -m MODEL, --model MODEL
                        huggingface model used to encode images and text if encode = True
  -e, --encode          encode images with VAE and text with CLIP
  --clip_layer CLIP_LAYER
                        which CLIP hidden state to use: all, last, pooled, 0,1,2,3,4,5,... (hidden layer index)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size, if not encoding, high values (like 512) get better throughput with little RAM
                        overhead
  -q, --quiet           suppress info messages
  -s {nearest,box,bilinear,hamming,bicubic,lanczos}, --scale_algorithm {nearest,box,bilinear,hamming,bicubic,lanczos}
                        the scaling algorithm used to resize images
  -d DEVICE, --device DEVICE
                        device used to run the VAE and CLIP for encoding
  --min MIN             minimum image side length, anything below will be stretched
  --max MAX             maximum total image size, MAX*MAX >= width*height
  --fixed_size FIXED_SIZE FIXED_SIZE
                        use a fixed image size instead of variable size
  --alpha ALPHA ALPHA ALPHA
                        color used to replace alpha channel if present
  --resume_from RESUME_FROM
                        a file containing a list of file names without extension that will be skipped. processed data
                        names will be appended.
  --no_parallel         disable parallel processing
  --no_lazy             disable lazy evaluation Lazy can get more performance as parallel synchronization points are
                        not fixed
  --delete_original     delete original input data once it has been processed (only if output path is set)

Copyright (C) 2022 Lopho <contact@lopho.org> | Licensed under the AGPLv3 <https://www.gnu.org/licenses/>
```
