# parallel_dataloader

## Examples
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

## Usage
```
usage: main.py [-h] -i INPUT_PATH [-o OUTPUT_PATH] [--no_save_image] [--no_save_text] [--no_save_encoded] [--zip]
               [--zip_algorithm {store,deflate,lzma,bzip2}] [--image_format {png,jpeg,webp}]
               [--image_quality IMAGE_QUALITY] [--image_compress] [-m MODEL] [-e] [--clip_layer CLIP_LAYER]
               [-b BATCH_SIZE] [-q] [-s {nearest,bilinear,lanczos}] [-d DEVICE] [--min MIN] [--max MAX]
               [--fixed_size FIXED_SIZE FIXED_SIZE] [--alpha ALPHA ALPHA ALPHA] [--resume_from RESUME_FROM]
               [--no_parallel] [--lazy]

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
                        batch size, if not encoding, high values (like 512) get better throughput with little RAM overhead
  -q, --quiet           suppress info messages
  -s {nearest,bilinear,lanczos}, --scale_algorithm {nearest,bilinear,lanczos}
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
  --lazy

Copyright (C) 2022 Lopho <contact@lopho.org> | Licensed under the AGPLv3 <https://www.gnu.org/licenses/>
```
