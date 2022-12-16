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

import sys
import os
import zipfile
from argparse import ArgumentParser
from threading import Lock
import time
from tqdm.auto import tqdm
from PIL import Image

from utils import seed_all
from folder_parser import parse_folder
from data_processor import DataProcessor


def main(args):
    time_start = time.time()
    scale_algorithms = dict()
    for a in Image.Resampling:
        scale_algorithms[str(a).split('.')[1].lower()] = a
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
            choices = zip_algorithms.keys(),
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
            '--fixed_size',
            type = int,
            nargs = 2,
            help = "use a fixed image size instead of variable size"
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
            '--no_parallel',
            action = 'store_true',
            help = "disable parallel processing"
    )
    parser.add_argument(
            '--no_lazy',
            action = 'store_true',
            help = "disable lazy evaluation Lazy can get more performance as parallel synchronization points are not fixed"
    )
    parser.add_argument(
            '--delete_original',
            action = 'store_true',
            help = 'delete original input data once it has been processed (only if output path is set)'
    )
    args = parser.parse_args(args)
    if args.no_parallel:
        args.no_lazy = True
    dataset = parse_folder(args.input_path, quiet = args.quiet)
    if args.output_path is not None:
        progress_file = 'progress.txt'
        if args.resume_from is not None:
            progress_file = args.resume_from
            with open(progress_file, 'r') as f:
                done = f.read().splitlines()
            dataset = { k: {**v} for k,v in dataset.items() if k not in done }
        plock = Lock()
        def resume_callback(entry):
            with plock:
                with open(progress_file, 'a') as f:
                    print(entry['id'], file = f)
            if args.delete_original:
                if not args.no_save_image:
                    os.remove(entry['input_image'])
                if not args.no_save_text:
                    os.remove(entry['input_text'])
    else:
        resume_callback = lambda x: x
    if not args.no_lazy and not args.quiet:
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
            image_size = args.fixed_size,
            max_image_size = args.max,
            min_image_size = args.min,
            alpha_color = tuple(args.alpha),
            scale_algorithm = scale_algorithms[args.scale_algorithm],
            clip_layer = args.clip_layer,
            parallel = not args.no_parallel
    )
    for e in data_processor(
            dataset = dataset,
            resume_from = args.resume_from,
            batch_size = args.batch_size,
            encode = args.encode,
            quiet = args.quiet,
            progress = args.no_lazy,
            lazy = not args.no_lazy
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
    main(sys.argv[1:])

