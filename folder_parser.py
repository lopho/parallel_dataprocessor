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
import zipfile
from tqdm.auto import tqdm
from PIL import Image
import torch


def parse_folder_encoded(path, quiet = False, progress = True):
    raise NotImplementedError
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

def parse_folder(path, quiet = False, progress = True, validate_image_content = False):
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
                        image = Image.open(f)
                        if validate_image_content:
                            image.load()
                        size = image.size
                        files_dict[id]['image_size'] = size
                        files_dict[id]['image'] = pimg
                        files_dict[id]['text'] = ptxt
                        files_dict[id]['zip'] = abspath
                else:
                    image = Image.open(abspath)
                    if validate_image_content:
                        image.load()
                    size = image.size
                    files_dict[id]['image_size'] = size
                    files_dict[id]['image'] = abspath
                    files_dict[id]['zip'] = None
            except Exception as e:
                tqdm.write(f"Skipping invalid file: {abspath} | {e}")
                continue
    entries = {}
    for id in files_dict:
        entry = files_dict[id]
        if 'text' in entry and 'image' in entry:
            entries[id] = entry
    return entries

