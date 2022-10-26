import jax
import flax
import numpy as np
from tqdm import tqdm
import requests
import os
import tempfile
import logging
import hashlib
import dill as pickle
import tensorflow as tf


logger = logging.getLogger(__name__)


def download(url, ckpt_dir=None):
    name = url[url.rfind('/') + 1: url.rfind('?')]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'flaxmodels')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        logger.info(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logger.error('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]


def prefetch(dataset, n_prefetch):
    # Taken from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter


class Cache():
    def __init__(self, args, f_path=None):
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8')).hexdigest()
        if f_path is None:
            # set to /tmp dir
            self.f_path = os.path.join('/', 'tmp', f'{md5}.pkl')
        else:
            self.f_path = os.path.join(f_path, f'{md5}.pkl')

    @property
    def exists(self):
        return tf.io.gfile.exists(self.f_path)

    def read(self):
        data = None
        if self.exists:
            logger.info(f'Reading from cache {self.f_path}')
            with tf.io.gfile.GFile(self.f_path, 'rb') as f:
                data = pickle.loads(f.read())
        return data

    def dump(self, data, add=True):
        if not isinstance(data, dict):
            raise ValueError('Cache data needs to be of type dict')
        elif len(data) == 0:
            logger.warning('Data to be dumped was empty. Not caching.')
            return
        # read old cache
        old_data = None
        if add:
            old_data = self.read()
        if old_data is not None:
            data = {**old_data, **data}
        logger.info(f'Writing to cache location {self.f_path}')
        with tf.io.gfile.GFile(self.f_path, "wb") as f:
            f.write(pickle.dumps(data))
