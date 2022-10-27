import tensorflow as tf
import jax
import flax
import numpy as np
import os
import json
import logging

logger = logging.getLogger(__name__)


def prefetch(dataset, n_prefetch):
    # Taken from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter


def get_data(data_dir, img_size, img_channels, num_classes, num_local_devices, batch_size, allow_resolution_mismatch=False, shuffle_buffer=1000):
    """

    Args:
        data_dir (str): Root directory of the dataset.
        img_size (int): Image size for training.
        img_channels (int): Number of image channels.
        num_classes (int): Number of classes, 0 for no classes.
        num_local_devices (int): Number of devices.
        batch_size (int): Batch size (per device).
        shuffle_buffer (int): Buffer used for shuffling the dataset.

    Returns:
        (tf.data.Dataset): Dataset.
    """

    def pre_process(serialized_example):
        feature = {'height': tf.io.FixedLenFeature([], tf.int64),
                   'width': tf.io.FixedLenFeature([], tf.int64),
                   'channels': tf.io.FixedLenFeature([], tf.int64),
                   'image': tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)}
        example = tf.io.parse_single_example(serialized_example, feature)

        height = tf.cast(example['height'], dtype=tf.int64)
        width = tf.cast(example['width'], dtype=tf.int64)
        channels = tf.cast(example['channels'], dtype=tf.int64)
        image = tf.io.decode_raw(example['image'], out_type=tf.uint8)
        image = tf.reshape(image, shape=[height, width, channels])
        image = tf.cast(image, dtype='float32')

        image = tf.image.resize(image, size=[img_size, img_size], method='bicubic', antialias=True)
        image = tf.image.random_flip_left_right(image)
        image = (image - 127.5) / 127.5
        label = tf.one_hot(example['label'], num_classes)
        return {'image': image, 'label': label}

    def shard(data):
        # Reshape images from [num_devices * batch_size, H, W, C] to [num_devices, batch_size, H, W, C]
        # because the first dimension will be mapped across devices using jax.pmap
        data['image'] = tf.reshape(data['image'], [num_local_devices, -1, img_size, img_size, img_channels])
        data['label'] = tf.reshape(data['label'], [num_local_devices, -1, num_classes])
        return data

    logger.info('Loading TFRecord...')
    with tf.io.gfile.GFile(os.path.join(data_dir, 'dataset_info.json'), 'r') as fin:
        dataset_info = json.load(fin)

    # check resolution mismatch
    if not allow_resolution_mismatch:
        if 'width' in dataset_info and 'height' in dataset_info:
            msg = 'Requested resolution {img_size} is different from input data {input_size}.' \
                  ' Provide the flag --allow_resolution_mismatch in order to allow this behaviour.'
            assert dataset_info['width'] == img_size, msg.format(img_size=img_size, input_size=dataset_info['width'])
            assert dataset_info['height'] == img_size, msg.format(img_size=img_size, input_size=dataset_info['height'])
        else:
            raise Exception(f'dataset_info.json does not contain keys "height" or "width". Ignore by providing --allow_resolution_mismatch.')

    for folder in [data_dir, os.path.join(data_dir, 'tfrecords')]:
        ckpt_files = tf.io.gfile.glob(os.path.join(folder, '*.tfrecords'))
        if len(ckpt_files) > 0:
            break
    else:
        raise FileNotFoundError(f'Could not find any tfrecord files in {data_dir}')
    ds = tf.data.TFRecordDataset(filenames=ckpt_files)
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.shuffle(min(dataset_info['num_examples'], shuffle_buffer))
    ds = ds.map(pre_process, tf.data.AUTOTUNE)
    ds = ds.batch(batch_size * num_local_devices, drop_remainder=True)  # uses per-worker batch size
    ds = ds.map(shard, tf.data.AUTOTUNE)
    ds = ds.prefetch(1)  # prefetches the next batch
    return ds, dataset_info
