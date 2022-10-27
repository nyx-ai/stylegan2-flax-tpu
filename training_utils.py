import jax
import re
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
import flax
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
from flax.core import frozen_dict
from flax import struct
import numpy as np
from PIL import Image
from typing import Any, Callable
from collections import defaultdict
import logging
from timeit import default_timer as timer
import subprocess

logger = logging.getLogger(__name__)


def sync_moving_stats(state):
    """
    Sync moving statistics across devices.

    Args:
        state (train_state.TrainState): Training state.

    Returns:
        (train_state.TrainState): Updated training state.
    """
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    return state.replace(moving_stats=cross_replica_mean(state.moving_stats))


def update_generator_ema(state_G, params_ema_G, config, ema_beta=None):
    """
    Update exponentially moving average of the generator weights.
    Moving stats and noise constants will be copied over.

    Args:
        state_G (train_state.TrainState): Generator state.
        params_ema_G (frozen_dict.FrozenDict): Parameters of the ema generator.
        config (Any): Config object.
        ema_beta (float): Beta parameter of the ema. If None, will be computed
                          from 'ema_nimg' and 'batch_size'.

    Returns:
        (frozen_dict.FrozenDict): Updates parameters of the ema generator.
    """
    def _update_ema(src, trg, beta):
        for name, src_child in src.items():
            if isinstance(src_child, DeviceArray):
                trg[name] = src[name] + ema_beta * (trg[name] - src[name])
            else:
                _update_ema(src_child, trg[name], beta)

    if ema_beta is None:
        ema_nimg = config.ema_kimg * 1000
        ema_beta = 0.5 ** (config.batch_size / max(ema_nimg, 1e-8))

    params_ema_G = params_ema_G.unfreeze()

    # Copy over moving stats
    params_ema_G['moving_stats']['mapping_network'] = state_G.moving_stats
    params_ema_G['noise_consts']['synthesis_network'] = state_G.noise_consts

    # Update exponentially moving average of the trainable parameters
    _update_ema(state_G.params['mapping'], params_ema_G['params']['mapping_network'], ema_beta)
    _update_ema(state_G.params['synthesis'], params_ema_G['params']['synthesis_network'], ema_beta)

    params_ema_G = frozen_dict.freeze(params_ema_G)
    return params_ema_G


class TrainStateG(train_state.TrainState):
    """
    Generator train state for a single Optax optimizer.

    Attributes:
        apply_mapping (Callable): Apply function of the Mapping Network.
        apply_synthesis (Callable): Apply function of the Synthesis Network.
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
        moving_stats (Any): Moving average of the latent W.
        noise_consts (Any): Noise constants from synthesis layers.
    """
    apply_mapping: Callable = struct.field(pytree_node=False)
    apply_synthesis: Callable = struct.field(pytree_node=False)
    dynamic_scale_main: dynamic_scale_lib.DynamicScale
    dynamic_scale_reg: dynamic_scale_lib.DynamicScale
    epoch: int
    moving_stats: Any = None
    noise_consts: Any = None


class TrainStateD(train_state.TrainState):
    """
    Discriminator train state for a single Optax optimizer.

    Attributes:
        dynamic_scale (dynamic_scale_lib.DynamicScale): Dynamic loss scaling for mixed precision gradients.
        epoch (int): Current epoch.
    """
    dynamic_scale_main: dynamic_scale_lib.DynamicScale
    dynamic_scale_reg: dynamic_scale_lib.DynamicScale
    epoch: int


def normalize_images(images, sample_normalization=True):
    if sample_normalization:
        minv = jnp.min(images)
        maxv = jnp.max(images)
        images = (images - minv)/(maxv - minv)
        images = jnp.clip(images * 255, 0, 255)
    else:
        images = 127.5 * images + 127.5
    return images


def get_grid_snapshot(images, grid_size=(None, None), grid_size_px=None, sample_normalization=True):
    assert len(grid_size) == 2, 'Grid size needs to be a tupe of (num_rows, num_cols)'
    # convert list to array of num_images x H x W x channels
    # images = jnp.array(images)
    # normalize images
    images = normalize_images(images, sample_normalization=sample_normalization)
    # check number of images
    num_images = len(images)
    if grid_size == (None, None) and grid_size_px is None:
        # just assume number of columns to be two
        grid_size = ((num_images // 2) + 1, 2)
    elif grid_size_px is not None:
        img_width = images[0].shape[1]
        num_rows = grid_size_px // img_width
        grid_size = (num_rows, num_rows)
    expected_num_images = grid_size[0] * grid_size[1]
    if num_images > expected_num_images:
        images = images[:expected_num_images]
    elif num_images < expected_num_images:
        # append white images
        for _ in range(expected_num_images - num_images):
            images = jnp.concatenate((images, jnp.ones((1,) + images[0].shape)), axis=0)
    assert len(images) == expected_num_images
    # stack into grid
    images = jnp.vstack([jnp.hstack(images[i*grid_size[0]:grid_size[0]*(i+1)]) for i in range(grid_size[1])])
    if images.shape[-1] == 1:
        images = jnp.repeat(images, 3, axis=-1)
    # convert to uint8
    images = np.uint8(images)
    return Image.fromarray(images)


def generate_random_labels(rng, shape, num_classes):
    rng, key = jax.random.split(rng)
    labels = jax.random.randint(key, shape, 0, num_classes)
    labels = jax.nn.one_hot(labels, num_classes)
    return labels


def count_params(model):
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda s: s.size, model))
    return sum(leaves)


def tree_shape(item):
    return jax.tree_util.tree_map(lambda c: c.shape, item)


def visualize_model(params, name=None):
    layers = []
    max_len = defaultdict(int)
    if name is not None:
        print(f'\n{name}:')
    for k, v in flax.traverse_util.flatten_dict(params).items():
        name = '.'.join(k)
        num_params = f'{v.size:,}'
        dtype = v.dtype.name
        shape = str(v.shape)
        item = dict(layer=name, parameters=num_params, dtype=dtype, shape=shape)
        for k, v in item.items():
            max_len[k] = max(max(len(str(v)), max_len[k]), len(k))
        layers.append(item)
    header = [k.ljust(v) for k, v in max_len.items()]
    print(' | '.join(header))
    print(' | '.join([len(h)*'-' for h in header]))
    for _l in layers:
        _l = {k: v.ljust(max_len[k]) for k, v in _l.items()}
        print(' | '.join(_l.values()))
    total_params = count_params(params)
    print(f'Total: {total_params:,}')


def partial_load_from(init_to, init_from, exclude_startswith=None):
    init_to = flax.traverse_util.flatten_dict(init_to)
    init_from = flax.traverse_util.flatten_dict(init_from)
    for (t_k, t_v) in init_from.items():
        layer_name = '.'.join(t_k)
        if exclude_startswith is not None and layer_name.startswith(exclude_startswith):
            logger.info(f'Explicitely skipping loading layer {layer_name}')
            continue
        elif t_k in init_to:
            if t_v.shape != init_to[t_k].shape:
                logger.warning(f'Tried to load layer {layer_name} but experienced shape mismatch (source: {t_v.shape} vs. target: {init_to[t_k].shape}). Skipping.')
                continue
            init_to[t_k] = t_v
        else:
            logger.info(f'Skipped loading layer {layer_name} as it was not found in target.')
    init_to = flax.traverse_util.unflatten_dict(init_to)
    return frozen_dict.freeze(init_to)


def get_total_device_memory():
    ts = timer()
    f_name = './mem.prof'
    memory_data = {}
    jax.profiler.save_device_memory_profile(f_name)
    out = subprocess.check_output(['./bin/pprof', '-tags', '-unit', 'MB', '-symbolize', 'none', f_name])
    out = out.decode()
    total_memory_str = out.split('\n')[0].split('Total ')[-1].rstrip('MB')
    memory_data['memory_usage_mb'] = float(total_memory_str)
    # collect per device data
    regex = r'\s*(\d+.\d+)MB \((\d+.\d+)%\)\: TPU_(\d+)\(process=(\d+)'
    memory_data['devices'] = []
    total_memory_all_devices = 0
    for line in out.split('\n')[1:]:
        match = re.match(regex, line)
        if match:
            memory_usage, memory_usage_perc, device_id, process_id = match.groups()
            device_data = dict(device_id=int(device_id), process_id=int(process_id), memory_usage_mb=float(memory_usage),
                               memory_usage_perc=float(memory_usage_perc))
            device_data['total_device_memory_mb'] = (1 / (device_data['memory_usage_perc']/100))*device_data['memory_usage_mb'] if device_data['memory_usage_perc'] > 0 else 0
            memory_data['devices'].append(device_data)
            total_memory_all_devices += device_data['total_device_memory_mb']
        else:
            # device list is over
            break
    memory_data['memory_usage_perc'] = 100 * memory_data['memory_usage_mb'] / total_memory_all_devices if total_memory_all_devices > 0 else 0
    memory_data['total_memory_all_devices_mb'] = total_memory_all_devices
    logger.debug(f'Total memory used: {memory_data["memory_usage_mb"]:.5f}MB (took {timer() - ts:.2f}s)')
    return memory_data
