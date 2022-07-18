import flax
import dill as pickle
import os
import builtins
from jax._src.lib import xla_client
import tensorflow as tf


# Hack: this is the module reported by this object.
# https://github.com/google/jax/issues/8505
builtins.bfloat16 = xla_client.bfloat16


def pickle_dump(obj, filename):
    """ Wrapper to dump an object to a file."""
    with tf.io.gfile.GFile(filename, "wb") as f:
        f.write(pickle.dumps(obj))


def pickle_load(filename):
    """ Wrapper to load an object from a file."""
    with tf.io.gfile.GFile(filename, 'rb') as f:
        pickled = pickle.loads(f.read())
    return pickled


def save_checkpoint(ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch, fid_score=None, keep=2):
    """
    Saves checkpoint.

    Args:
        ckpt_dir (str): Path to the directory, where checkpoints are saved.
        state_G (train_state.TrainState): Generator state.
        state_D (train_state.TrainState): Discriminator state.
        params_ema_G (frozen_dict.FrozenDict): Parameters of the ema generator.
        pl_mean (array): Moving average of the path length (generator regularization).
        config (argparse.Namespace): Configuration.
        step (int): Current step.
        epoch (int): Current epoch.
        fid_score (float): FID score corresponding to the checkpoint.
        keep (int): Number of checkpoints to keep.
    """
    state_dict = {'state_G': flax.jax_utils.unreplicate(state_G),
                  'state_D': flax.jax_utils.unreplicate(state_D),
                  'params_ema_G': params_ema_G,
                  'pl_mean': flax.jax_utils.unreplicate(pl_mean),
                  'config': config,
                  'fid_score': fid_score,
                  'step': step,
                  'epoch': epoch}

    pickle_dump(state_dict, os.path.join(ckpt_dir, f'ckpt_{step}.pickle'))
    ckpts = tf.io.gfile.glob(os.path.join(ckpt_dir, '*.pickle'))
    if len(ckpts) > keep:
        modified_times = {}
        for ckpt in ckpts:
            stats = tf.io.gfile.stat(ckpt)
            modified_times[ckpt] = stats.mtime_nsec
        oldest_ckpt = sorted(modified_times, key=modified_times.get)[0]
        tf.io.gfile.remove(oldest_ckpt)


def load_checkpoint(filename):
    """
    Loads checkpoints.

    Args:
        filename (str): Path to the checkpoint file.

    Returns:
        (dict): Checkpoint.
    """
    state_dict = pickle_load(filename)
    return state_dict


def get_latest_checkpoint(ckpt_dir):
    """
    Returns the path of the latest checkpoint.

    Args:
        ckpt_dir (str): Path to the directory, where checkpoints are saved.

    Returns:
        (str): Path to latest checkpoint (if it exists).
    """
    ckpts = tf.io.gfile.glob(os.path.join(ckpt_dir, '*.pickle'))
    if len(ckpts) == 0:
        return None

    modified_times = {}
    for ckpt in ckpts:
        stats = tf.io.gfile.stat(ckpt)
        modified_times[ckpt] = stats.mtime_nsec
    latest_ckpt = sorted(modified_times, key=modified_times.get)[-1]
    return latest_ckpt
