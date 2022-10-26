import optax
import jax
import jax.numpy as jnp
import flax
from flax.core import frozen_dict
import logging
import numpy as np
from collections import Counter


logger = logging.getLogger(__name__)


def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


def get_g_optimizer(config, params_G):
    adam = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    if config.freeze_g is None:
        return adam
    # with layer freezing
    assert config.freeze_g <= int(np.log2(config.resolution)) - 1, 'freeze_g is larger than the number of layers in generator'
    adam = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)

    def create_mask(params, freeze_g):
        mask = flax.traverse_util.flatten_dict(params)
        frozen_synth_blocks = tuple([f'synthesis.SynthesisBlock_{i}' for i in range(freeze_g)])
        counter = Counter()
        for k, v in mask.items():
            key = '.'.join(k)
            if freeze_g < 1:
                mask[k] = 'trainable'
            elif key.startswith('mapping'):
                logger.info(f'Freezing layer {key}')
                mask[k] = 'frozen'  # even with freeze_g == 1 we freeze entire mapping network
            elif key.startswith(frozen_synth_blocks):
                logger.info(f'Freezing layer {key}')
                mask[k] = 'frozen'
            else:
                mask[k] = 'trainable'
            counter[mask[k]] += v.size
        mask = flax.traverse_util.unflatten_dict(mask)
        return frozen_dict.freeze(mask), counter
    mask, stats = create_mask(params_G, config.freeze_g)
    logger.info(f'Generator has {stats["trainable"]:,} trainable parameters and {stats["frozen"]:,} frozen parameters')
    tx_G = optax.multi_transform({'trainable': adam, 'frozen': zero_grads()}, mask)
    return tx_G


def get_d_optimizer(config, params_D):
    adam = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    if config.freeze_d is None:
        return adam
    # with layer freezing
    assert config.freeze_d <= int(np.log2(config.resolution)) - 2, 'freeze_d is larger than the number of blocks in discriminator'

    def create_mask(params, freeze_d, top_layer):
        mask = flax.traverse_util.flatten_dict(params)
        frozen_disc_blocks = tuple([f'params.DiscriminatorBlock_{i}' for i in range(top_layer, top_layer-freeze_d, -1)])
        counter = Counter()
        for k, v in mask.items():
            key = '.'.join(k)
            # TODO: Should we also freeze the very last layers here? (i.e. DiscriminatorLayer_0 and LinearLayer_0/1)
            if freeze_d < 1:
                mask[k] = 'trainable'
            elif key.startswith(frozen_disc_blocks):
                logger.info(f'Freezing layer {key}')
                mask[k] = 'frozen'
            else:
                mask[k] = 'trainable'
            counter[mask[k]] += v.size
        mask = flax.traverse_util.unflatten_dict(mask)
        return frozen_dict.freeze(mask), counter
    top_layer = int(np.log2(config.resolution)) - 3
    mask, stats = create_mask(params_D, config.freeze_d, top_layer)
    logger.info(f'Discriminator has {stats["trainable"]:,} trainable parameters and {stats["frozen"]:,} frozen parameters')
    tx_D = optax.multi_transform({'trainable': adam, 'frozen': zero_grads()}, mask)
    return tx_D
