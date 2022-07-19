import argparse
import functools
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tqdm import tqdm

import checkpoint
from stylegan2.generator import Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s', force=True)
logger = logging.getLogger(__name__)


def generate_images(args):
    logger.info(f"Loading checking '{args.checkpoint}'...")
    ckpt = checkpoint.load_checkpoint(args.checkpoint)
    config = ckpt['config']
    params_ema_G = ckpt['params_ema_G']

    generator_ema = Generator(
        resolution=config.resolution,
        num_channels=config.img_channels,
        z_dim=config.z_dim,
        c_dim=config.c_dim,
        w_dim=config.w_dim,
        num_ws=int(np.log2(config.resolution)) * 2 - 3,
        num_mapping_layers=8,
        fmap_base=config.fmap_base,
        dtype=jnp.float32
    )

    generator_apply = jax.jit(
        functools.partial(generator_ema.apply, truncation_psi=args.truncation_psi, train=False, noise_mode='const')
    )

    logger.info(f"Generating {len(args.seeds)} images with truncation {args.truncation_psi}...")
    for seed in tqdm(args.seeds):
        rng = jax.random.PRNGKey(seed)
        z_latent = jax.random.normal(rng, shape=(1, config.z_dim))
        image = generator_apply(params_ema_G, jax.lax.stop_gradient(z_latent), None)
        image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))

        Image.fromarray(np.uint8(np.clip(image[0] * 255, 0, 255))).save(os.path.join(args.out_path, f'{seed}.png'))
    logger.info(f"Images saved in '{args.out_path}/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint.', required=True)
    parser.add_argument('--out_path', type=str, default='generated_images', help='Path where the generated images are stored.')
    parser.add_argument('--truncation_psi', type=float, default=0.5, help='Controls truncation (trading off variation for quality). If 1, truncation is disabled.')
    parser.add_argument('--seeds', type=int, nargs='*', default=[0], help='List of random seeds.')
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    generate_images(args)
