import jax
import jax.numpy as jnp
import flax
import numpy as np
import functools
import scipy
import logging
from timeit import default_timer as timer
import os

from . import inception
from . import utils

logger = logging.getLogger(__name__)


class FID:
    def __init__(self, generator, dataset, config, use_cache=True, truncation_psi=1.0):
        """
        Evaluates the FID score for a given generator and a given dataset.
        Implementation mostly taken from https://github.com/matthias-wright/jax-fid

        Reference: https://arxiv.org/abs/1706.08500

        Args:
            generator (nn.Module): Generator network.
            dataset (tf.data.Dataset): Dataset containing the real images.
            config (argparse.Namespace): Configuration.
            use_cache (bool): If True, only compute the activation stats once for the real images and store them.
            truncation_psi (float): Controls truncation (trading off variation for quality). If 1, truncation is disabled.
        """
        self.num_images_fake = config.num_fid_images_fake
        self.num_images_real = config.num_fid_images_real
        self.batch_size = config.batch_size
        self.c_dim = config.c_dim
        self.z_dim = config.z_dim
        self.dataset = dataset
        self.num_devices = jax.device_count()
        self.num_local_devices = jax.local_device_count()
        self.use_cache = use_cache
        self.has_labels = config.c_dim > 0
        self.do_resize = config.resolution < 75  # running anything below res 75 breaks inception.
        if self.do_resize:
            logger.warning(f'FID computation for resolution {config.resolution} is not supported. Will resize images to (299, 299, 3)!')

        if self.use_cache:
            data_dir = os.path.basename(config.data_dir.rstrip('/'))
            cached_args = dict(metric='fid', resolution=config.resolution, c_dim=config.c_dim, num_fid_images_real=config.num_fid_images_real,
                               num_fid_images_fake=config.num_fid_images_fake, data_dir=data_dir, augmentation='random_left_right_flip', version=0)
            self.cache = utils.Cache(cached_args, f_path=config.metric_cache_location)

        rng = jax.random.PRNGKey(0)
        inception_net = inception.InceptionV3(pretrained=True)
        self.inception_params = inception_net.init(rng, jnp.ones((1, config.resolution, config.resolution, 3)))
        self.inception_params = flax.jax_utils.replicate(self.inception_params)
        # self.inception = jax.jit(functools.partial(model.apply, train=False))
        self.inception_apply = jax.pmap(functools.partial(inception_net.apply, train=False), axis_name='batch')
        self.generator_apply = jax.pmap(functools.partial(generator.apply, truncation_psi=truncation_psi, train=False, noise_mode='const'), axis_name='batch')

    def compute_fid(self, generator_params, seed_offset=0):
        generator_params = flax.jax_utils.replicate(generator_params)
        stats_real = self.compute_stats_for_dataset()
        stats_fake = self.compute_stats_for_generator(generator_params, seed_offset)
        fid_scores = self.compute_frechet_distance(stats_real, stats_fake, eps=1e-6)
        return fid_scores

    def compute_frechet_distance(self, stats_real, stats_fake, eps=1e-6):
        # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        if self.has_labels:
            all_labels = list(range(self.c_dim)) + [None]
        else:
            all_labels = [None]

        fid_scores = {}
        for label in all_labels:
            if label is None:
                mu1 = stats_real['mu']
                mu2 = stats_fake['mu']
                sigma1 = stats_real['sigma']
                sigma2 = stats_fake['sigma']
            else:
                try:
                    mu1 = stats_real[f'mu_{label}']
                    mu2 = stats_fake[f'mu_{label}']
                    sigma1 = stats_real[f'sigma_{label}']
                    sigma2 = stats_fake[f'sigma_{label}']
                except KeyError:
                    logger.warning(f'Could not compute FID for class {label}')
                    continue
            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)
            sigma1 = np.atleast_1d(sigma1)
            sigma2 = np.atleast_1d(sigma2)

            assert mu1.shape == mu2.shape
            assert sigma1.shape == sigma2.shape

            diff = mu1 - mu2

            covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if not np.isfinite(covmean).all():
                msg = ('fid calculation produces singular product; '
                       'adding %s to diagonal of cov estimates') % eps
                logger.info(msg)
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            # Numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError('Imaginary component {}'.format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)
            fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
            if label is None:
                fid_scores['fid'] = fid
            else:
                fid_scores[f'fid_{label}'] = fid
        return fid_scores

    def compute_stats_for_dataset(self):
        if self.use_cache and self.cache.exists:
            data = self.cache.read()
            if 'mu' in data and 'sigma' in data:
                logger.info('Use cached statistics for dataset...')
                return data

        logger.info('Compute statistics for dataset...')
        ts = timer()
        image_count = 0
        activations = []
        labels = np.array([])
        for batch in utils.prefetch(self.dataset, n_prefetch=2):
            if self.do_resize:
                batch['image'] = jax.image.resize(batch['image'], batch['image'].shape[:-3] + (299, 299, 3), method='bilinear')
            act = self.inception_apply(self.inception_params, jax.lax.stop_gradient(batch['image']))
            act = jnp.reshape(act, (self.num_local_devices * self.batch_size, -1))
            activations.append(np.array(act))
            if self.has_labels:
                batch_labels = np.reshape(np.argmax(np.array(batch['label']), axis=-1), (self.batch_size * self.num_local_devices,))
                labels = np.concatenate((labels, batch_labels), axis=0) if labels.size else batch_labels
                per_class_counts = np.bincount(labels)
                if len(per_class_counts) == self.c_dim and (per_class_counts > self.num_images_real).all():
                    break
            else:
                image_count += self.num_local_devices * self.batch_size
                if image_count >= self.num_images_real:
                    break
        else:
            if self.has_labels:
                logger.warning(f'Could not find {self.num_images_real:,} images for each class for per-class FID calculation, instead used entire dataset.')
            else:
                logger.warning(f'Could not find {self.num_images_real:,} images in dataset, instead used entire dataset.')

        activations = np.concatenate(activations, axis=0)
        data_to_cache = {}
        if self.has_labels:
            # compute statistics per label
            for lab in np.unique(labels):
                act_label = activations[labels == lab]
                act_label = act_label[:self.num_images_real]
                logger.info(f'... computing FID dataset statistics for class {lab} over {len(act_label):,} images')
                mu = np.mean(act_label, axis=0)
                sigma = np.cov(act_label, rowvar=False)
                data_to_cache[f'mu_{lab}'] = mu
                data_to_cache[f'sigma_{lab}'] = sigma
                data_to_cache[f'size_class_{lab}'] = len(act_label)
        else:
            activations = activations[:self.num_images_real]

        # overall statistics
        logger.info(f'... computing overall FID dataset statistics over {len(activations):,} images')
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        data_to_cache['mu'] = mu
        data_to_cache['sigma'] = sigma
        if self.use_cache:
            self.cache.dump(data_to_cache)
        logger.info(f'... dataset statistics took {(timer()-ts)/60:.1f}min')
        return data_to_cache

    def compute_stats_for_generator(self, generator_params, seed_offset):
        logger.info('Compute statistics for generator...')
        ts = timer()
        if self.has_labels:
            all_labels = list(range(self.c_dim))
        else:
            all_labels = [None]

        stats = {}
        all_activations = []
        for label in all_labels:
            num_batches = int(np.ceil(self.num_images_fake / (self.batch_size * self.num_local_devices)))
            activations = []

            for i in range(num_batches):
                rng = jax.random.PRNGKey(seed_offset + i)
                z_latent = jax.random.normal(rng, shape=(self.num_local_devices, self.batch_size, self.z_dim))

                labels = None
                if self.has_labels:
                    labels = jnp.ones((self.num_local_devices * self.batch_size,)) * label
                    labels = jax.nn.one_hot(labels, num_classes=self.c_dim)
                    labels = jnp.reshape(labels, (self.num_local_devices, self.batch_size, self.c_dim))

                image = self.generator_apply(generator_params, jax.lax.stop_gradient(z_latent), labels)
                image = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image))
                image = 2 * image - 1
                if self.do_resize:
                    image = jax.image.resize(image, image.shape[:-3] + (299, 299, 3), method='bilinear')
                act = self.inception_apply(self.inception_params, jax.lax.stop_gradient(image))
                act = jnp.reshape(act, (self.num_local_devices * self.batch_size, -1))
                activations.append(np.array(act))

            activations = np.concatenate(activations, axis=0)
            activations = activations[:self.num_images_fake]
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)
            if self.has_labels:
                stats[f'mu_{label}'] = mu
                stats[f'sigma_{label}'] = sigma
                all_activations.append(activations)
            else:
                stats['mu'] = mu
                stats['sigma'] = sigma
        if self.has_labels:
            # compute overall
            all_activations = np.concatenate(all_activations, axis=0)
            stats['mu'] = np.mean(all_activations, axis=0)
            stats['sigma'] = np.cov(all_activations, rowvar=False)
        logger.info(f'... generator statistics took {(timer()-ts)/60:.1f}min')
        return stats
