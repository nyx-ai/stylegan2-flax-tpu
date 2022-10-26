import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
import flax
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.core import frozen_dict
import numpy as np
import functools
import wandb
from timeit import default_timer as timer
from jax.lib import xla_bridge
import logging
import sys

from fid import FID
import stylegan2
import data_pipeline
import checkpoint
import training_utils
import training_steps
import optimizers
import gcs

logger = logging.getLogger(__name__)

cc.initialize_cache("jax_cache")


def train_and_evaluate(config):
    num_devices = jax.device_count()   # 8
    num_local_devices = jax.local_device_count()  # 4
    num_workers = jax.process_count()

    # --------------------------------------
    # Data
    # --------------------------------------
    ds_train, dataset_info = data_pipeline.get_data(data_dir=config.data_dir,
                                                    img_size=config.resolution,
                                                    img_channels=config.img_channels,
                                                    num_classes=config.c_dim,
                                                    num_local_devices=num_local_devices,
                                                    batch_size=config.batch_size,
                                                    allow_resolution_mismatch=config.allow_resolution_mismatch)
    has_labels = config.c_dim > 0

    # --------------------------------------
    # Seeding and Precision
    # --------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    platform = xla_bridge.get_backend().platform
    if config.mixed_precision:
        if platform == 'tpu':
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float16
    else:
        dtype = jnp.float32
    logger.info(f'Running on dtype {dtype}')
    device_type = None
    if platform == 'tpu':
        device_type = jax.devices()[0].device_kind
        logger.info(f'Running on device type {device_type}')

    dynamic_scale_G_main = None
    dynamic_scale_D_main = None
    dynamic_scale_G_reg = None
    dynamic_scale_D_reg = None
    if config.mixed_precision:
        # for float16 we require loss scaling in order to avoid precision issues
        if platform != 'tpu':
            # loss scale is likely not required for bfloat16
            dynamic_scale_G_main = dynamic_scale_lib.DynamicScale()
            dynamic_scale_D_main = dynamic_scale_lib.DynamicScale()
            dynamic_scale_G_reg = dynamic_scale_lib.DynamicScale()
            dynamic_scale_D_reg = dynamic_scale_lib.DynamicScale()
        clip_conv = 256
        num_fp16_res = 4
    else:
        clip_conv = None
        num_fp16_res = 0

    cpu_devices = jax.devices('cpu')
    assert len(cpu_devices) >= 1

    # --------------------------------------
    # Initialize Models
    # --------------------------------------
    logger.info('Initialize models...')

    rng, init_rng = jax.random.split(rng)

    # Generator initialization for training
    start_mn = timer()
    logger.info("Creating MappingNetwork...")
    mapping_net = stylegan2.MappingNetwork(z_dim=config.z_dim,
                                           c_dim=config.c_dim,
                                           w_dim=config.w_dim,
                                           num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                           num_layers=8,
                                           dtype=dtype)

    mapping_net_vars = mapping_net.init(init_rng,
                                        jnp.ones((1, config.z_dim)),
                                        jnp.ones((1, config.c_dim)))

    mapping_net_params, moving_stats = mapping_net_vars['params'], mapping_net_vars['moving_stats']
    logger.info(f"MappingNetwork took {timer() - start_mn:.2f}s")
    training_utils.visualize_model(mapping_net_params, name='Mapping network')

    logger.info("Creating SynthesisNetwork...")
    start_sn = timer()
    synthesis_net = stylegan2.SynthesisNetwork(resolution=config.resolution,
                                               num_channels=config.img_channels,
                                               w_dim=config.w_dim,
                                               fmap_base=config.fmap_base,
                                               fmap_max=config.fmap_max,
                                               num_fp16_res=num_fp16_res,
                                               clip_conv=clip_conv,
                                               dtype=dtype)

    synthesis_net_vars = synthesis_net.init(init_rng, jnp.ones((1, mapping_net.num_ws, config.w_dim)))
    synthesis_net_params, noise_consts = synthesis_net_vars['params'], synthesis_net_vars['noise_consts']
    logger.info(f"SynthesisNetwork took {timer() - start_sn:.2f}s")
    training_utils.visualize_model(synthesis_net_params, name='Synthesis network')

    params_G = frozen_dict.FrozenDict(
        {'mapping': mapping_net_params,
         'synthesis': synthesis_net_params}
    )

    # Discriminator initialization for training
    logger.info("Creating Discriminator...")
    start_d = timer()
    discriminator = stylegan2.Discriminator(resolution=config.resolution,
                                            num_channels=config.img_channels,
                                            c_dim=config.c_dim,
                                            mbstd_group_size=config.mbstd_group_size,
                                            num_fp16_res=num_fp16_res,
                                            clip_conv=clip_conv,
                                            dtype=dtype)
    rng, init_rng = jax.random.split(rng)
    params_D = discriminator.init(init_rng,
                                  jnp.ones((1, config.resolution, config.resolution, config.img_channels)),
                                  jnp.ones((1, config.c_dim)))
    logger.info(f"Discriminator took {timer() - start_d:.2f}s")
    training_utils.visualize_model(params_D, name='Discriminator')

    # Exponential average Generator initialization
    logger.info("Creating Generator EMA...")
    start_g = timer()
    generator_ema = stylegan2.Generator(resolution=config.resolution,
                                        num_channels=config.img_channels,
                                        z_dim=config.z_dim,
                                        c_dim=config.c_dim,
                                        w_dim=config.w_dim,
                                        num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                        num_mapping_layers=8,
                                        fmap_base=config.fmap_base,
                                        fmap_max=config.fmap_max,
                                        num_fp16_res=num_fp16_res,
                                        clip_conv=clip_conv,
                                        dtype=dtype)

    params_ema_G = generator_ema.init(init_rng,
                                      jnp.ones((1, config.z_dim)),
                                      jnp.ones((1, config.c_dim)))
    logger.info(f"Generator EMA took {timer() - start_g:.2f}s")

    # --------------------------------------
    # Initialize States and Optimizers
    # --------------------------------------
    logger.info('Initialize states...')
    start_state = timer()
    tx_G = optimizers.get_g_optimizer(config, params_G)
    tx_D = optimizers.get_d_optimizer(config, params_D)

    state_G = training_utils.TrainStateG.create(apply_fn=None,
                                                apply_mapping=mapping_net.apply,
                                                apply_synthesis=synthesis_net.apply,
                                                params=params_G,
                                                moving_stats=moving_stats,
                                                noise_consts=noise_consts,
                                                tx=tx_G,
                                                dynamic_scale_main=dynamic_scale_G_main,
                                                dynamic_scale_reg=dynamic_scale_G_reg,
                                                epoch=0)

    state_D = training_utils.TrainStateD.create(apply_fn=discriminator.apply,
                                                params=params_D,
                                                tx=tx_D,
                                                dynamic_scale_main=dynamic_scale_D_main,
                                                dynamic_scale_reg=dynamic_scale_D_reg,
                                                epoch=0)

    # Copy over the parameters from the training generator to the ema generator
    params_ema_G = training_utils.update_generator_ema(state_G, params_ema_G, config, ema_beta=0)
    logger.info(f'Initializing states took {timer()-start_state:.2f}s')

    # Running mean of path length for path length regularization
    pl_mean = jnp.zeros((), dtype=dtype)

    step = 0
    epoch_offset = 0
    best_fid_score = np.inf
    steps_since_best_fid = 0
    ckpt_path = None
    num_steps_profiling = 1

    if config.resume_run_id is not None:
        #  Resume training from existing checkpoint
        ckpt_path = checkpoint.get_latest_checkpoint(config.ckpt_dir)
        logger.info(f'Resume training from checkpoint: {ckpt_path}')
        ckpt = checkpoint.load_checkpoint(ckpt_path)
        step = ckpt['step']
        epoch_offset = ckpt['epoch']
        best_fid_score = ckpt['fid_score']
        pl_mean = ckpt['pl_mean']
        state_G = ckpt['state_G']
        state_D = ckpt['state_D']
        params_ema_G = ckpt['params_ema_G']
        config = ckpt['config']
    elif config.load_from_ckpt is not None:
        # Load checkpoint and start new run
        ts_ckpt = timer()
        ckpt_path = config.load_from_ckpt
        logger.info(f'Load model state from {ckpt_path}')
        ckpt = checkpoint.load_checkpoint(ckpt_path)
        ckpt_config = ckpt['config']
        if ckpt_config.resolution > config.resolution:
            raise Exception(f'Loaded checkpoint has resolution {ckpt_config.resolution} which is smaller than requested resolution {config.resolution}')
        elif ckpt_config.resolution < config.resolution:
            # Finetune from smaller res model
            logger.info(f'Loading {config.resolution} res model from {ckpt_config.resolution} res checkpoint...')
            # Note: Avoiding copying pl_mean here
            state_D = state_D.replace(params=training_utils.partial_load_from(state_D.params, ckpt['state_D'].params,
                                      exclude_startswith=('params.LinearLayer', 'params.DiscriminatorLayer')))  # exclude head
            state_G = state_G.replace(params=training_utils.partial_load_from(state_G.params, ckpt['state_G'].params))
            params_ema_G = training_utils.partial_load_from(params_ema_G, ckpt['params_ema_G'])
        else:
            # simply copying over entire state (includes optimizer)
            pl_mean = ckpt['pl_mean']
            state_G = ckpt['state_G']
            state_D = ckpt['state_D']
            params_ema_G = ckpt['params_ema_G']
        logger.info(f'... successfully loaded from ckpt (took {timer()-ts_ckpt:.2f}s)')

    # Log param counts
    num_params_mapping = training_utils.count_params(mapping_net_params)
    num_params_synth = training_utils.count_params(synthesis_net_params)
    num_params_disc = training_utils.count_params(params_D)
    num_params_gen = training_utils.count_params(params_G)
    num_params_total = num_params_gen + num_params_disc
    logger.info(f'Total parameter counts:\n- Mapping Network:\t{num_params_mapping:>12,}\n'
                f'- Synthesis network:\t{num_params_synth:>12,}\n- Generator total:\t{num_params_gen:>12,}\n'
                f'- Discriminator total:\t{num_params_disc:>12,}\n- Total:\t\t{num_params_total:>12,}')

    # Replicate states across devices
    pl_mean = flax.jax_utils.replicate(pl_mean)
    state_G = flax.jax_utils.replicate(state_G)
    state_D = flax.jax_utils.replicate(state_D)

    # --------------------------------------
    # Precompile train and eval steps
    # --------------------------------------
    logger.info('Precompile training steps...')
    p_main_step_G = jax.pmap(training_steps.main_step_G, axis_name='batch')
    if config.G_reg_interval > 0:
        p_regul_step_G = jax.pmap(functools.partial(training_steps.regul_step_G, config=config), axis_name='batch')
    else:
        logger.info('Not using generator regularization')

    p_main_step_D = jax.pmap(training_steps.main_step_D, axis_name='batch')
    if config.D_reg_interval > 0:
        p_regul_step_D = jax.pmap(functools.partial(training_steps.regul_step_D, config=config), axis_name='batch')
    else:
        logger.info('Not using discriminator regularization')

    # --------------------------------------
    # Training
    # --------------------------------------
    logger.info('Start training...')
    if config.eval_fid_every > 0:
        fid_metric = FID(generator_ema, ds_train, config)
        fid_metric_trunc = FID(generator_ema, ds_train, config, truncation_psi=.5) if config.eval_fid_trunc else None
    else:
        logger.info('Not running FID')

    # Dict to collect training statistics / losses
    metrics = {}
    num_imgs_processed = 0
    num_steps_per_epoch = dataset_info['num_examples'] // (config.batch_size * num_devices)
    effective_batch_size = config.batch_size * num_devices
    if config.wandb and jax.process_index() == 0:
        # do some more logging
        wandb.config.effective_batch_size = effective_batch_size
        wandb.config.num_steps_per_epoch = num_steps_per_epoch
        wandb.config.num_workers = num_workers
        wandb.config.device_count = num_devices
        wandb.config.num_examples = dataset_info['num_examples']
        wandb.config.vm_name = gcs.get_vm_name()
        wandb.config.device_type = device_type
        if 'classes' not in dataset_info:
            msg = 'dataset_info.json does not contain list of classes under key `classes`.'
            if has_labels:
                raise Exception(msg)
            else:
                logger.warning(msg)
        else:
            wandb.config.label_list = dataset_info['classes']

    # compile training data snapshot for W&B
    compiling_train_snapshot = config.generate_snapshot_every > 0  # disable training snapshots if negative
    train_snapshot_data = jnp.array([])
    size_train_snapshot = 4096

    # generate a fixed latent vector used for eval
    rng, key = jax.random.split(rng)
    z_latent_fixed = jax.random.normal(key, (config.num_samples_per_eval_snapshot // config.batch_size, config.batch_size, config.z_dim), dtype)
    if has_labels:
        labels_fixed_arr = training_utils.generate_random_labels(rng, (config.num_samples_per_eval_snapshot // config.batch_size, config.batch_size), config.c_dim)
    else:
        labels_fixed_arr = None

    is_profiling = False
    if config.profile:
        logger.info('Start profiling...')
        jax.profiler.start_trace('./profile')
        is_profiling = True

    for epoch in range(epoch_offset, config.num_epochs):
        if config.wandb and jax.process_index() == 0:
            wandb.log({'training/epochs': epoch}, step=step)

        for batch in data_pipeline.prefetch(ds_train, config.num_prefetch):
            assert batch['image'].shape[1] == config.batch_size, f"Mismatched batch (batch size: {config.batch_size}, this batch: {batch['image'].shape[1]})"

            # pbar.update(num_devices * config.batch_size)
            iteration_start_time = timer()

            if not has_labels:
                # No labels in the dataset
                batch['label'] = None

            # Create two latent noise vectors and combine them for the style mixing regularization
            rng, key = jax.random.split(rng)
            z_latent1 = jax.random.normal(key, (num_local_devices, config.batch_size, config.z_dim), dtype)
            rng, key = jax.random.split(rng)
            z_latent2 = jax.random.normal(key, (num_local_devices, config.batch_size, config.z_dim), dtype)

            # Split PRNGs across devices
            rkey = jax.random.split(key, num=num_local_devices)
            mixing_prob = flax.jax_utils.replicate(config.mixing_prob)

            # --------------------------------------
            # Update Discriminator
            # --------------------------------------
            time_d_start = timer()
            state_D, metrics = p_main_step_D(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            time_d_end = timer()
            if config.D_reg_interval > 0 and step % config.D_reg_interval == 0:
                state_D, metrics = p_regul_step_D(state_D, batch, metrics)

            # --------------------------------------
            # Update Generator
            # --------------------------------------
            time_g_start = timer()
            state_G, metrics = p_main_step_G(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            if config.G_reg_interval > 0 and step % config.G_reg_interval == 0:
                H, W = batch['image'].shape[-3], batch['image'].shape[-2]
                rng, key = jax.random.split(rng)
                pl_noise = jax.random.normal(key, batch['image'].shape, dtype=dtype) / np.sqrt(H * W)
                state_G, metrics, pl_mean = p_regul_step_G(state_G, batch, z_latent1, pl_noise, pl_mean, metrics, rng=rkey)

            params_ema_G = training_utils.update_generator_ema(flax.jax_utils.unreplicate(state_G),
                                                               params_ema_G,
                                                               config)
            time_g_end = timer()

            # --------------------------------------
            # Logging and Checkpointing
            # --------------------------------------
            if config.save_every > 0 and step % config.save_every == 0 and jax.process_index() == 0:
                checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step,
                                           epoch, keep_best=config.keep_n_best_checkpoints)

            num_imgs_processed += num_devices * config.batch_size
            if config.eval_fid_every > 0 and step % config.eval_fid_every == 0 and jax.process_index() == 0:
                # If FID evaluation is enabled, only save a checkpoint if FID score is better.
                logger.info('Computing FID...')
                ts = timer()
                fid_scores = fid_metric.compute_fid(params_ema_G)
                te = timer()
                if config.wandb:
                    wandb.log({f'training/gen/{k}': v for k, v in fid_scores.items()}, step=step)
                fid_scores_str = ' | '.join([f'{k}: {v:.3f}' for k, v in fid_scores.items()])
                logger.info(f'Computed FID score(s): | {fid_scores_str} | (took {(te-ts):.1f}s)')
                if config.eval_fid_trunc:
                    logger.info('Computing truncated FID...')
                    ts = timer()
                    fid_scores_trunc = fid_metric_trunc.compute_fid(params_ema_G)
                    te = timer()
                    if config.wandb:
                        wandb.log({f'training/gen/{k}_trunc': v for k, v in fid_scores_trunc.items()}, step=step)
                    fid_scores_str = ' | '.join([f'{k}: {v:.3f}' for k, v in fid_scores_trunc.items()])
                    logger.info(f'Computed truncated FID score(s): | {fid_scores_str} | (took {(te-ts):.1f}s)')
                if best_fid_score is None or fid_scores['fid'] < best_fid_score:
                    best_fid_score = fid_scores['fid']
                    steps_since_best_fid = 0
                    logger.info(f'🎉 New best FID score ({best_fid_score:.3f}). Saving checkpoint...')
                    checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch,
                                               fid_score=fid_scores['fid'], is_best=True, keep_best=config.keep_n_best_checkpoints)
                else:
                    steps_since_best_fid += config.eval_fid_every
                if config.early_stopping_after_steps is not None and steps_since_best_fid > config.early_stopping_after_steps:
                    logger.info(f'Early stopping of training after {steps_since_best_fid:,} steps with no FID improvement.')
                    sys.exit(0)

            sec_per_kimg = (timer() - iteration_start_time) / (num_devices * config.batch_size / 1000.0)
            time_taken_g = time_g_end - time_g_start
            time_taken_d = time_d_end - time_d_start
            time_taken_per_step = timer() - iteration_start_time
            g_loss = jnp.mean(metrics['G_loss']).item()
            d_loss = jnp.mean(metrics['D_loss']).item()

            if config.wandb and jax.process_index() == 0:
                # wandb logging - happens every step
                wandb.log({'training/gen/loss': jnp.mean(metrics['G_loss']).item()}, step=step, commit=False)
                wandb.log({'training/gen/pl_mean': jnp.mean(pl_mean).item()}, step=step, commit=False)
                wandb.log({'training/dis/loss': jnp.mean(metrics['D_loss']).item()}, step=step, commit=False)
                wandb.log({'training/dis/fake_logits': jnp.mean(metrics['fake_logits']).item()}, step=step, commit=False)
                wandb.log({'training/dis/real_logits': jnp.mean(metrics['real_logits']).item()}, step=step, commit=False)
                wandb.log({'training/time_taken_g': time_taken_g, 'training/time_taken_d': time_taken_d}, step=step, commit=False)
                wandb.log({'training/time_taken_per_step': time_taken_per_step}, step=step, commit=False)
                wandb.log({'training/num_imgs_trained': num_imgs_processed}, step=step, commit=False)
                wandb.log({'training/sec_per_kimg': sec_per_kimg}, step=step)

                # compile training data snapshot
                if compiling_train_snapshot:
                    real_images = jax.device_put(flax.jax_utils.unreplicate(batch['image']), cpu_devices[0])
                    train_snapshot_data = jnp.concatenate((train_snapshot_data, real_images), axis=0) if train_snapshot_data.size else real_images
                    img_width = batch['image'].shape[-3]
                    if len(train_snapshot_data) >= (size_train_snapshot // img_width)**2:
                        # this will only be executed once per training
                        logger.info('Writing training/real_snapshot...')
                        train_snapshot = training_utils.get_grid_snapshot(train_snapshot_data, grid_size_px=size_train_snapshot, sample_normalization=False)
                        wandb.log({'training/real_snapshot': wandb.Image(train_snapshot)})
                        compiling_train_snapshot = False

            mem = None
            if step % config.log_every == 0:
                # console logging - happens every log_every steps
                log_str = f'Total steps: {step:>6,} - epoch {epoch:>3,}/{config.num_epochs} @ {step % num_steps_per_epoch:>6,}/{num_steps_per_epoch:,} ' \
                          f'- G loss: {g_loss:>8.5f} - D loss: {d_loss:>8.5f} - sec/kimg: {sec_per_kimg:>8.2f}s - time per step: {time_taken_per_step:>8.3f}s'
                if not config.disable_memory_profile:
                    mem = training_utils.get_total_device_memory()
                    log_str += f' - memory usage: {mem["memory_usage_mb"]:>5.0f}MB ({mem["memory_usage_perc"]:.2f}%)'
                logger.info(log_str)

            if not config.disable_memory_profile and config.wandb and (step < 50 or step % config.log_every == 0) and jax.process_index() == 0:
                # log device memory
                if mem is None:
                    mem = training_utils.get_total_device_memory()
                memory_logs = {f'training/memory_usage_device_{v["device_id"]}_mb': v['memory_usage_mb'] for v in mem['devices']}
                wandb.log({'training/memory_usage_mb': mem['memory_usage_mb'], 'training/memory_usage_perc': mem['memory_usage_perc'], **memory_logs}, step=step)

            if config.generate_snapshot_every > 0 and step % config.generate_snapshot_every == 0 and config.wandb and jax.process_index() == 0:
                # Get "fake" training samples (there are only batch_size many)
                snapshot_data = jax.device_put(flax.jax_utils.unreplicate(metrics['image_gen']), cpu_devices[0])
                train_snapshot = training_utils.get_grid_snapshot(snapshot_data, grid_size=(2, 2))
                wandb.log({'training/snapshot': wandb.Image(train_snapshot)}, commit=False, step=step)

                # Generate evaluation images
                eval_images = jnp.array([])
                eval_images_trunc = jnp.array([])
                eval_images_fixed = jnp.array([])
                num_batches = config.num_samples_per_eval_snapshot // config.batch_size
                labels_caption = {k: jnp.array([]) for k in ['not_fixed', 'fixed']}
                for idx in range(num_batches):
                    # generate random latents
                    if has_labels:
                        labels = training_utils.generate_random_labels(rng, (config.batch_size,), config.c_dim)
                        labels_classes = jnp.argmax(labels, axis=-1)
                        labels_caption['not_fixed'] = jnp.concatenate((labels_caption['not_fixed'], labels_classes), axis=0) if labels_caption['not_fixed'].size else labels_classes
                        labels_fixed = labels_fixed_arr[idx]
                        labels_fixed_classes = jnp.argmax(labels_fixed, axis=-1)
                        labels_caption['fixed'] = jnp.concatenate((labels_caption['fixed'], labels_fixed_classes), axis=0) if labels_caption['fixed'].size else labels_fixed_classes
                    else:
                        labels = None
                        labels_fixed = None
                    rng, key = jax.random.split(rng)
                    z_latent_eval = jax.random.normal(key, (config.batch_size, config.z_dim), dtype)
                    # without truncation
                    image_gen_eval = training_steps.eval_step_G(
                        generator_ema,
                        params=params_ema_G,
                        z_latent=z_latent_eval,
                        labels=labels,
                        truncation=1
                    )
                    eval_images = jnp.concatenate((eval_images, image_gen_eval), axis=0) if eval_images.size else image_gen_eval
                    # with truncation
                    image_gen_eval_trunc = training_steps.eval_step_G(
                        generator_ema,
                        params=params_ema_G,
                        z_latent=z_latent_eval,
                        labels=labels,
                        truncation=0.5
                    )
                    eval_images_trunc = jnp.concatenate((eval_images_trunc, image_gen_eval_trunc), axis=0) if eval_images_trunc.size else image_gen_eval_trunc
                    # with fixed latent vector + truncation
                    image_gen_eval_fixed = training_steps.eval_step_G(
                        generator_ema,
                        params=params_ema_G,
                        z_latent=z_latent_fixed[idx],
                        labels=labels_fixed,
                        truncation=0.5
                    )
                    eval_images_fixed = jnp.concatenate((eval_images_fixed, image_gen_eval_fixed), axis=0) if eval_images_fixed.size else image_gen_eval_fixed
                grid_size_row = int(np.sqrt(config.num_samples_per_eval_snapshot))
                grid_size = (grid_size_row, grid_size_row)
                eval_snapshot = training_utils.get_grid_snapshot(eval_images, grid_size=grid_size)
                eval_snapshot_trunc = training_utils.get_grid_snapshot(eval_images_trunc, grid_size=grid_size)
                eval_snapshot_fixed = training_utils.get_grid_snapshot(eval_images_fixed, grid_size=grid_size)
                labels_caption['not_fixed'] = list(np.array(labels_caption['not_fixed']))
                labels_caption['fixed'] = list(np.array(labels_caption['fixed']))
                captions_not_fixed = ''.join([f'{l} ' if (i + 1) % grid_size[0] != 0 else f'{l}\n' for i, l in enumerate(labels_caption['not_fixed'])])
                captions_fixed = ''.join([f'{l} ' if (i + 1) % grid_size[0] != 0 else f'{l}\n' for i, l in enumerate(labels_caption['fixed'])])
                wandb.log({'eval/snapshot': wandb.Image(eval_snapshot, caption=captions_not_fixed)}, commit=False, step=step)
                wandb.log({'eval/snapshot_trunc': wandb.Image(eval_snapshot_trunc, caption=captions_not_fixed)}, commit=False, step=step)
                wandb.log({'eval/snapshot_fixed': wandb.Image(eval_snapshot_fixed, caption=captions_fixed)}, step=step)

            # profiling
            if is_profiling and config.profile and (step + 1) >= num_steps_profiling:
                logger.info('Stopping profiling...')
                jax.profiler.stop_trace()
                is_profiling = False
                logger.info('... done')

            step += 1
            if step > config.num_steps:
                logger.info(f'Stopping training since having reached maximum of {config.num_steps:,} steps of training.')
                sys.exit(0)

        # Sync moving stats across devices
        state_G = training_utils.sync_moving_stats(state_G)

        # Sync moving average of path length mean (Generator regularization)
        pl_mean = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='batch'), axis_name='batch')(pl_mean)
    else:
        logger.info(f'Stopping training since having reached maximum of {config.num_epochs:,} epochs of training.')
