import jax
import jax.numpy as jnp
import flax
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.core import frozen_dict
import optax
import numpy as np
import functools
import wandb
import time

import stylegan2
import data_pipeline
import checkpoint
import training_utils
import training_steps
from fid import FID

import logging

logger = logging.getLogger(__name__)


def tree_shape(item):
    return jax.tree_map(lambda c: c.shape, item)


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
                                                    batch_size=config.batch_size)

    # --------------------------------------
    # Seeding and Precision
    # --------------------------------------
    rng = jax.random.PRNGKey(config.random_seed)

    if config.mixed_precision:
        dtype = jnp.float16
    elif config.bf16:
        dtype = jnp.bfloat16
    else:
        dtype = jnp.float32
    logger.info(f'Running on dtype {dtype}')

    platform = jax.local_devices()[0].platform
    if config.mixed_precision and platform == 'gpu':
        dynamic_scale_G_main = dynamic_scale_lib.DynamicScale()
        dynamic_scale_D_main = dynamic_scale_lib.DynamicScale()
        dynamic_scale_G_reg = dynamic_scale_lib.DynamicScale()
        dynamic_scale_D_reg = dynamic_scale_lib.DynamicScale()
        clip_conv = 256
        num_fp16_res = 4
    else:
        dynamic_scale_G_main = None
        dynamic_scale_D_main = None
        dynamic_scale_G_reg = None
        dynamic_scale_D_reg = None
        clip_conv = None
        num_fp16_res = 0

    # --------------------------------------
    # Initialize Models
    # --------------------------------------
    logger.info('Initialize models...')

    rng, init_rng = jax.random.split(rng)

    # Generator initialization for training
    start_mn = time.time()
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

    logger.info(f"MappingNetwork took {time.time() - start_mn:.2f}s")

    logger.info("Creating SynthesisNetwork...")
    start_sn = time.time()
    synthesis_net = stylegan2.SynthesisNetwork(resolution=config.resolution,
                                               num_channels=config.img_channels,
                                               w_dim=config.w_dim,
                                               fmap_base=config.fmap_base,
                                               num_fp16_res=num_fp16_res,
                                               clip_conv=clip_conv,
                                               dtype=dtype)

    synthesis_net_vars = synthesis_net.init(init_rng,
                                            jnp.ones((1, mapping_net.num_ws, config.w_dim)))
    synthesis_net_params, noise_consts = synthesis_net_vars['params'], synthesis_net_vars['noise_consts']

    logger.info(f"SynthesisNetwork took {time.time() - start_sn:.2f}s")

    params_G = frozen_dict.FrozenDict(
        {'mapping': mapping_net_params,
         'synthesis': synthesis_net_params}
    )

    # Discriminator initialization for training
    logger.info("Creating Discriminator...")
    start_d = time.time()
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
    logger.info(f"Discriminator took {time.time() - start_d:.2f}s")

    # Exponential average Generator initialization
    logger.info("Creating Generator EMA...")
    start_g = time.time()
    generator_ema = stylegan2.Generator(resolution=config.resolution,
                                        num_channels=config.img_channels,
                                        z_dim=config.z_dim,
                                        c_dim=config.c_dim,
                                        w_dim=config.w_dim,
                                        num_ws=int(np.log2(config.resolution)) * 2 - 3,
                                        num_mapping_layers=8,
                                        fmap_base=config.fmap_base,
                                        num_fp16_res=num_fp16_res,
                                        clip_conv=clip_conv,
                                        dtype=dtype)

    params_ema_G = generator_ema.init(init_rng,
                                      jnp.ones((1, config.z_dim)),
                                      jnp.ones((1, config.c_dim)))
    logger.info(f"Took {time.time() - start_g:.2f}s")

    # --------------------------------------
    # Initialize States and Optimizers
    # --------------------------------------
    logger.info('Initialize states...')
    tx_G = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)
    tx_D = optax.adam(learning_rate=config.learning_rate, b1=0.0, b2=0.99)

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

    # Running mean of path length for path length regularization
    pl_mean = jnp.zeros((), dtype=dtype)

    step = 0
    epoch_offset = 0
    best_fid_score = np.inf
    ckpt_path = None

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
    elif config.load_from_pkl is not None:
        # Load checkpoint and start new run
        ckpt_path = config.load_from_pkl
        logger.info(f'Load model state from from : {ckpt_path}')
        ckpt = checkpoint.load_checkpoint(ckpt_path)
        pl_mean = ckpt['pl_mean']
        state_G = ckpt['state_G']
        state_D = ckpt['state_D']
        params_ema_G = ckpt['params_ema_G']

    # Replicate states across devices
    pl_mean = flax.jax_utils.replicate(pl_mean)
    state_G = flax.jax_utils.replicate(state_G)
    state_D = flax.jax_utils.replicate(state_D)

    # --------------------------------------
    # Precompile train and eval steps
    # --------------------------------------
    logger.info('Precompile training steps...')
    p_main_step_G = jax.pmap(training_steps.main_step_G, axis_name='batch')
    p_regul_step_G = jax.pmap(functools.partial(training_steps.regul_step_G, config=config), axis_name='batch')

    p_main_step_D = jax.pmap(training_steps.main_step_D, axis_name='batch')
    p_regul_step_D = jax.pmap(functools.partial(training_steps.regul_step_D, config=config), axis_name='batch')

    # --------------------------------------
    # Training
    # --------------------------------------
    logger.info('Start training...')
    fid_metric = FID(generator_ema, ds_train, config)

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
        wandb.config.vm_name = training_utils.get_vm_name()

    for epoch in range(epoch_offset, config.num_epochs):
        if config.wandb and jax.process_index() == 0:
            wandb.log({'training/epochs': epoch}, step=step)

        for batch in data_pipeline.prefetch(ds_train, config.num_prefetch):
            assert batch['image'].shape[1] == config.batch_size, f"Mismatched batch (batch size: {config.batch_size}, this batch: {batch['image'].shape[1]})"

            # pbar.update(num_devices * config.batch_size)
            iteration_start_time = time.time()

            if config.c_dim == 0:
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
            time_d_start = time.time()
            state_D, metrics = p_main_step_D(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            time_d_end = time.time()
            if step % config.D_reg_interval == 0:
                state_D, metrics = p_regul_step_D(state_D, batch, metrics)

            # --------------------------------------
            # Update Generator
            # --------------------------------------
            time_g_start = time.time()
            state_G, metrics = p_main_step_G(state_G, state_D, batch, z_latent1, z_latent2, metrics, mixing_prob, rkey)
            if step % config.G_reg_interval == 0:
                H, W = batch['image'].shape[-3], batch['image'].shape[-2]
                rng, key = jax.random.split(rng)
                pl_noise = jax.random.normal(key, batch['image'].shape, dtype=dtype) / np.sqrt(H * W)
                state_G, metrics, pl_mean = p_regul_step_G(state_G, batch, z_latent1, pl_noise, pl_mean, metrics,
                                                           rng=rkey)

            params_ema_G = training_utils.update_generator_ema(flax.jax_utils.unreplicate(state_G),
                                                               params_ema_G,
                                                               config)
            time_g_end = time.time()

            # --------------------------------------
            # Logging and Checkpointing
            # --------------------------------------
            if step % config.save_every == 0 and config.disable_fid:
                # If FID evaluation is disabled, a checkpoint will be saved every 'save_every' steps.
                if jax.process_index() == 0:
                    logger.info('Saving checkpoint...')
                    checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step,
                                               epoch)

            num_imgs_processed += num_devices * config.batch_size
            if step % config.eval_fid_every == 0 and not config.disable_fid:
                # If FID evaluation is enabled, only save a checkpoint if FID score is better.
                if jax.process_index() == 0:
                    logger.info('Computing FID...')
                    fid_score = fid_metric.compute_fid(params_ema_G).item()
                    if config.wandb:
                        wandb.log({'training/gen/fid': fid_score}, step=step)
                    logger.info(f'Computed FID: {fid_score:.2f}')
                    if fid_score < best_fid_score:
                        best_fid_score = fid_score
                        logger.info(f'New best FID score ({best_fid_score:.3f}). Saving checkpoint...')
                        ts = time.time()
                        checkpoint.save_checkpoint(config.ckpt_dir, state_G, state_D, params_ema_G, pl_mean, config, step, epoch, fid_score=fid_score)
                        te = time.time()
                        logger.info(f'... successfully saved checkpoint in {(te-ts)/60:.1f}min')

            sec_per_kimg = (time.time() - iteration_start_time) / (num_devices * config.batch_size / 1000.0)
            time_taken_g = time_g_end - time_g_start
            time_taken_d = time_d_end - time_d_start
            time_taken_per_step = time.time() - iteration_start_time
            g_loss = jnp.mean(metrics['G_loss']).item()
            d_loss = jnp.mean(metrics['D_loss']).item()

            if config.wandb and jax.process_index() == 0:
                # wandb logging - happens every step
                wandb.log({'training/gen/loss': jnp.mean(metrics['G_loss']).item()}, step=step, commit=False)
                wandb.log({'training/dis/loss': jnp.mean(metrics['D_loss']).item()}, step=step, commit=False)
                wandb.log({'training/dis/fake_logits': jnp.mean(metrics['fake_logits']).item()}, step=step, commit=False)
                wandb.log({'training/dis/real_logits': jnp.mean(metrics['real_logits']).item()}, step=step, commit=False)
                wandb.log({'training/time_taken_g': time_taken_g, 'training/time_taken_d': time_taken_d}, step=step, commit=False)
                wandb.log({'training/time_taken_per_step': time_taken_per_step}, step=step, commit=False)
                wandb.log({'training/num_imgs_trained': num_imgs_processed}, step=step, commit=False)
                wandb.log({'training/sec_per_kimg': sec_per_kimg}, step=step)

            if step % config.log_every == 0:
                # console logging - happens every log_every steps
                logger.info(f'Total steps: {step:>6,} - epoch {epoch:>3,}/{config.num_epochs} @ {step % num_steps_per_epoch:>6,}/{num_steps_per_epoch:,} - G loss: {g_loss:.5f} - D loss: {d_loss:.5f} - sec/kimg: {sec_per_kimg:.2f}s - time per step: {time_taken_per_step:.3f}s')

            if step % config.generate_samples_every == 0 and config.wandb and jax.process_index() == 0:
                # Generate training images
                train_snapshot = training_utils.get_training_snapshot(
                    image_real=flax.jax_utils.unreplicate(batch['image']),
                    image_gen=flax.jax_utils.unreplicate(metrics['image_gen']),
                    max_num=10
                )
                wandb.log({'training/snapshot': wandb.Image(train_snapshot)}, commit=False, step=step)

                # Generate evaluation images
                labels = None if config.c_dim == 0 else batch['label'][0]
                image_gen_eval = training_steps.eval_step_G(
                    generator_ema, params=params_ema_G,
                    z_latent=z_latent1[0],
                    labels=labels,
                    truncation=1
                )
                image_gen_eval_trunc = training_steps.eval_step_G(
                    generator_ema,
                    params=params_ema_G,
                    z_latent=z_latent1[0],
                    labels=labels,
                    truncation=0.5
                )
                eval_snapshot = training_utils.get_eval_snapshot(image=image_gen_eval, max_num=10)
                eval_snapshot_trunc = training_utils.get_eval_snapshot(image=image_gen_eval_trunc, max_num=10)
                wandb.log({'eval/snapshot': wandb.Image(eval_snapshot)}, commit=False, step=step)
                wandb.log({'eval/snapshot_trunc': wandb.Image(eval_snapshot_trunc)}, step=step)

            step += 1

        # Sync moving stats across devices
        state_G = training_utils.sync_moving_stats(state_G)

        # Sync moving average of path length mean (Generator regularization)
        pl_mean = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='batch'), axis_name='batch')(pl_mean)
