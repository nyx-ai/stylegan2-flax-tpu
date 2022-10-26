import argparse
import os
import jax
import wandb
import training
import logging
import json
import gcs


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s', force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of tfrecord files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where checkpoints will be written to. A subfolder with the run ID will be created.')
    parser.add_argument('--load_from_ckpt', type=str, help='If provided, start training from an existing checkpoint.')
    parser.add_argument('--resume_run_id', type=str, help='If provided, resume existing training run. If --wandb is enabled W&B will also resume.')
    parser.add_argument('--project', type=str, default='sg2-flax', help='Name of this project.')
    # Training
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--num_steps', type=int, default=1_000_000_000, help='Number of step to run.')
    parser.add_argument('--early_stopping_after_steps', type=int, default=None, help='Early stopping after n steps without FID improvement (by default disabled).')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='(Per-device) Batch size.')
    parser.add_argument('--num_prefetch', type=int, default=2, help='Number of prefetched examples for the data pipeline.')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution. Must be a multiple of 2.')
    parser.add_argument('--allow_resolution_mismatch', action='store_true', help='Allow mismatch of requested resolution (--resolution) and dataset resolution. By default raises error.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    # Generator
    parser.add_argument('--fmap_base', type=int, default=16384, help='Overall multiplier for the number of feature maps.')
    parser.add_argument('--fmap_max', type=int, default=512, help='Maximum number of feature maps in any layer.')
    parser.add_argument('--freeze_g', type=int, default=None, help='Freeze n lowest layers of the generator. Disabled by default. Has to be <= log2(resolution)-1')
    # Discriminator
    parser.add_argument('--mbstd_group_size', type=int, help='Group size for the minibatch standard deviation layer, None = entire minibatch.')
    parser.add_argument('--freeze_d', type=int, default=None, help='Freeze n lowest layers of the discriminator. Disabled by default. Has to be <= log2(resolution)-2')
    # Exponentially Moving Average of Generator Weights
    parser.add_argument('--ema_kimg', type=float, default=20.0, help='Controls the ema of the generator weights (larger value -> larger beta).')
    # Losses
    parser.add_argument('--pl_decay', type=float, default=0.01, help='Exponentially decay for mean of path length (Path length regul).')
    parser.add_argument('--pl_weight', type=float, default=2, help='Weight for path length regularization.')
    # Regularization
    parser.add_argument('--mixing_prob', type=float, default=0.9, help='Probability for style mixing.')
    parser.add_argument('--G_reg_interval', type=int, default=4, help='How often to perform regularization for G. Disable if negative.')
    parser.add_argument('--D_reg_interval', type=int, default=16, help='How often to perform regularization for D. Disable if negative.')
    parser.add_argument('--r1_gamma', type=float, default=10.0, help='Weight for R1 regularization.')
    # Model
    parser.add_argument('--z_dim', type=int, default=512, help='Input latent (Z) dimensionality.')
    parser.add_argument('--c_dim', type=int, default=0, help='Conditioning label (C) dimensionality, 0 = no label.')
    parser.add_argument('--w_dim', type=int, default=512, help='Conditioning label (W) dimensionality.')
    # Logging
    parser.add_argument('--log_every', type=int, default=100, help='Log every log_every steps.')
    parser.add_argument('--generate_snapshot_every', type=int, default=10000, help='Generate eval and training snapshots every generate_snapshot_every steps. Disable if negative.')
    parser.add_argument('--num_samples_per_eval_snapshot', type=int, default=16, help='Number of eval images that will be generated as part of a snapshot')
    # Checkpointing
    parser.add_argument('--save_every', type=int, default=10_000, help='Save every save_every steps. Set to negative number in order to disable. This is indepedent of FID-based "best" checkpoints.')
    parser.add_argument('--keep_n_best_checkpoints', type=int, default=2, help='Keep best n checkpoints based on FID.')
    # FID
    parser.add_argument('--eval_fid_every', type=int, default=1000, help='Compute FID score every eval_fid_every steps. Disable if negative.')
    parser.add_argument('--eval_fid_trunc', action='store_true', help='Also compute truncated FID')
    parser.add_argument('--num_fid_images_real', type=int, default=100000, help='Number of images (overall or per class if training class conditional) to use from dataset for FID computation.')
    parser.add_argument('--num_fid_images_fake', type=int, default=10000, help='Number of images (overall or per class if training class conditional) to generate with model for FID computation.')
    parser.add_argument('--metric_cache_location', help='Location where metric cache is stored. By default writes to /tmp. Recommended to be set to a GCS bucket!')
    # W&B
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&Biases.')
    parser.add_argument('--name', type=str, default=None, help='Name of this experiment in Weights&Biases.')
    parser.add_argument('--entity', type=str, default=None, help='Entity for this experiment in Weights&Biases.')
    parser.add_argument('--group', type=str, default=None, help='Group name of this experiment for Weights&Biases.')
    # Debug
    parser.add_argument('--debug', action='store_true', help='Show debug log.')
    parser.add_argument('--profile', action='store_true', help='Run Jax profiler. Run `tensorboard --logdir ./profile` in separate terminal.')
    parser.add_argument('--disable_memory_profile', action='store_true', help='Disable device memory profiling')

    args = parser.parse_args()

    # debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # suppress INFO logs by absl logger (leads to logspam for jax caching)
        absl_logger = logging.getLogger('absl')
        absl_logger.setLevel(logging.WARNING)

    # some validation
    if args.resume_run_id is not None:
        assert args.load_from_ckpt is None, 'When resuming a run one cannot also specify --load_from_ckpt'
    if args.save_dir.startswith('gs://'):
        assert gcs.validate_dir_is_in_current_region(args.save_dir)  # do not load data from different region

    # set unique Run ID
    if args.resume_run_id:
        resume = 'must'  # throw error if cannot find id to be resumed
        args.run_id = args.resume_run_id
    else:
        resume = None  # default
        args.run_id = wandb.util.generate_id()
    args.ckpt_dir = os.path.join(args.save_dir, args.run_id)

    if jax.process_index() == 0:
        if not args.ckpt_dir.startswith('gs://') and not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if args.wandb:
            wandb.init(id=args.run_id,
                       project=args.project,
                       group=args.group,
                       config=args,
                       name=args.name,
                       entity=args.entity,
                       resume=resume)
        logger.info('Starting new run with config:')
        print(json.dumps(vars(args), indent=4))

    training.train_and_evaluate(args)


if __name__ == '__main__':
    main()
