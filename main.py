import argparse
import os
import jax
import wandb
import training
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s', force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset.')
    parser.add_argument('--save_dir', type=str, default='gs://ig-standard-usc1/sg2-flax/checkpoints/', help='Directory where checkpoints will be written to. A subfolder with run_id will be created.')
    parser.add_argument('--load_from_pkl', type=str, help='If provided, start training from an existing checkpoint pickle file.')
    parser.add_argument('--resume_run_id', type=str, help='If provided, resume existing training run. If --wandb is enabled W&B will also resume.')
    parser.add_argument('--project', type=str, default='sg2-flax', help='Name of this project.')
    # Training
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--num_prefetch', type=int, default=2, help='Number of prefetched examples for the data pipeline.')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution. Must be a multiple of 2.')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--bf16', action='store_true', help='Use bf16 dtype (This is still WIP).')
    # Generator
    parser.add_argument('--fmap_base', type=int, default=16384, help='Overall multiplier for the number of feature maps.')
    # Discriminator
    parser.add_argument('--mbstd_group_size', type=int, help='Group size for the minibatch standard deviation layer, None = entire minibatch.')
    # Exponentially Moving Average of Generator Weights
    parser.add_argument('--ema_kimg', type=float, default=20.0, help='Controls the ema of the generator weights (larger value -> larger beta).')
    # Losses
    parser.add_argument('--pl_decay', type=float, default=0.01, help='Exponentially decay for mean of path length (Path length regul).')
    parser.add_argument('--pl_weight', type=float, default=2, help='Weight for path length regularization.')
    # Regularization
    parser.add_argument('--mixing_prob', type=float, default=0.9, help='Probability for style mixing.')
    parser.add_argument('--G_reg_interval', type=int, default=4, help='How often to perform regularization for G.')
    parser.add_argument('--D_reg_interval', type=int, default=16, help='How often to perform regularization for D.')
    parser.add_argument('--r1_gamma', type=float, default=10.0, help='Weight for R1 regularization.')
    # Model
    parser.add_argument('--z_dim', type=int, default=512, help='Input latent (Z) dimensionality.')
    parser.add_argument('--c_dim', type=int, default=0, help='Conditioning label (C) dimensionality, 0 = no label.')
    parser.add_argument('--w_dim', type=int, default=512, help='Conditioning label (W) dimensionality.')
    # Logging
    parser.add_argument('--log_every', type=int, default=100, help='Log every log_every steps.')
    parser.add_argument('--save_every', type=int, default=2000, help='Save every save_every steps. Will be ignored if FID evaluation is enabled.')
    parser.add_argument('--generate_samples_every', type=int, default=10000, help='Generate samples every generate_samples_every steps.')
    parser.add_argument('--debug', action='store_true', help='Show debug log.')
    # FID
    parser.add_argument('--eval_fid_every', type=int, default=1000, help='Compute FID score every eval_fid_every steps.')
    parser.add_argument('--num_fid_images', type=int, default=10000, help='Number of images to use for FID computation.')
    parser.add_argument('--disable_fid', action='store_true', help='Disable FID evaluation.')
    # W&B
    parser.add_argument('--wandb', action='store_true', help='Log to Weights&Biases.')
    parser.add_argument('--name', type=str, default=None, help='Name of this experiment in Weights&Biases.')
    parser.add_argument('--entity', type=str, default='nyxai', help='Entity for this experiment in Weights&Biases.')
    parser.add_argument('--group', type=str, default=None, help='Group name of this experiment for Weights&Biases.')

    args = parser.parse_args()

    # debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # some validation
    if args.resume_run_id is not None:
        assert args.load_from_pkl is None, 'When resuming a run one cannot also specify --load_from_pkl'

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
