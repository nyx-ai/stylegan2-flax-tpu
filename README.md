# StyleGAN2 Flax TPU


This implementation is adapted from the [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/flaxmodels/stylegan2) codebase by [Matthias Wright](https://github.com/matthias-wright).

Specifically, the features we've added allow for better scaling of [StyleGAN2](https://arxiv.org/abs/1912.04958) training on TPUs:
* 🏭 Enable data-parallel training on TPU pods (tested on TPU v2 to v4 generations)
* 💾 Google Cloud Storage (GCS) integration/dataset sharding between workers
* 🏖 Quality-of-life improvements (e.g. improved W&B logging)

**[This food does not exist! Click to see more samples 🍪🍰🍣🍹🍔](https://nyx-ai.github.io/stylegan2-flax-tpu/)**

![This food does not exist](https://user-images.githubusercontent.com/140592/198327038-b73c1a5c-297b-43c8-8638-8191fe961802.png)

## 🏗 Changelog
<details>
  <summary>v0.2</summary>
  
* Better support for class-conditional training, adding per-class moving average statistics to generator
* Training data can now be split into multiple tfrecord files (can be either in data_dir or in a subdirectory `tfrecords`). Still requires `dataset_info.json` in `--data_dir` location (containing `width`, `heigh`, `num_examples`, and list of `classes` if class-conditional). 
* Renaming arg `--load_from_pkl` => `--load_from_ckpt`
* Added `--num_steps` argument to specify a fixed number of steps to run
* Added `--early_stopping_after_steps` argument to stop after n steps of no FID improvement
* Removal of `--bf16` flag and consolidation with `--mixed_precision`. 
* Allow layer freezing with `--freeze_g` and `--freeze_d` arguments
* Add `--fmap_max` argument, in order to have better control over feature map dimensions
* Allow disabling of generator and discriminator regularization
* Change checkpointing behaviour from saving every 2k steps to saving every 10k steps and keeping 2 best checkpoints (see `--save_every` and `--keep_n_checkpoints`)
* Add `--metric_cache_location` in order to cache dataset statistics (currently for FID only)
* Log TPU memory usage, shoutout to ayaka14732 for help (see also https://github.com/ayaka14732/jax-smi)
* Visualise model architecture & parameters on startup
* Improve W&B logging (e.g. adding eval snapshots with fixed latents)
* Experimental: Add jax profiling
  
</details>
<details>
  <summary>v0.1</summary>
  
  * Enable training on TPUs
  * Google Cloud Storage (GCS) integration
  * Several quality-of-life improvements
  
</details>

## 🏗 Changelog
<details>
  <summary>v0.2</summary>
  
* Better support for class-conditional training, adding per-class moving average statistics to generator
* Training data can now be split into multiple tfrecord files (can be either in `--data_dir` or in a subdirectory `tfrecords`). Still requires `dataset_info.json` in `--data_dir` location (containing `width`, `heigh`, `num_examples`, and list of `classes` if class-conditional). 
* Renaming arg `--load_from_pkl` => `--load_from_ckpt`
* Added `--num_steps` argument to specify a fixed number of steps to run
* Added `--early_stopping_after_steps` argument to stop after n steps of no FID improvement
* Removal of `--bf16` flag and consolidation with `--mixed_precision`. 
* Allow layer freezing with `--freeze_g` and `--freeze_d` arguments
* Add `--fmap_max` argument, in order to have better control over feature map dimensions
* Allow disabling of generator and discriminator regularization
* Change checkpointing behaviour from saving every 2k steps to saving every 10k steps and keeping 2 best checkpoints (see `--save_every` and `--keep_n_checkpoints`)
* Add `--metric_cache_location` in order to cache dataset statistics (currently for FID only)
* Log TPU memory usage, shoutout to ayaka14732 for help (see also https://github.com/ayaka14732/jax-smi)
* Visualise model architecture & parameters on startup
* Improve W&B logging (e.g. adding eval snapshots with fixed latents)
* Experimental: Add jax profiling
  
</details>
<details>
  <summary>v0.1</summary>
  
  * Enable training on TPUs
  * Google Cloud Storage (GCS) integration
  * Several quality-of-life improvements
  
</details>

## 🧑‍🔧 Install
1. Clone the repository:
   ```sh
   git clone https://github.com/nyx-ai/stylegan2-flax-tpu.git
   ```
2. Go into the directory:
   ```sh
   cd stylegan2-flax-tpu
   ```
3. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```

## 🖼 Generate Images

We released four 256x256 as well as 512x512 models. Download them from the [latest release](https://github.com/nyx-ai/stylegan2-flax-tpu/releases).

```
python generate_images.py \
   --checkpoint checkpoints/cookie-256.pkl \
   --seeds 0 42 420 666 \
   --truncation_psi 0.7 \
   --out_path generated_images
```

Check the Colab notebook for more examples: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nyx-ai/stylegan2-flax-tpu/blob/master/notebook/image_generation.ipynb)


## ⚙️ Train Custom Models
Add your images into a folder `/path/to/image_dir`:
```
/path/to/image_dir/
    0.jpg
    1.jpg
    2.jpg
    4.jpg
    ...
```
and create a TFRecord dataset:
```sh
python dataset_utils/images_to_tfrecords.py --image_dir /path/to/image_dir/ --data_dir /path/to/tfrecord
```
For more detailed instructions please refer to [this README](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2#preparing-datasets-for-training).

The following command trains with 128 resolution and batch size of 8.
```sh
python main.py --data_dir /path/to/tfrecord
```
Read more about suitable training parameters [here](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2#training).

Our experiments have been run and tested on TPU VMs (generation v2 to v4). At the time of writing Colab is offering an older generation of TPUs. Therefore training (and especially compilation) may be significantly slower. If you still wish to train on Colab, the following may get you started: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KyJFofaA_SRzIYC4zs2mtQ790KpntdXL?usp=sharing)

## 🙏 Acknowledgements
* This work is based on Matthias Wright's [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2) implementation.
* The project received generous support from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
* The image datasets were built using the [LAION5B index](https://laion.ai/blog/laion-5b/)
* We are grateful to [Weights & Biases](https://wandb.ai/) for preserving our sanity
