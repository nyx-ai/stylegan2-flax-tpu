# StyleGAN2 Flax TPU
This implementation is adapted from the [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/flaxmodels/stylegan2) codebase by [Matthias Wright](https://github.com/matthias-wright).

Specifically, the features we've added allow for better scaling of training on TPUs:
* 🏭 Enable training on TPU pods (tested on TPU v4 generation)
* 💾 Google Cloud Storage integration/dataset sharding
* 🏖 Quality-of-life improvements (e.g. better W&B logging)


## Install
1. Clone the repository:
   ```sh
   git clone https://github.com/nyx-ai/stylegan2-flax-tpu.git
   ```
2. Go into the directory:
   ```sh
   cd stylegan2-flax-tpu
   ```
3. Install [Jax](https://github.com/google/jax#installation) accoring to your platform.
4. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```

## Preparing Datasets
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
For more detailed instructions check out [this README](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2#preparing-datasets-for-training) from Matthias Wright's repository.

## Train
The following command trains with 128 resolution and batch size of 8.
```sh
python main.py --data_dir /path/to/tfrecord
```

## 🙏 Acknowledgements
* This work is based on Matthias Wright's [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2) implementation.
* The project was supported with generous quota from Google's TPU Research Cloud (TRC).
