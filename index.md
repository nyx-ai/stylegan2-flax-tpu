## This Food Does Not Exist 🍪🍰🍣🍹

`cookie-256.pkl`

![cookies](https://user-images.githubusercontent.com/140592/179369671-32cf8c67-a3d5-43a4-a200-1ba91e736ae2.png)

`cheesecake-256.pkl`

![cheesecake](https://user-images.githubusercontent.com/140592/179959973-df75351d-db07-4ff9-8f9f-97334bab20a8.png)

`cocktail-256.pkl`

![cocktail](https://user-images.githubusercontent.com/140592/179956003-8db513d2-b0b1-4a1f-8f15-827b56bedb25.png)

`sushi-256.pkl`

![sushi](https://user-images.githubusercontent.com/140592/179958220-45324fe7-90d8-49dd-94be-877b03201160.png)

Cherry-picked results, check out the Colab notebook to generate your own: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nyx-ai/stylegan2-flax-tpu/blob/master/notebook/image_generation.ipynb)

Or train your own model: [https://github.com/nyx-ai/stylegan2-flax-tpu](https://github.com/nyx-ai/stylegan2-flax-tpu)

## Why not diffusion models? 🤔

Diffusion models are all the rage these days: [DALL·E 2](https://openai.com/dall-e-2/), [Craiyon](https://www.craiyon.com/) (formerly DALL·E mini), [ruDALL-E](https://rudalle.ru/en/)... Why not go in this direction?

### Realism vs control

StyleGAN models shine in terms of photorealism, as can be some by some of our food results. For another example, the website [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/) produces very believable face images. While GANs are still better at this, [diffusion models are catching up](https://arxiv.org/abs/2105.05233) and this may change soon.

Diffusion models offer better control and flexibility, thanks in large part to text guidance. This comes at the cost of larger models and slower generation times.

### Training resources

We were able to train the provided models in less than 10h each using a single TPU v4-8:

![Training plots](static/fid_pretrained.png)

In comparison, Craiyon is being training on a v3-256 TPU pod which means 32x the resources (albeit using the previous TPU generation) and the training [has been going on for over a month](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega-Training-Journal--VmlldzoxODMxMDI2)!

## Acknowledgements 🙏

* This work is based on Matthias Wright's [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2) implementation.
* The project received generous support from Google's TPU Research Cloud (TRC).
