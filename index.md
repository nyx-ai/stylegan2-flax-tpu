## This Food Does Not Exist 🍪🍰🍣🍹🍔

**We trained StyleGAN2 models to generate food pictures. The images below are all synthetic!**

This work is done in partnership with the [Food & You](https://www.foodandyou.org/) project by the [Digital Epidemiology Lab](https://www.digitalepidemiologylab.org/) at [EPFL](https://www.epfl.ch/en/). In this context, we are researching the potential of synthetic data augmention for vision tasks.

This research is part of the technology underlying our AI-generated photography platform [Nyx.gallery](https://nyx.gallery/). You can also follow our work on [🐦 Twitter](https://twitter.com/NyxAI_Lab).

The [code](https://github.com/nyx-ai/stylegan2-flax-tpu) optimized for TPU training as well as the [pretrained models](https://github.com/nyx-ai/stylegan2-flax-tpu/releases) are openly available.

<iframe src="https://ghbtns.com/github-btn.html?user=nyx-ai&repo=stylegan2-flax-tpu&type=star&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>

## Multi-class 512x512 model 🆕

**[Release v0.2](https://github.com/nyx-ai/stylegan2-flax-tpu/releases/tag/v0.2), October 2022**

![v0.2 model samples](https://user-images.githubusercontent.com/140592/198279827-f50459f3-9d31-47ce-9380-5e535c612700.png)

`food-512.pkl`

We have released a new and much improved model:

- Single 5-class model (burger/cheesecake/cocktail/cookie/sushi) instead of 1-class models
- Resolution of 512x512 instead of 256x256
- Trained for much longer: 8 days at 256x256 then 28 days at 512x512 instead of 10 hours
- Trained on more data: 558k 512x512 images instead of 100k 256x256 images

🍒 The sample above are cherry-picked: check out the [Colab notebook](https://colab.research.google.com/github/nyx-ai/stylegan2-flax-tpu/blob/master/notebook/image_generation.ipynb) to generate your own, or [train your own model](https://github.com/nyx-ai/stylegan2-flax-tpu).


## Single-class 256x256 models

**[Release v0.1](https://github.com/nyx-ai/stylegan2-flax-tpu/releases/tag/v0.1), July 2022**

The models below were released in July 2022. Each model was trained on a single food class: cookie, cheescake, cocktail and sushi. 
They can still be used with the v0.2 code.

![cookies](https://user-images.githubusercontent.com/140592/179369671-32cf8c67-a3d5-43a4-a200-1ba91e736ae2.png)

`cookie-256.pkl`

![cheesecake](https://user-images.githubusercontent.com/140592/179959973-df75351d-db07-4ff9-8f9f-97334bab20a8.png)

`cheesecake-256.pkl`

![cocktail](https://user-images.githubusercontent.com/140592/179956003-8db513d2-b0b1-4a1f-8f15-827b56bedb25.png)

`cocktail-256.pkl`

![sushi](https://user-images.githubusercontent.com/140592/179958220-45324fe7-90d8-49dd-94be-877b03201160.png)

`sushi-256.pkl`

## Why not DALL·E/diffusion models? 🤔

Recent methods like diffusion and auto-regressive models are all the rage these days: [DALL·E 2](https://openai.com/dall-e-2/), [Craiyon](https://www.craiyon.com/) (formerly DALL·E mini), [ruDALL-E](https://rudalle.ru/en/)... Why not go in this direction?

### Realism vs control

StyleGAN models shine in terms of photorealism, as can be some by some of our food results. For another example, the website [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com/) produces very believable face images. While GANs are still better at this, [diffusion models are catching up](https://arxiv.org/abs/2105.05233) and this may change soon.

Diffusion models offer better control and flexibility, thanks in large part to text guidance. This comes at the cost of larger models and slower generation times.

### Training resources

We were able to train the provided models in less than 10h each using a single TPU v4-8:

![Training plots](static/fid_pretrained.png)

[FID (Fréchet inception distance)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) is a metric used to assess the quality of images created by a generative model.

In comparison, Craiyon is being training on a v3-256 TPU pod which means 32x the resources (albeit using the previous TPU generation) and the training [has been going on for over a month](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-Mega-Training-Journal--VmlldzoxODMxMDI2).

### Result comparison

No cherry-picking!

Ours

![bdc76775-2c9f-4110-a2f1-fcbc07a588e7](https://user-images.githubusercontent.com/140592/179997085-53c0cb55-35ba-4333-b1fd-10df2ddbb238.png)

[Craiyon](https://www.craiyon.com/) ("a pile of cookies on a plate")

![a-pile-of-cookies-on-a-plate](https://user-images.githubusercontent.com/140592/179996330-6fe568d2-bc83-4755-b556-6059e1fdd231.jpeg)

[DALL·E 2](https://openai.com/dall-e-2/) ("a pile of cookies on a plate")

<img width="2320" alt="Screenshot 2022-07-20 at 15 31 55" src="https://user-images.githubusercontent.com/140592/179996470-66c56b77-8305-4b25-92c0-f9570970a7b3.png">


## Acknowledgements 🙏

* This work is based on Matthias Wright's [stylegan2](https://github.com/matthias-wright/flaxmodels/tree/main/training/stylegan2) implementation
* The project received generous support from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)
* The image datasets were built using the [LAION5B index](https://laion.ai/blog/laion-5b/)
* We are grateful to [Weights & Biases](https://wandb.ai/) for preserving our sanity

**Follow our Generative AI research: [📘 GitHub](https://github.com/nyx-ai) [🐦 Twitter](https://twitter.com/NyxAI_Lab) [📩 Newsletter](http://eepurl.com/ia0lfP) [👨‍💼 LinkedIn](https://www.linkedin.com/company/nyxai) [📷 Instagram](https://www.instagram.com/NyxAI_Lab)**
