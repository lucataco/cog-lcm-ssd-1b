# latent-consistency/lcm-ssd-1b

This is an implementation of the [latent-consistency/lcm-ssd-1b](https://huggingface.co/latent-consistency/lcm-ssd-1b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="a close-up picture of an old man standing in the rain"

## Example:

"a close-up picture of an old man standing in the rain"

![alt text](output.0.png)
