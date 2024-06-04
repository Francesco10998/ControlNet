# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image
from matplotlib import cm

# download an image
"""image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)"""
image = Image.open("74.png")
#image.save("originale.png")
np_image = np.array(image)
np_image = cv2.resize(np_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
image = Image.fromarray(np_image.astype('uint8'), 'RGB')

counterfactual = Image.open("75.png")
np_counterfactual = np.array(counterfactual)
np_counterfactual = cv2.resize(np_counterfactual, (512, 512), interpolation=cv2.INTER_LANCZOS4)
# get canny image
np_image = cv2.Canny(np_counterfactual, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",safety_checker = None, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(134344)
image = pipe(
    "face of a male",
    num_inference_steps=50,
    generator=generator,
    image=image,
    control_image=canny_image,
).images[0]

image.save("condizionata.png")
