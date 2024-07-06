import torch
import ssl
import urllib3
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# close https verification
ssl.create_default_context()

# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)
# sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# # ZoeD_K
# conf = get_config("zoedepth", "infer", config_version="kitti")
# model_zoe_k = build_model(conf)
# # sample prediction
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# zoe = model_zoe_k.to(DEVICE)

# # ZoeD_NK
# conf = get_config("zoedepth_nk", "infer")
# model_zoe_nk = build_model(conf)
# # sample prediction
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# zoe = model_zoe_nk.to(DEVICE)

# Local file
from PIL import Image

# local path
image = Image.open("/Users/ushiushi/PycharmProjects/ZoeDepth/assets/cat1.jpeg").convert("RGB")  # load

# # system path
# image = Image.open("/home/s_234712236/ZoeDepth/assets/cat1.jpeg").convert("RGB")  # load

depth = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor
from zoedepth.utils.misc import pil_to_batched_tensor

X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)

# # From URL
# from zoedepth.utils.misc import get_image_from_url
#
# # Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"
#
# image = get_image_from_url(URL)  # fetch
# depth = zoe.infer_pil(image)

# # Save raw
# from zoedepth.utils.misc import save_raw_16bit
#
# fpath = "/Users/ushiushi/PycharmProjects/ZoeDepth/assets/output.png"
# save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# # save path system
# fpath_colored = "/home/s_234712236/ZoeDepth/assets/output_colored_cat1_n.png"

# save colored output local
fpath_colored = "/Users/ushiushi/PycharmProjects/ZoeDepth/assets/output_colored_cat1_n.png"

Image.fromarray(colored).save(fpath_colored)
print("Successfully saved the colored image to the target path")
