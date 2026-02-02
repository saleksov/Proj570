#
#
#
# Created by anonymus student for 570 project 
# heavily based on OpenAI Guided Diffusion
#
#

import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from utils import utils_image as util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def load_image(path, image_size):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

def save_image(tensor, path):
    tensor = tensor.clamp(0, 1).detach().cpu()
    img = T.ToPILImage()(tensor)
    img.save(path)

# CHANGE THE ADRESSES BELLOW

def main():
    parser = argparse.ArgumentParser()
    defaults = model_and_diffusion_defaults()
    defaults.update({
        "image_size": 256,
        "num_channels": 128,
        "num_res_blocks": 2, # 2 or 3, depends on model
        "image_cond": True,
        "model_path": r"E:\PythonTest\Code\DiffPIR\models\model_final_2.pt", # CHANGE ME
        "input_dark_dir": r"E:\PythonTest\Code\DiffPIR\testsets\demo_test",     # CHANGE ME
        "output_dir": r"E:\PythonTest\Code\DiffPIR\results\dark",               # CHANGE ME
        "timestep_respacing": "1000", #ddim25 "50" "100" "500" "1000" (best 1000, slow)
        "use_fp16": False,
        "class_cond": False,
        "batch_size": 1,
    })
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract model name from path (e.g., model_4000.pt → model_4000)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # Update output_dir to include model name
    args.output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    model_args = args_to_dict(args, set(model_and_diffusion_defaults().keys()) | {"image_cond"})
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(device)

    # Load model weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    image_paths = util.get_image_paths(args.input_dark_dir)
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.splitext(os.path.basename(image_path))[0]
        dark = load_image(image_path, args.image_size).unsqueeze(0).to(device)
        B = dark.size(0)
        init_bright = dark.clone()

        def model_fn(x, t):
            cond = dark.expand_as(x)
            return model(torch.cat([x, cond], dim=1), t)

        sample = diffusion.p_sample_loop(
            model_fn,
            (B, 3, args.image_size, args.image_size),
            device=device,
            progress=True,
        )

        output_path = os.path.join(args.output_dir, f"{filename}_bright.png")
        save_image(sample[0], output_path)

    print(f"Saved results to {args.output_dir}")

if __name__ == "__main__":
    main()
