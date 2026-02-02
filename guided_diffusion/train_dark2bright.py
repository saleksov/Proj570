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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# Import functions from the guided-diffusion repo
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# Import diffusion functions from gaussian_diffusion.py
import guided_diffusion.gaussian_diffusion as gd

class DarkBrightPairDataset(Dataset):
    """
    Dataset that returns a pair:
       (bright_image, dark_image)
    Both images are assumed to be .png, RGB, and will be resized to a fixed image size.
    """
    def __init__(self, bright_dir, dark_dir, image_size=256):
        self.bright_paths = sorted([
            os.path.join(bright_dir, f)
            for f in os.listdir(bright_dir)
            if f.lower().endswith('.png')
        ])
        self.dark_paths = sorted([
            os.path.join(dark_dir, f)
            for f in os.listdir(dark_dir)
            if f.lower().endswith('.png')
        ])
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.bright_paths)

    def __getitem__(self, idx):
        bright = Image.open(self.bright_paths[idx]).convert("RGB")
        dark = Image.open(self.dark_paths[idx]).convert("RGB")
        bright = self.transform(bright)
        dark = self.transform(dark)
        return bright, dark


def main():
    parser = argparse.ArgumentParser()
    # Load defaults from guided diffusion defaults; add custom flags.
    defaults = model_and_diffusion_defaults()
    defaults.update({
        "data_dir_bright": "",  # Path to folder containing bright images
        "data_dir_dark": "",    # Path to folder containing corresponding dark images
        "iterations": 300000,
        "batch_size": 8,
        "lr": 1e-4,
        "log_interval": 100,
        "save_interval": 10000,
        "image_size": 256,
        "image_cond": True,   # Flag to build model for conditional (dark->bright)
        "use_fp16": False,
    })
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and diffusion using flags.
    
    # take only the keys that are in the defaults plus "image_cond"
    model_args = args_to_dict(args, set(model_and_diffusion_defaults().keys()) | {"image_cond"})
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    
    # Prepare paired dataset.
    dataset = DarkBrightPairDataset(
        bright_dir=args.data_dir_bright,
        dark_dir=args.data_dir_dark,
        image_size=args.image_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    global_step = 0
    model.train()

    # Training loop
    while global_step < args.iterations:
        for bright, dark in dataloader:
            # 'bright' is the target (clean) image; 'dark' is the condition.
            bright = bright.to(device)  # Shape: [B, 3, H, W]
            dark = dark.to(device)      # Shape: [B, 3, H, W]
            B = bright.size(0)
            
            # Sample a random timestep t for each image in the batch.
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            
            # Sample noise with same shape as bright image.
            noise = torch.randn_like(bright)
            
            # Use the forward diffusion process to get a noisy image x_t.
            # x_t = q_sample(x0, t, noise)
            x_t = diffusion.q_sample(bright, t, noise)
            
            # Prepare the model input: concatenate the noisy bright image and the dark conditioning image.
            # This yields an input of shape [B, 6, H, W].
            model_input = torch.cat([x_t, dark], dim=1)
            
            # The model is trained to predict the noise added.
            predicted_noise = model(model_input, t)
            loss = mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if global_step % args.log_interval == 0:
                print(f"Step {global_step}: Loss = {loss.item():.6f}")
            if global_step % args.save_interval == 0 and global_step > 0:
                ckpt_dir = "/content/drive/MyDrive/diffpir_dark/checkpoints" # 570 save in drive
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"model_{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved at step {global_step}")
            
            global_step += 1
            if global_step >= args.iterations:
                break

    # Save the final model
    os.makedirs("/content/drive/MyDrive/diffpir_dark/checkpoints", exist_ok=True)
    final_ckpt = os.path.join("/content/drive/MyDrive/diffpir_dark/checkpoints", "model_final.pt")
    torch.save(model.state_dict(), final_ckpt)
    print("Training complete! Final model saved.")


if __name__ == "__main__":
    main()
