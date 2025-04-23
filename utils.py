import numpy as np
import torch
import imageio

def rgb_to_gray_tensor(rgb_images: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy RGB image or batch of images to a PyTorch grayscale tensor.
    Input: (H, W, 3) or (B, H, W, 3)
    Output: (B, 1, H, W)
    """
    if rgb_images.ndim == 3:
        # Single image case, add batch dimension
        rgb_images = np.expand_dims(rgb_images, axis=0)

    # rgb_images: (B, H, W, 3)
    B, H, W, _ = rgb_images.shape

    # Convert RGB to grayscale
    gray = np.dot(rgb_images[..., :3], [0.2989, 0.5870, 0.1140])  # (B, H, W)

    # Convert to torch tensor, add channel dimension
    gray_tensor = torch.from_numpy(gray.copy()).unsqueeze(1).float() / 255.0  # (B, 1, H, W)

    return gray_tensor

def save_vid(frames, episode):
    gif_path = f"episode_{episode}.gif"
    imageio.mimsave(gif_path, frames, fps=30)
