# added dependencies
# matplotlib.pypot
# wandb
# roboflow

from roboflow import Roboflow
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
import os, subprocess, sys, pathlib
from armor_unet.lit_module import ArmorUNet


rf = Roboflow(api_key="API-KEY")
project = rf.workspace("crowd-analysis-dataset").project("dataset_robomaster-jlvqw")
version = project.version(1)
dataset = version.download("coco")

# requires: roboflow
# import os, subprocess, sys, pathlib
REPO_URL = "https://github.com/GoldenPandaMRW/torch-lightning-with-ray.git"  # @param {type:"string"}
TARGET_DIR = "/content/torch-lightning-with-ray"  # @param {type:"string"}
BRANCH_NAME = "main"  # @param {type:"string"}
target_path = pathlib.Path(TARGET_DIR)
if not target_path.exists():
    subprocess.run(["git", "clone", REPO_URL, str(target_path)], check=True)
else:
    subprocess.run(["git", "-C", str(target_path), "fetch"], check=True)
    subprocess.run(["git", "-C", str(target_path), "reset", "--hard", "origin/" + BRANCH_NAME], check=True)
subprocess.run(["git", "-C", str(target_path), "checkout", BRANCH_NAME], check=True)
subprocess.run(["git", "-C", str(target_path), "pull", "origin", BRANCH_NAME], check=True)
os.chdir(target_path)
if str(target_path) not in sys.path:
    sys.path.insert(0, str(target_path))
print(f'Working directory: {target_path} (branch: {BRANCH_NAME})')

# import wandb
wandb.login()
wandb.init(project="armor_unet")

#@title Run training
from scripts.train import train_armor_detector
model, trainer, datamodule = train_armor_detector(
    data_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    learning_rate=LEARNING_RATE,
    base_channels=BASE_CHANNELS,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
)

# import os
# import torch
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch.nn.functional as F

# from armor_unet.lit_module import ArmorUNet


CHECKPOINT_PATH = "/content/checkpoints/armor-unet-epoch=04-val_dice=0.8103.ckpt"
IMAGE_PATH = "/content/images/thumb2.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
TARGET_SIZE = (640, 640)  # set to None to keep original size

def load_model(ckpt_path: str, device: str) -> ArmorUNet:
    model = ArmorUNet.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    return model

def preprocess_image(image_path: str, target_size=None):
    image = Image.open(image_path).convert("RGB")
    if target_size is not None:
        image = image.resize(target_size, Image.BILINEAR)

    arr = np.asarray(image).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return image, tensor

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))
    return padded, pad_h, pad_w

def infer_mask(model: ArmorUNet, tensor: torch.Tensor, threshold: float, device: str):
    tensor = tensor.to(device)
    tensor, pad_h, pad_w = pad_to_multiple(tensor)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    probs = probs[..., : probs.shape[-2] - pad_h or None, : probs.shape[-1] - pad_w or None]
    prob_map = probs[0, 0].cpu().numpy()
    mask = (prob_map > threshold).astype("float32")
    return prob_map, mask

def plot_result(image: Image.Image, prob_map: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(prob_map, cmap="viridis")
    plt.title("Probability Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, cmap="jet", alpha=0.4)
    plt.title("Thresholded Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    model = load_model(CHECKPOINT_PATH, DEVICE)
    image, tensor = preprocess_image(IMAGE_PATH, TARGET_SIZE)
    prob_map, mask = infer_mask(model, tensor, THRESHOLD, DEVICE)
    plot_result(image, prob_map, mask)

    # Save the probability map and binary mask
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    prob_img = Image.fromarray((prob_map * 255).astype("uint8"))
    prob_img.save(os.path.join(outputs_dir, "prob_map.png"))

    mask_img = Image.fromarray((mask * 255).astype("uint8"))
    mask_img.save(os.path.join(outputs_dir, "mask_binary.png"))

if __name__ == "__main__":
    main()