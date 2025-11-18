import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import DataLoader, Dataset, Subset

from segment_anything import SamPredictor, sam_model_registry


@dataclass
class ExperimentConfig:
    data_root: str = "DIC-C2DH-HeLa"
    sequence: str = "01"
    batch_size: int = 1
    epochs: int = 5
    lr: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 7
    sam_model_type: str = "vit_b"
    sam_checkpoint_path: str = os.path.join("weights", "sam_vit_b_01ec64.pth")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_root, self.sequence)

    @property
    def masks_dir(self) -> str:
        return os.path.join(self.data_root, f"{self.sequence}_ST", "SEG")


class RandomRotationPair:
    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = random.uniform(-self.degrees, self.degrees)
        img_rot = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

        mask = mask.unsqueeze(0)
        mask_rot = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        mask_rot = mask_rot.squeeze(0)
        mask_rot = (mask_rot > 0.5).float()

        return img_rot, mask_rot


class ElasticDeformationPair:
    """
    Elastic deformation similar to the augmentations suggested in the U-Net paper
    (Ronneberger et al., 2015).
    """

    def __init__(self, alpha: float = 50, sigma: float = 4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = img.squeeze(0).numpy()
        mask_np = mask.squeeze(0).numpy()

        shape = img_np.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma=self.sigma, mode="reflect") * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma=self.sigma, mode="reflect") * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

        img_deformed = map_coordinates(img_np, indices, order=1, mode="reflect").reshape(shape)
        mask_deformed = map_coordinates(mask_np, indices, order=0, mode="reflect").reshape(shape)

        img_deformed = torch.from_numpy(img_deformed).unsqueeze(0).float()
        mask_deformed = torch.from_numpy(mask_deformed).unsqueeze(0).float()

        return img_deformed, mask_deformed


class FullAugmentation:
    def __init__(self):
        self.rotate = RandomRotationPair(180)
        self.elastic = ElasticDeformationPair(alpha=50, sigma=4)

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.rotate(img, mask)
        img, mask = self.elastic(img, mask)
        mask = (mask > 0.5).float()
        return img, mask


class CellSegDataSet(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.imgs = sorted([f for f in os.listdir(images_dir) if f.endswith(".tif")])
        self.masks = sorted([f for f in os.listdir(annotations_dir) if f.endswith(".tif")])
        self.to_tensor = T.ToTensor()
        self.transform = transform

    def _load_mask(self, path: str) -> torch.Tensor:
        mask_np = np.array(Image.open(path).convert("L"))
        mask_bin = (mask_np > 0).astype(np.float32)
        return torch.from_numpy(mask_bin).unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path = os.path.join(self.images_dir, self.imgs[index])
        mask_path = os.path.join(self.annotations_dir, self.masks[index])

        img = Image.open(img_path).convert("L")
        img_tensor = self.to_tensor(img)
        mask = self._load_mask(mask_path)

        sample = {"img": img_tensor, "mask": mask}

        if self.transform:
            img_aug, mask_aug = self.transform(img_tensor.clone(), mask.clone())
            sample["img_aug"] = img_aug
            sample["mask_aug"] = mask_aug.unsqueeze(0) if mask_aug.dim() == 2 else mask_aug

        return sample

    def __len__(self) -> int:
        return len(self.imgs)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base_channels)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        bottleneck = self.bottleneck(self.pool(d3))

        u3 = torch.cat([self.up3(bottleneck), d3], dim=1)
        u3 = self.conv3(u3)

        u2 = torch.cat([self.up2(u3), d2], dim=1)
        u2 = self.conv2(u2)

        u1 = torch.cat([self.up1(u2), d1], dim=1)
        u1 = self.conv1(u1)

        return self.out_conv(u1)


def stack_original_and_augmented(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = [batch["img"]]
    masks = [batch["mask"]]

    if "img_aug" in batch and "mask_aug" in batch:
        imgs.append(batch["img_aug"])
        masks.append(batch["mask_aug"])

    img_tensor = torch.cat(imgs, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    return img_tensor, mask_tensor


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.sigmoid(logits)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return dice.mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0

    for batch in loader:
        imgs, masks = stack_original_and_augmented(batch)
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            running_dice += dice_from_logits(logits, masks)

    n = len(loader)
    return running_loss / n, running_dice / n


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    base_dataset = CellSegDataSet(cfg.images_dir, cfg.masks_dir, transform=None)
    indices = list(range(len(base_dataset)))
    random.Random(cfg.seed).shuffle(indices)

    num_val = max(1, int(len(indices) * cfg.val_fraction))
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    if not train_indices:
        raise RuntimeError("Validation split consumed all samples; decrease val_fraction.")

    train_dataset = Subset(CellSegDataSet(cfg.images_dir, cfg.masks_dir, transform=FullAugmentation()), train_indices)
    val_dataset = Subset(CellSegDataSet(cfg.images_dir, cfg.masks_dir, transform=None), val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def ensure_sam_checkpoint(path: str, model_type: str) -> str:
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = SAM_URLS[model_type]
    print(f"Downloading SAM weights ({model_type}) to {path} ...")
    torch.hub.download_url_to_file(url, path)
    return path


def mask_to_box(mask: torch.Tensor) -> np.ndarray:
    binary = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Mask is empty; cannot derive bounding box.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def run_sam_with_box(image: torch.Tensor, box: np.ndarray, cfg: ExperimentConfig, device: torch.device) -> np.ndarray:
    checkpoint_path = ensure_sam_checkpoint(cfg.sam_checkpoint_path, cfg.sam_model_type)
    sam = sam_model_registry[cfg.sam_model_type](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    image_np = (image.squeeze().cpu().numpy() * 255).astype(np.uint8)
    image_rgb = np.stack([image_np] * 3, axis=-1)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=box, point_coords=None, point_labels=None, multimask_output=False)
    print(f"SAM bounding-box prompt score: {scores[0]:.4f}")
    return masks[0]


def main():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    train_loader, val_loader = prepare_dataloaders(cfg)

    model = SimpleUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"- train loss: {train_loss:.4f} "
            f"- val loss: {val_loss:.4f} "
            f"- val dice: {val_dice:.4f}"
        )

    if len(val_loader.dataset) == 0:
        print("Validation set is empty; skipping SAM test.")
        return

    sample_batch = next(iter(val_loader))
    sample_img = sample_batch["img"][0]
    sample_mask = sample_batch["mask"][0]
    bbox = mask_to_box(sample_mask)
    sam_mask = run_sam_with_box(sample_img, bbox, cfg, device)

    gt_mask = sample_mask.squeeze().cpu().numpy()
    sam_binary = (sam_mask > 0).astype(np.float32)
    intersection = np.logical_and(sam_binary, gt_mask).sum()
    union = np.logical_or(sam_binary, gt_mask).sum()
    iou = intersection / max(union, 1)
    print(f"SAM IoU against ground-truth mask: {iou:.4f}")


if __name__ == "__main__":
    main()
