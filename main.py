import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:  # pragma: no cover - ensures actionable error
    raise SystemExit(
        "segment-anything must be installed. Clone the repo and run `pip install -e .`."
    ) from exc


@dataclass
class ExperimentConfig:
    data_root: str = "DIC-C2DH-HeLa"
    sequence: str = "01"
    batch_size: int = 1
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    seed: int = 7
    sam_model_type: str = "vit_b"
    sam_checkpoint_path: str = os.path.join("weights", "sam_vit_b_01ec64.pth")
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = True

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
            sample["mask_aug"] = mask_aug if mask_aug.dim() == 3 else mask_aug.unsqueeze(0)

        return sample

    def __len__(self) -> int:
        return len(self.imgs)


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

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
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


def build_sam_model(cfg: ExperimentConfig, device: torch.device):
    checkpoint_path = ensure_sam_checkpoint(cfg.sam_checkpoint_path, cfg.sam_model_type)
    model = sam_model_registry[cfg.sam_model_type](checkpoint=checkpoint_path)
    model.to(device)
    return model


def configure_trainable_modules(model, cfg: ExperimentConfig):
    for param in model.image_encoder.parameters():
        param.requires_grad = not cfg.freeze_image_encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = not cfg.freeze_prompt_encoder
    for param in model.mask_decoder.parameters():
        param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable SAM parameters selected; adjust freezing options.")
    return trainable


def mask_to_box(mask: torch.Tensor) -> np.ndarray:
    binary = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Mask is empty; cannot derive bounding box.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def gather_views_from_batch(batch: Dict[str, torch.Tensor], include_augmented: bool) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    views: List[Tuple[torch.Tensor, torch.Tensor]] = []
    imgs = batch["img"]
    masks = batch["mask"]
    for idx in range(imgs.shape[0]):
        views.append((imgs[idx], masks[idx]))

    if include_augmented and "img_aug" in batch and "mask_aug" in batch:
        aug_imgs = batch["img_aug"]
        aug_masks = batch["mask_aug"]
        for idx in range(aug_imgs.shape[0]):
            views.append((aug_imgs[idx], aug_masks[idx]))
    return views


def prepare_image_for_sam(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] != 3:
        raise ValueError(f"Unexpected number of channels ({img.shape[0]}).")
    return (img * 255.0).float()


def sam_predict_logits(model, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    model_input = model.preprocess(image.unsqueeze(0))
    image_embeddings = model.image_encoder(model_input)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    dense_pe = model.prompt_encoder.get_dense_pe()
    dense_pe = dense_pe.to(image_embeddings.device).type_as(image_embeddings)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=dense_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    upscaled_masks = model.postprocess_masks(
        low_res_masks,
        input_size=image.shape[-2:],
        original_size=image.shape[-2:],
    )
    return upscaled_masks


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.sigmoid(logits)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return dice.mean().item()


def forward_view(model, img: torch.Tensor, mask: torch.Tensor, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if mask.max() <= 0:
        return None
    rgb = prepare_image_for_sam(img).to(device)
    try:
        bbox = mask_to_box(mask)
    except RuntimeError:
        return None

    box_tensor = torch.as_tensor(bbox, dtype=torch.float32, device=device).unsqueeze(0)
    logits = sam_predict_logits(model, rgb, box_tensor)
    target = mask.unsqueeze(0).to(device)
    return logits, target


def train_sam_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    steps = 0

    for batch in loader:
        views = gather_views_from_batch(batch, include_augmented=True)
        optimizer.zero_grad()

        view_losses = []
        view_dices = []
        for img, mask in views:
            result = forward_view(model, img, mask, device)
            if result is None:
                continue
            logits, target = result
            loss = criterion(logits, target)
            view_losses.append(loss)
            view_dices.append(dice_from_logits(logits, target))

        if not view_losses:
            continue

        batch_loss = torch.stack(view_losses).mean()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_dice += float(sum(view_dices) / len(view_dices))
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return epoch_loss / steps, epoch_dice / steps


@torch.no_grad()
def evaluate_sam(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    samples = 0

    for batch in loader:
        views = gather_views_from_batch(batch, include_augmented=False)
        for img, mask in views:
            result = forward_view(model, img, mask, device)
            if result is None:
                continue
            logits, target = result
            loss = criterion(logits, target)
            total_loss += loss.item()
            total_dice += dice_from_logits(logits, target)
            samples += 1

    if samples == 0:
        return 0.0, 0.0
    return total_loss / samples, total_dice / samples


def fine_tune_sam(model, train_loader, val_loader, cfg: ExperimentConfig, device: torch.device):
    trainable_params = configure_trainable_modules(model, cfg)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_dice = train_sam_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate_sam(model, val_loader, criterion, device)
        print(
            f"[SAM fine-tune] Epoch {epoch:02d}/{cfg.epochs} "
            f"- train loss: {train_loss:.4f} | train dice: {train_dice:.4f} "
            f"- val loss: {val_loss:.4f} | val dice: {val_dice:.4f}"
        )


def run_sam_with_box(image: torch.Tensor, box: np.ndarray, sam_model, device: torch.device) -> np.ndarray:
    predictor = SamPredictor(sam_model)
    predictor.model.to(device)
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

    sam_model = build_sam_model(cfg, device)
    fine_tune_sam(sam_model, train_loader, val_loader, cfg, device)

    if len(val_loader.dataset) == 0:
        print("Validation set is empty; skipping SAM inference.")
        return

    sam_model.eval()
    sample_batch = next(iter(val_loader))
    sample_img = sample_batch["img"][0]
    sample_mask = sample_batch["mask"][0]
    bbox = mask_to_box(sample_mask)
    sam_mask = run_sam_with_box(sample_img, bbox, sam_model, device)

    gt_mask = sample_mask.squeeze().cpu().numpy()
    sam_binary = (sam_mask > 0).astype(np.float32)
    intersection = np.logical_and(sam_binary, gt_mask).sum()
    union = np.logical_or(sam_binary, gt_mask).sum()
    iou = intersection / max(union, 1)
    print(f"SAM IoU against ground-truth mask: {iou:.4f}")


if __name__ == "__main__":
    main()
