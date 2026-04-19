from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ..engine.output_utils import parse_model_output
from ..utils.visualization import save_prediction_triplet


class Inferencer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cuda",
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(torch.device(device))
        self.device = torch.device(device)
        self.threshold = threshold

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        model_output = self.model(images)
        logits = parse_model_output(model_output).main
        probs = torch.sigmoid(logits)
        return probs

    @torch.no_grad()
    def predict_loader(self, dataloader: DataLoader) -> List[dict]:
        self.model.eval()
        outputs: List[dict] = []
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            probs = self.predict_batch(images).cpu()
            for i in range(images.shape[0]):
                outputs.append(
                    {
                        "id": batch["id"][i],
                        "prob": probs[i],
                        "mask": (probs[i] >= self.threshold).float(),
                        "orig_size": batch.get("orig_size", None),
                    }
                )
        return outputs

    @torch.no_grad()
    def save_predictions(
        self,
        dataloader: DataLoader,
        save_dir: str | Path,
        *,
        save_visualizations: bool = True,
    ) -> List[Path]:
        self.model.eval()
        save_dir = Path(save_dir)
        mask_dir = save_dir / "masks"
        viz_dir = save_dir / "viz"
        mask_dir.mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            viz_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            probs = self.predict_batch(images).cpu()
            masks = (probs >= self.threshold).to(torch.uint8) * 255

            for i in range(images.shape[0]):
                sample_id = batch["id"][i]
                pred_mask = masks[i, 0].numpy().astype(np.uint8)
                mask_path = mask_dir / f"{sample_id}.png"
                Image.fromarray(pred_mask).save(mask_path)
                saved.append(mask_path)

                if save_visualizations:
                    gt = batch["mask"][i] if "mask" in batch else None
                    save_prediction_triplet(
                        batch["image"][i].cpu(),
                        probs[i].cpu(),
                        gt.cpu() if torch.is_tensor(gt) else None,
                        viz_dir / f"{sample_id}.png",
                        threshold=self.threshold,
                        from_logits=False,
                    )
        return saved


__all__ = ["Inferencer"]
