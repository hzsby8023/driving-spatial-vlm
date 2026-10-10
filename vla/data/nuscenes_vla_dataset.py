
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesVLADataset(Dataset):
    def __init__(self, manifest, dataroot):
        self.dataroot = Path(dataroot)

        with open(manifest, encoding="utf-8") as f:
            self.records = [
                json.loads(line)
                for line in f if line.strip()
            ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        images = []
        for camera in CAMERAS:
            path = (
                self.dataroot
                / record["cameras"][camera]["path"]
            )
            with Image.open(path) as img:
                images.append(img.convert("RGB"))

        return {
            "sample_token": record["sample_token"],
            "images": images,
            "ego_history": torch.tensor(
                record["ego_history"],
                dtype=torch.float32,
            ),
            "future_trajectory": torch.tensor(
                record["future_trajectory"],
                dtype=torch.float32,
            ),
            "history_time_offsets_sec": torch.tensor(
                record["history_time_offsets_sec"],
                dtype=torch.float32,
            ),
            "future_time_offsets_sec": torch.tensor(
                record["future_time_offsets_sec"],
                dtype=torch.float32,
            ),
        }
