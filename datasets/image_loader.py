from pathlib import Path

from PIL import Image


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class MultiViewImageLoader:
    def __init__(self, nuscenes_root):
        """
        Args:
            nuscenes_root:
                Path to the nuScenes root directory.

                Example:
                /home/leo/projects/DriveLM/challenge/
                llama_adapter_v2_multimodal7b/data/nuscenes
        """
        self.nuscenes_root = Path(nuscenes_root)

    def resolve_path(self, image_path):
        """
        Convert DriveLM annotation path:

            ../nuscenes/samples/CAM_FRONT/xxx.jpg

        into:

            <nuscenes_root>/samples/CAM_FRONT/xxx.jpg
        """
        image_path = Path(image_path)

        parts = image_path.parts

        try:
            nuscenes_idx = parts.index("nuscenes")
        except ValueError:
            raise ValueError(
                f"Cannot find 'nuscenes' in image path: {image_path}"
            )

        relative_path = Path(*parts[nuscenes_idx + 1:])

        return self.nuscenes_root / relative_path

    def load(self, image_paths):
        """
        Args:
            image_paths:
                {
                    "CAM_FRONT": "...",
                    "CAM_FRONT_LEFT": "...",
                    ...
                }

        Returns:
            {
                "CAM_FRONT": PIL.Image,
                ...
            }
        """

        images = {}

        for camera in CAMERA_ORDER:
            if camera not in image_paths:
                raise KeyError(f"Missing camera: {camera}")

            path = self.resolve_path(image_paths[camera])

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found for {camera}: {path}"
                )

            image = Image.open(path).convert("RGB")

            images[camera] = image

        return images