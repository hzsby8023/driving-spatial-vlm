import json
import hashlib
from pathlib import Path

def build_sample_id(
    scene_token: str,
    frame_token: str,
    task: str,
    qa_index: int,
) -> str:
    """
    Build a deterministic QA-level sample ID.

    Same DriveLM QA -> always same sample_id.
    """

    raw = (
        f"{scene_token}|"
        f"{frame_token}|"
        f"{task}|"
        f"{qa_index}"
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]
    

class DriveLMParser:

    def __init__(self, annotation_path):
        self.annotation_path = Path(annotation_path)

        with self.annotation_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

    def parse(self):
        samples = []

        for scene_token, scene_data in self.data.items():

            scene_description = scene_data["scene_description"]

            for frame_token, frame_data in scene_data["key_frames"].items():

                image_paths = frame_data["image_paths"]
                qa_dict = frame_data["QA"]

                for task, qa_list in qa_dict.items():

                    for qa_index, qa in enumerate(qa_list):

                        sample = {
                            "sample_id": build_sample_id(
                                scene_token=scene_token,
                                frame_token=frame_token,
                                task=task,
                                qa_index=qa_index,
                            ),

                            "scene_token": scene_token,
                            "frame_token": frame_token,
                            "scene_description": scene_description,

                            "task": task,
                            "qa_index": qa_index,

                            "images": image_paths,

                            "question": qa["Q"],
                            "answer": qa["A"],
                        }

                        samples.append(sample)

        return samples
