from torch.utils.data import Dataset


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


CAMERA_DISPLAY_NAMES = {
    "CAM_FRONT": "Front camera",
    "CAM_FRONT_LEFT": "Front-left camera",
    "CAM_FRONT_RIGHT": "Front-right camera",
    "CAM_BACK": "Rear camera",
    "CAM_BACK_LEFT": "Rear-left camera",
    "CAM_BACK_RIGHT": "Rear-right camera",
}


class DriveLMSFTDataset(Dataset):

    def __init__(
        self,
        samples,
        image_loader,
    ):
        self.samples = samples
        self.image_loader = image_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------
        # Load six surround-view images
        # ----------------------------------------

        images = self.image_loader.load(
            sample["images"]
        )

        # ----------------------------------------
        # Build multimodal user content
        # ----------------------------------------

        user_content = []

        ordered_images = []

        for camera_name in CAMERA_ORDER:

            image = images[camera_name]

            ordered_images.append(image)

            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"{CAMERA_DISPLAY_NAMES[camera_name]}:"
                    ),
                }
            )

            user_content.append(
                {
                    "type": "image",
                    "image": image,
                }
            )

        # ----------------------------------------
        # Same task protocol as baseline
        # ----------------------------------------

        instruction = (
            "\nThe camera labels only indicate where each image "
            "was captured relative to the ego vehicle. "
            "Do not answer with camera names. "
            "Answer the question itself based on the visible "
            "physical objects. "
            "If the question asks what objects are present, "
            "name the objects. "
            "If the question asks for an object's status, "
            "state its status. "
            "Answer in one concise sentence.\n\n"
            f"Question: {sample['question']}"
        )

        user_content.append(
            {
                "type": "text",
                "text": instruction,
            }
        )

        # ----------------------------------------
        # Conversation
        # ----------------------------------------

        messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["answer"],
                    }
                ],
            },
        ]

        return {
            "sample_id": sample["sample_id"],
            "scene_token": sample["scene_token"],
            "frame_token": sample["frame_token"],
            "task": sample["task"],

            "question": sample["question"],
            "answer": sample["answer"],

            "images": ordered_images,

            "messages": messages,
        }