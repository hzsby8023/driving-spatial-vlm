from datasets.drivelm_dataset import DriveLMParser
from datasets.image_loader import MultiViewImageLoader


ANNOTATION_PATH = (
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

NUSCENES_ROOT = (
    "/home/leo/projects/DriveLM/"
    "challenge/llama_adapter_v2_multimodal7b/"
    "data/nuscenes"
)


parser = DriveLMParser(ANNOTATION_PATH)
samples = parser.parse()

sample = samples[0]


loader = MultiViewImageLoader(
    nuscenes_root=NUSCENES_ROOT
)

images = loader.load(sample["images"])


print("Question:")
print(sample["question"])

print("\nAnswer:")
print(sample["answer"])

print("\nLoaded images:")

for camera, image in images.items():
    print(
        camera,
        "size =", image.size,
        "mode =", image.mode
    )