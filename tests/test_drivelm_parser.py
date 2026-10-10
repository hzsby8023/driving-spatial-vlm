from datasets.drivelm_dataset import DriveLMParser


annotation_path = (
    "/home/leo/projects/DriveLM/"
    "challenge/data/train_sample.json"
)

parser = DriveLMParser(annotation_path)

samples = parser.parse()

print("Total samples:", len(samples))

sample = samples[0]

print("\nTask:")
print(sample["task"])

print("\nImages:")
for camera, path in sample["images"].items():
    print(camera, "->", path)

print("\nQuestion:")
print(sample["question"])

print("\nAnswer:")
print(sample["answer"])

# --------------------------------------------------
# Stable sample ID checks
# --------------------------------------------------

sample_ids = [
    sample["sample_id"]
    for sample in samples
]

# Every sample should have an ID.
assert all(sample_ids)

# IDs should be unique at QA level.
assert len(sample_ids) == len(set(sample_ids))

print(
    f"Unique sample IDs: "
    f"{len(set(sample_ids))}"
)

# Check required lineage fields.
first = samples[0]

assert "scene_token" in first
assert "frame_token" in first
assert "task" in first
assert "qa_index" in first
assert "sample_id" in first

print("Sample identity test: PASS")