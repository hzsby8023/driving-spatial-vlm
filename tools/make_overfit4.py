
import json
from pathlib import Path

src = Path("outputs/autovla_codebook/train_sft.jsonl")
dst = Path("outputs/autovla_codebook/overfit4.jsonl")

rows = [json.loads(line) for line in src.open() if line.strip()]

selected = []
seen = set()

for row in rows:
    actions = row["action_tokens"]

    if len(actions) != 10:
        continue
    if 0 in actions:
        continue

    signature = tuple(actions)
    if signature in seen:
        continue

    selected.append(row)
    seen.add(signature)

    if len(selected) == 4:
        break

if len(selected) != 4:
    raise RuntimeError(
        f"Only found {len(selected)} suitable samples"
    )

dst.parent.mkdir(parents=True, exist_ok=True)

with dst.open("w") as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved:", dst)
for i, row in enumerate(selected):
    print(
        i,
        "scene:", row.get("scene_token"),
        "actions:", row["action_tokens"],
    )
