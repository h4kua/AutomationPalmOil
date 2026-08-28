"""
Convert Label Studio annotations to a YOLO-format training dataset, using
the protected-validation-set methodology that every RGB version from v10
onward (including the deployed v12) has used.

This is the exact script that built the currently-deployed v12 dataset
(originally named prepare_dataset_v12.py). Renamed for the deliverable;
logic unchanged.

Input format note:
    RAW (rgb_v12_raw_annotations.txt) is NOT the standard Label Studio COCO
    export -- it's a simpler pipe-delimited pull of one line per task:
        <task_id>|<image_path>|<annotation_result_json>
    produced directly from the database (see WORKFLOW.md for the exact
    query). If starting instead from a fresh COCO export
    (data/annotation_export/result.json in this deliverable), that JSON
    needs a small adapter to reach this same per-task
    (id, image_path, boxes) shape before this script's logic applies --
    the box-decoding math below (percent-of-image x/y/width/height ->
    normalized YOLO cx/cy/w/h) is the part that would carry over directly.

Protected validation set (protected_val_task_ids.txt):
    A fixed set of 59 task IDs, permanently excluded from the train/val
    split BEFORE any random shuffling happens, for every version from v10
    onward. This exists specifically to prevent train/validation leakage
    (a real bug found in an earlier version, v9, where validation images
    ended up in the training set and inflated reported metrics). Never
    delete or regenerate this file -- every version's reported numbers are
    only comparable to each other because they all evaluate against this
    same fixed set.

Data source used for v12 (verified from this script directly, not
assumed): IMAGES_SRC = rgb_v9_images, sourced only from RAW's own task
list -- no VisDrone, no Kaggle-derived synthetic compositing, no other
external dataset path anywhere in this script. v12's training lineage is
100% PSN's own manually-annotated drone footage.

Output:
    yolo_dataset_v12/
      images/{train,val,val_protected}/, labels/{train,val,val_protected}/
      dataset.yaml               -- train + combined val (protected + new)
      dataset_protected_val.yaml -- train + ONLY the protected 59, for an
                                     isolated eval comparable across every
                                     version
"""
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

BASE = Path(r'C:\Users\asus\psn-training')
RAW = BASE / 'rgb_v12_raw_annotations.txt'
IMAGES_SRC = BASE / 'rgb_v9_images'
PROTECTED_IDS_FILE = BASE / 'protected_val_task_ids.txt'
YOLO_DIR = BASE / 'yolo_dataset_v12'

CLASS_NAMES = ['Bridge', 'Car', 'Motorcycle', 'Palm_Oil_Fruit', 'Person', 'Truck']
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

protected_ids = set(int(l.strip()) for l in PROTECTED_IDS_FILE.read_text().splitlines() if l.strip())
print(f'Protected (permanently excluded from split) task_ids: {len(protected_ids)}')

for split in ['train', 'val', 'val_protected']:
    (YOLO_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

boxes_by_task = defaultdict(list)
image_path_by_task = {}

with open(RAW, encoding='utf-8') as f:
    for line in f:
        task_id, image_path, result_json = line.rstrip('\n').split('|', 2)
        task_id = int(task_id)
        image_path_by_task[task_id] = image_path
        result = json.loads(result_json)
        for r in result:
            v = r['value']
            cls_name = v['rectanglelabels'][0]
            if cls_name not in CLASS_IDX:
                continue
            cx = (v['x'] + v['width'] / 2) / 100
            cy = (v['y'] + v['height'] / 2) / 100
            bw = v['width'] / 100
            bh = v['height'] / 100
            boxes_by_task[task_id].append((CLASS_IDX[cls_name], cx, cy, bw, bh))

all_task_ids = set(image_path_by_task.keys())
assert protected_ids.issubset(all_task_ids), 'protected ids missing from pool!'

# STEP 1: exclude protected BEFORE splitting
remaining_ids = sorted(all_task_ids - protected_ids)
print(f'Total annotated pool: {len(all_task_ids)}')
print(f'Remaining after excluding protected 59: {len(remaining_ids)}')

random.seed(42)
shuffled = remaining_ids[:]
random.shuffle(shuffled)
split_idx = int(len(shuffled) * 0.8)
train_ids = set(shuffled[:split_idx])
val_new_ids = set(shuffled[split_idx:])

print(f'Train (new pool only): {len(train_ids)}')
print(f'Val-new (new pool only): {len(val_new_ids)}')
print(f'Val-protected (fixed 59): {len(protected_ids)}')
print(f'Full val (protected + new) = {len(protected_ids) + len(val_new_ids)}')


def write_task(task_id, split):
    image_path = image_path_by_task[task_id]
    _, video, filename = image_path.strip('/').split('/', 2)
    src = IMAGES_SRC / video / filename
    dst_img = YOLO_DIR / 'images' / split / f'task_{task_id}.jpg'
    if not src.exists():
        return False
    shutil.copy(src, dst_img)
    label_path = YOLO_DIR / 'labels' / split / f'task_{task_id}.txt'
    with open(label_path, 'w') as f:
        for cls, cx, cy, bw, bh in boxes_by_task[task_id]:
            f.write(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
    return True


n_missing = 0
for task_id in train_ids:
    if not write_task(task_id, 'train'):
        n_missing += 1

for task_id in val_new_ids:
    # goes into BOTH the combined 'val' (used during training) ...
    if not write_task(task_id, 'val'):
        n_missing += 1

for task_id in protected_ids:
    # protected goes into BOTH combined 'val' AND the isolated 'val_protected'
    ok1 = write_task(task_id, 'val')
    ok2 = write_task(task_id, 'val_protected')
    if not (ok1 and ok2):
        n_missing += 1

print(f'Missing source images: {n_missing}')
print(f'images/train: {len(list((YOLO_DIR / "images" / "train").iterdir()))}')
print(f'images/val (combined): {len(list((YOLO_DIR / "images" / "val").iterdir()))}')
print(f'images/val_protected: {len(list((YOLO_DIR / "images" / "val_protected").iterdir()))}')

yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
(YOLO_DIR / 'dataset.yaml').write_text(yaml_content)

yaml_protected = f"""path: {YOLO_DIR}
train: images/train
val: images/val_protected

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
(YOLO_DIR / 'dataset_protected_val.yaml').write_text(yaml_protected)

print()
print('dataset.yaml (train + combined val):')
print(yaml_content)
print('dataset_protected_val.yaml (for isolated post-training eval):')
print(yaml_protected)
