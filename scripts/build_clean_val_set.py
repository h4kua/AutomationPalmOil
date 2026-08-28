"""
Build the "clean-236" non-protected validation set for inference-policy
tuning/comparison, without touching protected-59.

Why this exists: `yolo_dataset_v12/images/val` (295 images) is NOT
disjoint from `images/val_protected` (59 images) -- val = protected-59
PLUS 236 additional held-out images, combined by `prepare_dataset.py`
(its own docstring: "train + combined val (protected + new)"). Using the
raw `val` folder for threshold tuning or model/policy selection would
silently include protected-59 in that selection process -- the same class
of leakage bug this project has hit twice before (v9's train/val leak,
the RGB/thermal frame contamination). This script produces the 236-image
remainder only, by exact filename exclusion, so tuning work has a
genuinely non-protected held-out set to use.

Usage:
    python build_clean_val_set.py
    # writes to psn-training/eval_sets/clean_236/{images,labels}/

Verification this script performs and prints:
    - exact output count (must be 295 - 59 = 236)
    - zero filename overlap with val_protected
    - zero byte-identical (md5) image overlap with val_protected
    - every copied image has a corresponding label file
"""
import hashlib
import shutil
from pathlib import Path

SRC = Path("/home/ai-intern/psn-training/yolo_dataset_v12")
OUT = Path("/home/ai-intern/psn-training/eval_sets/clean_236")


def md5_of(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    protected_names = {p.stem for p in (SRC / "images" / "val_protected").glob("*.jpg")}
    protected_md5 = {md5_of(p) for p in (SRC / "images" / "val_protected").glob("*.jpg")}
    print(f"protected-59 filenames: {len(protected_names)}")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    val_images = sorted((SRC / "images" / "val").glob("*.jpg"))
    print(f"source val/ pool: {len(val_images)} images")

    copied = 0
    for img_path in val_images:
        if img_path.stem in protected_names:
            continue
        if md5_of(img_path) in protected_md5:
            raise RuntimeError(
                f"{img_path.name} is not in the protected filename list but IS a "
                f"byte-identical duplicate of a protected image -- aborting, "
                f"investigate before proceeding."
            )
        lbl_path = SRC / "labels" / "val" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            raise RuntimeError(f"no label file for {img_path.name} -- aborting.")
        shutil.copy(img_path, OUT / "images" / img_path.name)
        shutil.copy(lbl_path, OUT / "labels" / lbl_path.name)
        copied += 1

    print(f"copied: {copied} images (expected 295 - 59 = 236)")

    # post-hoc verification
    out_names = {p.stem for p in (OUT / "images").glob("*.jpg")}
    out_md5 = {md5_of(p) for p in (OUT / "images").glob("*.jpg")}
    name_overlap = out_names & protected_names
    md5_overlap = out_md5 & protected_md5
    print(f"filename overlap with protected-59: {len(name_overlap)} (must be 0)")
    print(f"md5 (byte-identical) overlap with protected-59: {len(md5_overlap)} (must be 0)")
    missing_labels = [n for n in out_names if not (OUT / "labels" / f"{n}.txt").exists()]
    print(f"images missing a label file: {len(missing_labels)} (must be 0)")

    assert copied == 236, f"expected 236, got {copied}"
    assert len(name_overlap) == 0, "filename overlap with protected-59 detected!"
    assert len(md5_overlap) == 0, "byte-identical overlap with protected-59 detected!"
    assert len(missing_labels) == 0, "some images have no label file!"
    print("\nAll checks passed. clean-236 set is genuinely non-protected.")

    with open(OUT / "dataset.yaml", "w") as f:
        f.write(f"""path: {OUT}
train: images
val: images

nc: 6
names: ['Bridge', 'Car', 'Motorcycle', 'Palm_Oil_Fruit', 'Person', 'Truck']
""")
    print(f"dataset.yaml written to {OUT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
