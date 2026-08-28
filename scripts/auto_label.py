"""
Run a trained YOLO model over Label Studio tasks and push its detections in
as *predictions* -- draft/suggested boxes shown to a human annotator to
confirm or edit, never written as final annotations. This is different from
`label_studio_import.py`, which only creates empty tasks for a human to
label from scratch; this script pre-fills those tasks with model output to
speed up review.

Usage:
    export LABEL_STUDIO_API_TOKEN=<your personal access token>
    python auto_label.py --project 4 --unlabeled-only
    python auto_label.py --project 4 --task-ids 17691,17957,18809
    python auto_label.py --project 4 --unlabeled-only --conf 0.25 \
        --weights /home/ai-intern/psn-training/thermal/runs/psn_thermal_v1/weights/best.pt
    python auto_label.py --project 4 --unlabeled-only --inference-policy optimized

Inference policy:
    --inference-policy vanilla (default): single model (--weights, default
    the deployed v12), single --conf threshold -- unchanged original
    behavior, nothing below affects this path.
    --inference-policy optimized: the validated v12 optimized inference
    policy (per-class confidence thresholds + a class-selective 4-checkpoint
    ensemble for Person only) from
    runs/v12_optimized_inference/v12_optimized_inference.py -- see that
    module and report/inference_validation/VALIDATION_REPORT.md for the
    full methodology and evidence (validated on a 236-image non-protected
    set, confirmed on a final protected-59 check: Person and Bridge
    precision/recall improve, no class regresses in F1; a small full-curve
    mAP50 trade-off is disclosed there). Uses v12's unchanged weights plus
    v10/v13/v16 for the Person ensemble -- all pre-existing checkpoints, no
    retraining. --weights and --conf are ignored in this mode (the
    policy's own validated thresholds apply); a notice is printed if either
    was explicitly set. This is opt-in only -- the default remains vanilla.
    All required checkpoints are validated eagerly before any tasks are
    processed -- a broken/missing checkpoint fails the run immediately with
    a clear error, it never silently falls back to vanilla and never burns
    through the task queue producing zero predictions.

    "optimized" is one entry in OPTIMIZED_POLICIES (see that dict below),
    not a permanent synonym for "v12". Model weights, an inference policy,
    and the validation benchmark that justified it are three separate
    versioned artifacts. Lifecycle for adding a future one, after a new
    model has already been independently trained and evaluated:

        new dataset -> retrain -> candidate model vX
            -> re-run the same kind of held-out evaluation this v12 policy
               went through (see VALIDATION_REPORT.md for the template)
            -> if it's worth doing, write a new small policy module
               (per-class thresholds and/or an ensemble, or neither --
               some future model may not need one at all) following the
               predict_optimized(image_path) contract
            -> add one entry to OPTIMIZED_POLICIES here
            -> validate the new --inference-policy choice the same way,
               opt-in only
            -> only then does it become an available choice here

    New data or a new model does NOT automatically change what "optimized"
    currently does -- that requires a new entry going through the same
    validation discipline the current one did, same as everywhere else in
    this project (see PROGRESS_REPORT.md's repeated emphasis on verifying
    before reporting).

Auth:
    Token is read from the LABEL_STUDIO_API_TOKEN environment variable --
    never hardcode it. Generate/rotate a token from the Label Studio UI:
    Account & Settings -> Personal Access Token.
    NOTE: a token was found hardcoded in plaintext in this project's history
    (both in a script and in shell history on the Label Studio server) --
    treat it as compromised and rotated, never reuse it here.

Weights:
    Defaults to the deployed RGB v12 checkpoint. Pass --weights to point at
    any other checkpoint (e.g. the thermal model, or a future RGB version).

Task selection:
    --task-ids: explicit comma-separated Label Studio task IDs.
    --unlabeled-only: every task in the project with total_annotations == 0
    (paginated via the same /api/projects/{id}/tasks endpoint used by
    label_studio_import.py). Exactly one of these must be given.

Image resolution:
    Each task's image lives at LABEL_STUDIO_URL + task["data"][--data-key]
    (default data key: "image", matching the "image" field used throughout
    this project -- see label_studio_import.py). That path is fetched over
    HTTP (with the same API token) and saved to a temp file for inference,
    since the images live on the Label Studio server (<LABEL_STUDIO_SERVER>) while
    training/inference happens on the L40S GPU box (<GPU_SERVER>) -- this
    script works from either machine as long as LABEL_STUDIO_URL is
    reachable and the weights file is local.

Label config (from_name / to_name):
    Label Studio's rectanglelabels prediction JSON needs the labeling
    config's `from_name`/`to_name` values (e.g. `<RectangleLabels
    name="label" toName="image">`). These were not recorded anywhere in
    this project's local files/history, so this script defaults to
    Label Studio's standard template values ("label" / "image") via
    --from-name/--to-name -- verify against the actual project config
    (Settings -> Labeling Interface -> Code) before relying on predictions
    rendering correctly, and override the flags if they differ.

Confidence threshold:
    --conf, default 0.25 -- this project's standard reporting threshold
    (see WORKFLOW.md step 9); low-confidence noise below this is not
    pushed as a prediction.

Push mechanism:
    POST /api/predictions/ per task, one prediction object containing all
    of that task's detected boxes as `result` regions. This is the
    documented REST endpoint that correctly updates Label Studio's
    `total_predictions` cache (unlike writing to the database directly).
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import uuid
from collections import Counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8085")
DEFAULT_WEIGHTS = "/home/ai-intern/psn-training/runs/psn_palm_oil_v12/weights/best.pt"

# Each optimized policy is a self-contained module under its own runs/<dir>/,
# following one fixed contract: a function predict_optimized(image_path) ->
# [(class_name, confidence, [x1,y1,x2,y2]), ...]. Optionally, for eager
# checkpoint validation (see load_optimized_policy below), it may also expose
# CHECKPOINTS (dict of name -> .pt path) and _get_model(name) the way
# v12_optimized_inference.py does.
#
# "optimized" today resolves to the v12-validated policy (thresholds +
# Person ensemble, see report/inference_validation/VALIDATION_REPORT.md) --
# this is a specific, versioned artifact, not a permanent definition of what
# "optimized" means. A future model version gets its own independently
# validated module and its own entry here (e.g. a hypothetical
# "optimized_v18": {...}); --inference-policy's choices and both dispatch
# points below are already driven by this dict, so adding that entry is the
# only change needed elsewhere in this file. No such entry exists yet --
# none is invented here.
OPTIMIZED_POLICIES = {
    "optimized": {
        "version": "v12",
        "module_dir": "/home/ai-intern/psn-training/runs/v12_optimized_inference",
        "module_name": "v12_optimized_inference",
        "description": "per-class thresholds + Person ensemble (v10+v12+v13+v16), "
                        "validated in report/inference_validation/VALIDATION_REPORT.md",
    },
}


def _fatal(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_optimized_policy(policy_name):
    """Import the given optimized policy module and eagerly validate that
    every checkpoint it needs can actually load -- BEFORE any tasks are
    processed. Exits with a clear error on any failure; never falls back to
    vanilla and never leaves a broken policy to be discovered task-by-task
    partway through a batch (this is the fix for code-review finding C1)."""
    config = OPTIMIZED_POLICIES[policy_name]
    sys.path.insert(0, config["module_dir"])
    try:
        module = importlib.import_module(config["module_name"])
    except Exception as e:
        _fatal(f"failed to import optimized policy {policy_name!r} "
               f"(module {config['module_name']!r} from {config['module_dir']}): {e}")

    # Best-effort eager load: only the checkpoints this specific policy
    # declares. v12's module exposes CHECKPOINTS + _get_model(name) (reached
    # into directly rather than added to the module, since
    # v12_optimized_inference.py is a validated artifact this integration
    # must not modify). A future policy that doesn't need an ensemble at all
    # can simply omit these two names -- eager validation is then skipped
    # and any load failure surfaces on the first task instead, which is a
    # reasonable degradation for a policy this project hasn't written yet.
    checkpoints = getattr(module, "CHECKPOINTS", None)
    get_model = getattr(module, "_get_model", None)
    if checkpoints and get_model:
        for name, path in checkpoints.items():
            try:
                get_model(name)
            except Exception as e:
                _fatal(f"optimized policy {policy_name!r} (version {config['version']}) "
                       f"failed to load checkpoint {name!r} ({path}): {e}\n"
                       f"Refusing to start -- an optimized policy that can't load its "
                       f"checkpoints must fail immediately, not partway through the task queue.")

    return module.predict_optimized, config


def api_request(token, method, path, body=None):
    url = path if path.startswith("http") else f"{LABEL_STUDIO_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


def fetch_tasks(token, project_id, unlabeled_only):
    """All tasks in the project, optionally filtered to total_annotations == 0."""
    tasks = []
    page = 1
    while True:
        data = api_request(
            token, "GET",
            f"/api/projects/{project_id}/tasks?page={page}&page_size=1000",
        )
        results = data if isinstance(data, list) else data.get("tasks", [])
        if not results:
            break
        for t in results:
            if unlabeled_only and t.get("total_annotations", 0) > 0:
                continue
            tasks.append(t)
        page += 1
    return tasks


def fetch_tasks_by_id(token, project_id, task_ids):
    tasks = []
    for tid in task_ids:
        tasks.append(api_request(token, "GET", f"/api/tasks/{tid}/"))
    return tasks


def download_image(token, image_path):
    url = image_path if image_path.startswith("http") else f"{LABEL_STUDIO_URL}{image_path}"
    req = Request(url, headers={"Authorization": f"Token {token}"})
    with urlopen(req) as resp:
        suffix = os.path.splitext(image_path)[1] or ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(resp.read())
    return tmp_path


def detections_to_result(detections, img_w, img_h, from_name, to_name):
    """[(class_name, confidence, [x1,y1,x2,y2]), ...] -> Label Studio rectanglelabels
    prediction regions. Confidence filtering already happened upstream (vanilla:
    --conf passed to model.predict; optimized: the validated thresholds baked
    into v12_optimized_inference.py) -- this function only builds the JSON shape,
    the same way regardless of which policy produced the detections."""
    result = []
    for cls_name, conf, (x1, y1, x2, y2) in detections:
        result.append({
            "id": uuid.uuid4().hex[:10],
            "type": "rectanglelabels",
            "from_name": from_name,
            "to_name": to_name,
            "original_width": img_w,
            "original_height": img_h,
            "image_rotation": 0,
            "value": {
                "rotation": 0,
                "x": x1 / img_w * 100,
                "y": y1 / img_h * 100,
                "width": (x2 - x1) / img_w * 100,
                "height": (y2 - y1) / img_h * 100,
                "rectanglelabels": [cls_name],
            },
            "score": conf,
        })
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", type=int, default=4, help="Label Studio project ID")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO .pt weights")
    parser.add_argument("--task-ids", help="Comma-separated Label Studio task IDs")
    parser.add_argument("--unlabeled-only", action="store_true",
                         help="Process every task in the project with zero annotations")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Confidence threshold below which detections are dropped (default 0.25)")
    parser.add_argument("--data-key", default="image",
                         help="Key in task['data'] holding the image path/URL (default 'image')")
    parser.add_argument("--from-name", default="label",
                         help="Labeling config from_name for the rectanglelabels control (default 'label')")
    parser.add_argument("--to-name", default="image",
                         help="Labeling config to_name for the image object (default 'image')")
    parser.add_argument("--model-version", default=None,
                         help="model_version string stored on each prediction (default: weights filename)")
    parser.add_argument("--inference-policy", choices=["vanilla", *OPTIMIZED_POLICIES], default="vanilla",
                         help="'vanilla' (default): single model + --conf, unchanged original behavior. "
                              "Any other choice is a validated optimized policy from OPTIMIZED_POLICIES "
                              "(currently: 'optimized', the v12-validated per-class-threshold + "
                              "Person-ensemble policy) -- ignores --weights/--conf. Opt-in only.")
    return parser


def main():
    args = build_parser().parse_args()

    if not args.task_ids and not args.unlabeled_only:
        print("Error: pass either --task-ids or --unlabeled-only.", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if not token:
        print("Error: LABEL_STUDIO_API_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    model = None
    predict_fn = None
    if args.inference_policy in OPTIMIZED_POLICIES:
        if args.weights != DEFAULT_WEIGHTS or args.conf != 0.25:
            print(f"Note: --inference-policy {args.inference_policy} ignores --weights/--conf -- "
                  f"using its own validated configuration.", file=sys.stderr)
        predict_fn, policy_config = load_optimized_policy(args.inference_policy)
        model_version = args.model_version or f"{args.inference_policy}_{policy_config['version']}"
        print(f"Inference policy: {args.inference_policy} (version {policy_config['version']}) "
              f"-- {policy_config['description']}")
    else:
        from ultralytics import YOLO  # deferred: only needed once auth/args are validated
        model_version = args.model_version or os.path.basename(args.weights)
        print(f"Inference policy: vanilla. Loading weights: {args.weights}")
        model = YOLO(args.weights)

    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        tasks = fetch_tasks_by_id(token, args.project, task_ids)
    else:
        tasks = fetch_tasks(token, args.project, unlabeled_only=True)
    print(f"Tasks to process: {len(tasks)}")

    processed = 0
    predictions_created = 0
    failures = []
    class_counts = Counter()

    for task in tasks:
        task_id = task["id"]
        image_path = task.get("data", {}).get(args.data_key)
        if not image_path:
            failures.append((task_id, f"no '{args.data_key}' field in task data"))
            continue

        tmp_path = None
        try:
            tmp_path = download_image(token, image_path)

            if args.inference_policy in OPTIMIZED_POLICIES:
                detections = predict_fn(tmp_path)
                img_w, img_h = Image.open(tmp_path).size
            else:
                results = model.predict(source=tmp_path, conf=args.conf, verbose=False)
                r = results[0]
                img_h, img_w = r.orig_shape
                detections = [
                    (r.names[int(b.cls[0])], float(b.conf[0]), [float(v) for v in b.xyxy[0]])
                    for b in r.boxes
                ]

            region = detections_to_result(detections, img_w, img_h, args.from_name, args.to_name)

            if region:
                body = {
                    "task": task_id,
                    "model_version": model_version,
                    "result": region,
                    "score": sum(reg["score"] for reg in region) / len(region),
                }
                api_request(token, "POST", "/api/predictions/", body)
                predictions_created += 1
                for reg in region:
                    class_counts[reg["value"]["rectanglelabels"][0]] += 1

            processed += 1
        except HTTPError as e:
            failures.append((task_id, f"HTTP {e.code}: {e.reason}"))
        except Exception as e:  # noqa: BLE001 -- report and keep going across a large batch
            failures.append((task_id, str(e)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    print()
    print("=== Summary ===")
    print(f"Tasks processed:      {processed}/{len(tasks)}")
    print(f"Predictions created:  {predictions_created}")
    print("Detections by class:")
    for cls_name, count in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls_name}: {count}")
    if failures:
        print(f"Failures: {len(failures)}")
        for task_id, reason in failures:
            print(f"  task {task_id}: {reason}")


if __name__ == "__main__":
    main()
