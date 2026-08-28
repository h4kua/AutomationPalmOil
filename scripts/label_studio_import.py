"""
Import extracted drone-footage frames into Label Studio as new annotation
tasks, via the REST API.

Reconstructed and cleaned up for the final deliverable. There was no single
saved script for this step historically -- the actual imports (batches of
1,087 / 7,639 / remaining frames) were run as ad-hoc inline Python +
`curl -X POST .../api/projects/{id}/import`, with the API token typed
directly into the shell each time (visible in plain text in .bash_history
on the Label Studio server). This script reproduces that exact mechanism
as a single, reusable, non-secret-leaking tool.

Usage:
    export LABEL_STUDIO_API_TOKEN=<your personal access token>
    python label_studio_import.py --manifest candidates.txt

Auth:
    Token is read from the LABEL_STUDIO_API_TOKEN environment variable --
    never hardcode it. Generate/rotate a token from the Label Studio UI:
    Account & Settings -> Personal Access Token.
    NOTE: a token was found hardcoded in plaintext in this project's
    history (both in a script and in shell history on the Label Studio
    server) -- it should be treated as compromised and rotated.

Project ID:
    Set via --project (defaults to 4, the RGB/thermal PSN project used
    throughout this project).

Input manifest format:
    Tab-separated, one candidate frame per line:
        <video_folder>\t<frame_filename>\t<optional reason/class column>
    Only the first two columns are used. This matches the output format
    of this project's candidate-survey scripts (model-assisted detection
    + diverse random sampling over extracted frames).

Prerequisite:
    Extracted frames must already exist under /data/frames/<video_folder>/
    on the Label Studio server, and that path must be reachable by Label
    Studio's local-files serving -- in this project that was done via:
        sudo ln -s /data/frames ~/label-studio/mydata/media/frames
    so that a frame resolves to the URL /data/media/frames/<video>/<file>.

Output:
    Imports each frame as a new task (no annotations attached -- these
    become empty tasks for a human annotator to label). Prints how many
    were submitted and Label Studio's response summary. De-dupes against
    frames that are already tasks in the target project (by image URL),
    so this script is safe to re-run on a manifest that overlaps a
    previous import.
"""
import argparse
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8085")
FRAMES_URL_PREFIX = "/data/media/frames"
BATCH_SIZE = 500


def load_manifest(path):
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            video_folder, frame_filename = parts[0], parts[1]
            frames.append((video_folder, frame_filename))
    return frames


def fetch_existing_image_urls(token, project_id):
    """Existing task image URLs in the project, to skip on re-run."""
    existing = set()
    page = 1
    while True:
        req = Request(
            f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks?page={page}&page_size=1000",
            headers={"Authorization": f"Token {token}"},
        )
        try:
            with urlopen(req) as resp:
                data = json.loads(resp.read())
        except HTTPError as e:
            print(f"Warning: could not fetch existing tasks ({e}); "
                  f"proceeding without de-dup.", file=sys.stderr)
            return existing
        results = data if isinstance(data, list) else data.get("tasks", [])
        if not results:
            break
        for task in results:
            image = task.get("data", {}).get("image")
            if image:
                existing.add(image)
        page += 1
    return existing


def import_batch(token, project_id, tasks):
    body = json.dumps(tasks).encode("utf-8")
    req = Request(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Tab-separated candidate frame list")
    parser.add_argument("--project", type=int, default=4)
    args = parser.parse_args()

    token = os.environ.get("LABEL_STUDIO_API_TOKEN")
    if not token:
        print("Error: LABEL_STUDIO_API_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    frames = load_manifest(args.manifest)
    print(f"Manifest: {len(frames)} candidate frames")

    existing = fetch_existing_image_urls(token, args.project)
    print(f"Existing tasks in project {args.project}: {len(existing)}")

    tasks = []
    for video_folder, frame_filename in frames:
        url = f"{FRAMES_URL_PREFIX}/{video_folder}/{frame_filename}"
        if url in existing:
            continue
        tasks.append({"image": url})

    print(f"New tasks to import: {len(tasks)} (skipped {len(frames) - len(tasks)} already-present)")

    if not tasks:
        print("Nothing to do.")
        return

    total_imported = 0
    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i : i + BATCH_SIZE]
        result = import_batch(token, args.project, batch)
        imported = len(batch)
        total_imported += imported
        print(f"  batch {i // BATCH_SIZE + 1}: submitted {imported} tasks -> {result}")

    print(f"Done. {total_imported} tasks imported into project {args.project}.")


if __name__ == "__main__":
    main()
