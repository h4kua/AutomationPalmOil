"""
Regression test for auto_label.py's --inference-policy support: CLI
parsing, the policy registry, eager checkpoint validation (code-review
finding C1), and file synchronization.

Deliberately network-free (no LABEL_STUDIO_API_TOKEN, no Label Studio
server needed). Loading the real "optimized" policy's four checkpoints
(test_eager_load_succeeds_for_real_policy) does touch the GPU/local weight
files -- this is loading weights, not running inference, and is the exact
mechanism being verified, so it's intentional rather than "expensive
inference."

Usage:
    python test_auto_label_policy.py
Exits 0 and prints "ALL PASSED" if every check passes, otherwise raises.
"""
import filecmp
import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).parent
FINAL_DELIVERABLE_SCRIPTS = HERE.parent / "FINAL_DELIVERABLE" / "scripts"

spec = importlib.util.spec_from_file_location("auto_label", HERE / "auto_label.py")
auto_label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_label)


# --- 1. default policy = vanilla ---------------------------------------
def test_default_is_vanilla():
    args = auto_label.build_parser().parse_args(["--project", "4", "--unlabeled-only"])
    assert args.inference_policy == "vanilla", f"default should be 'vanilla', got {args.inference_policy!r}"


# --- 2. optimized policy remains selectable -----------------------------
def test_optimized_is_selectable_and_distinct():
    args = auto_label.build_parser().parse_args(
        ["--project", "4", "--unlabeled-only", "--inference-policy", "optimized"]
    )
    assert args.inference_policy == "optimized"
    assert args.inference_policy != "vanilla"


# --- 3. invalid policy rejected ------------------------------------------
def test_invalid_policy_rejected():
    try:
        auto_label.build_parser().parse_args(
            ["--project", "4", "--unlabeled-only", "--inference-policy", "bogus"]
        )
    except SystemExit as e:
        assert e.code != 0, "argparse should reject an invalid --inference-policy choice"
    else:
        raise AssertionError("expected argparse to reject an invalid --inference-policy choice")


# --- 4. current validated thresholds remain unchanged (values, not just keys) --
def test_validated_thresholds_unchanged():
    sys.path.insert(0, auto_label.OPTIMIZED_POLICIES["optimized"]["module_dir"])
    import v12_optimized_inference as m
    assert m.PER_CLASS_CONF == {0: 0.40, 1: 0.25, 2: 0.40, 3: 0.35, 5: 0.50}, (
        f"PER_CLASS_CONF changed from its validated values: {m.PER_CLASS_CONF!r} "
        f"-- this integration must never alter the validated v12 thresholds"
    )


# --- 5. current validated ensemble composition remains unchanged ---------
def test_validated_ensemble_composition_unchanged():
    sys.path.insert(0, auto_label.OPTIMIZED_POLICIES["optimized"]["module_dir"])
    import v12_optimized_inference as m
    assert set(m.CHECKPOINTS.keys()) == {"v10", "v12", "v13", "v16"}
    assert m.PERSON_ENSEMBLE_MIN_VOTES == 2
    assert m.PERSON_ENSEMBLE_NMS_IOU == 0.55
    assert m.PERSON_ENSEMBLE_CONF_IN == 0.15
    assert m.PERSON_ENSEMBLE_FINAL_CONF == 0.25


# --- 6 & 7. optimized validation happens before task processing; a ------
#            broken checkpoint fails immediately -------------------------
def test_eager_load_succeeds_for_real_policy():
    """The real, validated policy's checkpoints all load cleanly up front --
    proves load_optimized_policy() actually performs the eager-load step
    for a policy that declares CHECKPOINTS/_get_model, not just a no-op."""
    predict_fn, config = auto_label.load_optimized_policy("optimized")
    assert callable(predict_fn)
    assert config["version"] == "v12"


def test_broken_checkpoint_fails_immediately_before_task_loop():
    """A policy whose checkpoint can't load must raise/exit from
    load_optimized_policy() itself (called before the task loop in main()),
    not be silently deferred to the first task."""
    auto_label.OPTIMIZED_POLICIES["_test_broken"] = {
        "version": "v0-test",
        "module_dir": auto_label.OPTIMIZED_POLICIES["optimized"]["module_dir"],
        "module_name": "v12_optimized_inference",
        "description": "synthetic broken policy for testing only",
    }
    try:
        sys.path.insert(0, auto_label.OPTIMIZED_POLICIES["optimized"]["module_dir"])
        import v12_optimized_inference as m
        original_checkpoints = dict(m.CHECKPOINTS)
        original_cache = dict(m._models)
        m.CHECKPOINTS["v10"] = "/nonexistent/path/does-not-exist.pt"
        m._models.pop("v10", None)  # ensure it isn't already cached from another test
        try:
            auto_label.load_optimized_policy("_test_broken")
        except SystemExit as e:
            assert e.code != 0, "a broken checkpoint must exit non-zero"
        else:
            raise AssertionError(
                "expected load_optimized_policy() to fail fast on a broken checkpoint "
                "-- if this passes silently, C1 has regressed"
            )
        finally:
            m.CHECKPOINTS.clear()
            m.CHECKPOINTS.update(original_checkpoints)
            m._models.clear()
            m._models.update(original_cache)
    finally:
        del auto_label.OPTIMIZED_POLICIES["_test_broken"]


# --- 8. no fallback to vanilla --------------------------------------------
def test_no_fallback_on_broken_checkpoint():
    """A failed load must exit the process (SystemExit), never return a
    usable (predict_fn, config) pair that main() could mistake for success
    and fall through to -- there is no return-a-vanilla-callable path
    anywhere in load_optimized_policy()."""
    import inspect
    src = inspect.getsource(auto_label.load_optimized_policy)
    assert "YOLO(" not in src and "DEFAULT_WEIGHTS" not in src, (
        "load_optimized_policy() must not reference the vanilla model path at all -- "
        "any appearance of it would be a sign a fallback was introduced"
    )


# --- 9. future policy naming/versioning doesn't break the CLI design -----
def test_cli_choices_are_registry_driven():
    """--inference-policy's choices come FROM OPTIMIZED_POLICIES, not a
    separately hardcoded list -- adding a future entry to the registry
    must automatically make it a valid CLI choice with no other edits."""
    for name in auto_label.OPTIMIZED_POLICIES:
        args = auto_label.build_parser().parse_args(
            ["--project", "4", "--unlabeled-only", "--inference-policy", name]
        )
        assert args.inference_policy == name

    # Simulate a future registry entry and confirm the parser picks it up
    # automatically, with no changes to build_parser() itself.
    auto_label.OPTIMIZED_POLICIES["_test_future_policy"] = {
        "version": "v-test", "module_dir": ".", "module_name": "x", "description": "x"
    }
    try:
        args = auto_label.build_parser().parse_args(
            ["--project", "4", "--unlabeled-only", "--inference-policy", "_test_future_policy"]
        )
        assert args.inference_policy == "_test_future_policy"
    finally:
        del auto_label.OPTIMIZED_POLICIES["_test_future_policy"]


# --- shared JSON-shape contract (both policies feed the same function) ---
def test_detections_to_result_shape_is_policy_agnostic():
    detections = [("Person", 0.9, [10.0, 20.0, 30.0, 40.0])]
    region = auto_label.detections_to_result(detections, img_w=100, img_h=100,
                                              from_name="label", to_name="image")
    assert len(region) == 1
    v = region[0]["value"]
    assert v["rectanglelabels"] == ["Person"]
    assert v["x"] == 10.0 and v["y"] == 20.0
    assert v["width"] == 20.0 and v["height"] == 20.0
    assert region[0]["score"] == 0.9


# --- 10. FINAL_DELIVERABLE copies remain synchronized ---------------------
def test_final_deliverable_copies_are_synchronized():
    for name in ("auto_label.py", "test_auto_label_policy.py"):
        a = HERE / name
        b = FINAL_DELIVERABLE_SCRIPTS / name
        assert b.exists(), f"{b} is missing"
        assert filecmp.cmp(a, b, shallow=False), (
            f"{a} and {b} have diverged -- keep the two copies in sync "
            f"(project convention, see prior integration/review notes)"
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS: {t.__name__}")
    print("ALL PASSED")
