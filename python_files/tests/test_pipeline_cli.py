"""The scheduled refresh needs a predictable entry point and a predictable
output filename. The existing __main__ block has neither.
"""
import os
import subprocess
import sys

PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "target_and_calculated_pipeline.py")
# Asserted rather than probed, so moving the script fails here loudly instead of
# quietly resolving to a path that never existed.
assert os.path.exists(PIPELINE), f"pipeline script not found at {PIPELINE}"


def test_module_imports_without_running_a_build():
    """Importing must stay side-effect free so the module is testable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcp", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "build_final_dataset")
    assert hasattr(mod, "main")


def test_cli_reports_usage_and_exits_nonzero_without_required_args():
    proc = subprocess.run([sys.executable, PIPELINE], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "--base-path" in (proc.stderr + proc.stdout)


def test_out_name_is_deterministic_when_supplied():
    """A clock-based filename cannot be handed to the next stage of a scheduled
    run, so --out-name must win over the HHMM default.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcp", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.output_path("/tmp", None).startswith("/tmp")
    assert mod.output_path("/tmp", "Final_Target_Calc_current.csv") == os.path.join(
        "/tmp", "Final_Target_Calc_current.csv")
