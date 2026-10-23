import os
import subprocess
import sys

import relyflow


def test_version_is_set():
    assert isinstance(relyflow.__version__, str)
    assert relyflow.__version__ != ""


def test_top_level_import_does_not_load_deep_learning_stack():
    # The data and simulator layers must work without a Keras backend installed.
    code = (
        "import sys, relyflow, relyflow.models; "
        "loaded = [m for m in ('bayesflow', 'keras', 'jax', 'torch') if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_inference_imports_without_a_backend_setting(tmp_path):
    # Keras alone would read its config and fail on a missing TensorFlow;
    # BayesFlow must pick an installed backend first.
    (tmp_path / "keras.json").write_text('{"backend": "tensorflow"}')
    env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    env["KERAS_HOME"] = str(tmp_path)
    subprocess.run([sys.executable, "-c", "import relyflow.inference"], check=True, env=env)
