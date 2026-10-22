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
