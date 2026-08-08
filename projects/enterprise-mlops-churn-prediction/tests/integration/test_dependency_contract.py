"""Verify the TensorFlow serving and training dependency compatibility contract."""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
CRITICAL_PINS = {
    "tensorflow": "2.13.0",
    "keras": "2.13.1",
    "numpy": "1.24.3",
    "pandas": "2.0.3",
    "scikit-learn": "1.3.0",
    "typing-extensions": "4.5.0",
    "grpcio": "1.64.3",
    "cryptography": "41.0.7",
}


def _pinned_requirements(path: Path):
    """Return normalized package names and exact pins from a requirements file."""
    pins = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        exact = [item.version for item in requirement.specifier if item.operator == "=="]
        if exact:
            pins[requirement.name.lower()] = exact[0]
    return pins


def test_root_and_api_requirements_share_critical_runtime_pins():
    """Training and serving must not resolve different critical runtimes."""
    root_pins = _pinned_requirements(ROOT / "requirements.txt")
    api_pins = _pinned_requirements(ROOT / "docker/requirements.api.txt")

    for package, expected in CRITICAL_PINS.items():
        assert root_pins.get(package) == expected
        assert api_pins.get(package) == expected


def test_declared_typing_extensions_satisfies_tensorflow_213_contract():
    """Protect the Linux constraint that originally caused CI resolution failure."""
    root_pins = _pinned_requirements(ROOT / "requirements.txt")
    declared = Version(root_pins["typing-extensions"])

    assert declared in SpecifierSet(">=3.6.6,<4.6.0")


def test_installed_tensorflow_backend_accepts_installed_typing_extensions():
    """Check the active platform backend metadata, not only a hard-coded range."""
    installed_typing_extensions = Version(version("typing-extensions"))
    backends_checked = []

    for package in (
        "tensorflow-macos",
        "tensorflow-cpu-aws",
        "tensorflow-intel",
        "tensorflow-cpu",
        "tensorflow",
    ):
        try:
            metadata = distribution(package)
        except PackageNotFoundError:
            continue

        for raw_requirement in metadata.requires or []:
            requirement = Requirement(raw_requirement)
            if requirement.name.lower() != "typing-extensions":
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            backends_checked.append(package)
            assert installed_typing_extensions in requirement.specifier

    assert backends_checked, "No installed TensorFlow backend declared typing-extensions"


def test_critical_frameworks_import_in_a_fresh_process():
    """Exercise the resolved environment without contaminating the pytest process."""
    command = [
        sys.executable,
        "-c",
        (
            "import tensorflow as tf; import keras; import fastapi; import mlflow; "
            "assert tf.__version__ == '2.13.0'; "
            "assert keras.__version__ == '2.13.1'"
        ),
    ]
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
