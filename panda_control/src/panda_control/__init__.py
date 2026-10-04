from pathlib import Path

# Config files shipped with this package. They are resolved from the source tree,
# so the package must be installed in editable mode (pip install -e .).
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "config")
DEFAULT_INTERFACE_CFG = str(Path(CONFIG_DIR) / "charmander.yml")

# Hand-eye calibration results are setup-specific and not part of the repository.
# Place your own calibration files at these locations (or pass explicit paths).
HANDEYE_CALIBRATION_FILE = str(Path(CONFIG_DIR) / "handeye_calibration" / "transform_camera_2_ee.yaml")
EXTERNAL_HANDEYE_CALIBRATION_FILE = str(Path(CONFIG_DIR) / "handeye_calibration_external" / "transform_camera_2_ee.yaml")
