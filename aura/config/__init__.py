"""YAML configs. stage1.yaml is the single Stage 1 entry point."""

from pathlib import Path

CONFIG_DIR = Path(__file__).parent
STAGE1_YAML = CONFIG_DIR / "stage1.yaml"


def load_stage1():
    import yaml
    with open(STAGE1_YAML) as f:
        return yaml.safe_load(f)
