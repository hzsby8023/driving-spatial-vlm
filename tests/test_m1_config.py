import json

import pytest

from vla.config import apply_overrides, load_json_config


DEFAULTS = {"model": {"name": "base", "rank": 8}, "seed": 42}


def test_load_and_override(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"rank": 16}}), encoding="utf-8")
    config = load_json_config(path, DEFAULTS)
    assert config == {"model": {"name": "base", "rank": 16}, "seed": 42}
    assert apply_overrides(config, {"model.name": "new", "seed": None})["model"]["name"] == "new"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"trianing": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_json_config(path, DEFAULTS)
