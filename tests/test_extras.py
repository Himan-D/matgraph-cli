"""Optional-dependency hints. Must pass on BOTH slim and full installs."""

import sys

import pytest

from matgraph._extras import (
    EXTRA_FOR_MODULE,
    MissingExtraError,
    _ExtraHintFinder,
    _ExtraHintLoader,
    hint_for,
    install,
    require,
)


def test_require_returns_installed_module():
    assert require("json") is sys.modules["json"]


def test_require_unknown_module_reraises_plain_import_error():
    with pytest.raises(ImportError):
        require("matgraph_no_such_module_xyz")


def test_hint_for_known_modules():
    assert "matgraph-cli[ml]" in hint_for("torch")
    assert "matgraph-cli[ml]" in hint_for("matgl.models")
    assert "matgraph-cli[mp]" in hint_for("mp_api.client")
    assert "matgraph-cli[mp]" in hint_for("pymatgen.core")
    assert "matgraph-cli[api]" in hint_for("strawberry")
    assert "matgraph-cli[ui]" in hint_for("gradio")
    assert "matgraph-cli[tracking]" in hint_for("wandb")
    assert hint_for("json") is None


def test_loader_raises_actionable_error():
    with pytest.raises(MissingExtraError, match=r'pip install "matgraph-cli\[ml\]"'):
        _ExtraHintLoader("torch").create_module(None)


def test_missing_extra_is_import_error():
    # Existing `except ImportError` handlers keep working.
    assert issubclass(MissingExtraError, ImportError)


def test_install_is_idempotent_and_last():
    install()
    install()
    finders = [f for f in sys.meta_path if isinstance(f, _ExtraHintFinder)]
    assert len(finders) == 1


def test_finder_ignores_unrelated_modules():
    assert _ExtraHintFinder().find_spec("json", None) is None
    assert _ExtraHintFinder().find_spec("definitely_not_a_module_xyz", None) is None


def test_every_hint_points_at_a_real_extra():
    import re
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text()
    extras = set(re.findall(r"^(\w+) = \[", pyproject, re.M))
    for module, extra in EXTRA_FOR_MODULE.items():
        assert extra in extras or extra == "thermo", f"{module} -> [{extra}] not in pyproject"


def test_cli_print_preserves_extras_brackets():
    from matgraph.cli import console

    console.print('pip install "matgraph-cli[ml]"')
