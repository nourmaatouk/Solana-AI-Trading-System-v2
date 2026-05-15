"""
Top-level conftest.py for services/models/.

Adds module roots to sys.path so that tests can be collected from a parent
directory (e.g. `pytest services/models/`).

IMPORTANT — `src` namespace collision:
  Both `nlp/src/` and `spoofing_detection/src/` export a top-level `src`
  package.  Python caches the first one found in `sys.modules['src']`, which
  means they CANNOT coexist in the same pytest session.  Run them as separate
  invocations:

    pytest services/models/classification/tests/
    pytest services/models/nlp/tests/
    pytest services/models/fusion/tests/ services/models/spoofing_detection/tests/
    pytest services/models/tests/
"""
import sys
from pathlib import Path

import pytest

_MODELS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MODELS_ROOT.parents[1]

# The classification orderbook tests import `services.ingest...`, which only
# resolves in a clean CI checkout when the repository root is importable.
_repo_root = str(_REPO_ROOT)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# fusion/src is added so that `from fusion_aggregator import ...` works
# without a `from src.xxx` style (the fusion package uses flat imports).
_fusion_src = str(_MODELS_ROOT / "fusion" / "src")
if _fusion_src not in sys.path:
    sys.path.insert(0, _fusion_src)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail fast when nlp/ and spoofing_detection/ tests are collected together.

    Both packages export a top-level `src` Python package.  Mixing them in a
    single pytest session causes whichever `src` is imported first to shadow
    the other, producing confusing ImportErrors or silent wrong-module usage.
    """
    item_paths = [str(item.fspath) for item in session.items]
    has_nlp    = any("nlp" + "/" in p or "nlp\\" in p for p in item_paths)
    has_spoof  = any("spoofing_detection" in p for p in item_paths)
    if has_nlp and has_spoof:
        pytest.exit(
            "\n\nFATAL: nlp/ and spoofing_detection/ cannot run in the same pytest session.\n"
            "Both packages register a top-level 'src' module; whichever is imported first\n"
            "silently shadows the other.  Run them separately:\n\n"
            "  pytest services/models/nlp/tests/\n"
            "  pytest services/models/fusion/tests/ "
            "services/models/spoofing_detection/tests/\n",
            returncode=4,
        )
