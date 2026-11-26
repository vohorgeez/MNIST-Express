"""Garde-fou de la règle de dépendance de l'architecture hexagonale.

Le coeur (core/, preprocessing.py, inference.py) ne doit dépendre ni de l'interface, ni de la sérialisation, ni du système de fichiers, ni des adaptateurs. Si quelqu'un ajoute un `import joblib` dans le coeur, ce test casse.
"""
import ast
from pathlib import Path
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "mnist_express"
CORE_FILES = sorted((PACKAGE / "core").glob("*.py")) + [
    PACKAGE / "preprocessing.py",
    PACKAGE / "inference.py",
]
FORBIDDEN_ROOTS = {
    "streamlit",
    "streamlit_drawable_canvas",
    "matpllotlib",
    "joblib",
    "pickle",
    "json",
    "os",
    "pathlib",
    "shutil",
}

def imported_modules(source: str) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append("." * node.level + (node.module or ""))
    return modules

def forbidden_imports(source: str) -> list[str]:
    offending = []
    for module in imported_modules(source):
        parts = module.split(".")
        root = parts[0]
        if root in FORBIDDEN_ROOTS or "adapters" in parts or "bootstrap" in parts:
            offending.append(module)
    return offending

@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_core_does_not_depend_on_infrastructure(path):
    assert forbidden_imports(path.read_text(encoding="utf-8")) == []

def test_streamlit_app_never_touches_storage_directly():
    """Règle de docs/ARCHITECTURE.md : l'app ne charge jamais un joblib elle-même."""
    source = (PACKAGE.parent / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    roots = {module.split(".")[0] for module in imported_modules(source)}

    assert roots.isdisjoint({"joblib", "pickle", "json", "os"})

def test_the_guard_actually_detects_violations():
    source = "import joblib\nfrom ..adapters.clock import utc_now_iso\nimport numpy as np\n"

    assert forbidden_imports(source) == ["joblib", "..adapters.clock"]