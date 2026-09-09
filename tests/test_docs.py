"""Small source and checker guards; full numerical examples run through Quarto."""

import ast
import hashlib
import importlib.util
import inspect
import json
import re
from html import escape
from pathlib import Path

import crabbymetrics as cm
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(), reason="Documentation sources are excluded from source distributions"
)


def test_executable_constructor_keywords_match_public_api():
    failures = []
    for page in DOCS.rglob("*.qmd"):
        source = "\n".join(re.findall(r"```[{]python[^\n]*[}]\n(.*?)```",
                                     page.read_text(), flags=re.DOTALL))
        tree = ast.parse(source, filename=str(page))
        modules = set()
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.asname or alias.name for alias in node.names
                               if alias.name == "crabbymetrics")
            elif isinstance(node, ast.ImportFrom) and node.module == "crabbymetrics":
                names.update({alias.asname or alias.name: alias.name for alias in node.names})
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = None
            if isinstance(node.func, ast.Name):
                name = names.get(node.func.id)
            elif (isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id in modules):
                name = node.func.attr
            cls = getattr(cm, name, None) if name else None
            if not inspect.isclass(cls) or any(k.arg is None for k in node.keywords):
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                continue
            try:
                inspect.signature(cls).bind_partial(
                    *[None] * len(node.args), **{k.arg: None for k in node.keywords}
                )
            except TypeError as error:
                failures.append(f"{page.relative_to(ROOT)}: {name}: {error}")
    assert not failures, "\n".join(failures)


def test_api_overview_covers_exported_classes():
    overview = (DOCS / "api.qmd").read_text()
    missing = [name for name in dir(cm) if not name.startswith("_")
               and inspect.isclass(getattr(cm, name))
               and f"cm.{name}" not in overview]
    assert not missing


def test_doc_helpers_parse_and_expose_constructor():
    for helper in DOCS.rglob("*.py"):
        ast.parse(helper.read_text(), filename=str(helper))
    helper = DOCS / "reference/_api_doc_utils.py"
    tree = ast.parse(helper.read_text(), filename=str(helper))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "public_methods")
    # Test this pure helper without requiring the notebook display stack in CI.
    scope = {"inspect": inspect, "escape": escape}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(helper), "exec"), scope)  # noqa: S102
    rows = scope["public_methods"](cm.OLS)
    assert rows[0] == ["<code>OLS()</code>"]
    assert any("fit(" in row[0] for row in rows)


def test_docs_have_no_private_paths_or_obsolete_scaffolds():
    for page in DOCS.rglob("*.qmd"):
        source = page.read_text()
        assert "/Users/" not in source, page
        assert "ding_w_source" not in source, page
        assert "This page mirrors `examples/" not in source, page
        assert "embed-resources: true" not in source, page
        assert r"\(" not in source and r"\[" not in source, page
    assert not (DOCS / "_generate_class_reference.py").exists()
    assert not (DOCS / "specs/sparse-rotations.qmd").exists()


def test_ding_data_checksums():
    checksums = {
        "card1995.csv": "102b8856ed50ba53696879f0ed584917",
        "cps1re74.csv": "984ef72e2187423feedc018c0ddd92a8",
        "jobsdata.csv": "e249939c9b58c51e52ea964482fd085b",
        "nhanes_bmi.csv": "8e5d1e1c1dfb44ecc64d2de5cb0f857d",
        "Penn46_ascii.txt": "84233c5c8a31ac08d235a49ceb117fc2",
        "resume.csv": "a09b2afb2a0a6271c39a9bef98d60281",
        "star.dta": "dff69b726e37eb1f44a31065aae6d348",
        "ZeaMays.csv": "9cef1c99b8ed51cdda09f4fafcd2a498",
    }
    for name, checksum in checksums.items():
        assert hashlib.md5((DOCS / "data/ding" / name).read_bytes()).hexdigest() == checksum


def test_rendered_site_checker_finds_links_anchors_and_search_gaps(tmp_path):
    spec = importlib.util.spec_from_file_location("check_site", DOCS / "_check_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "index.html").write_text(
        '<h1 id="top">Docs</h1><a href="reference.html#api">API</a>'
        '<img src="missing.png"><a href="index.html#missing">Bad anchor</a>'
    )
    (tmp_path / "reference.html").write_text('<h1 id="api">API</h1>')
    (tmp_path / "search.json").write_text(json.dumps([{"href": "index.html"}]))
    result = module.check_site(tmp_path)
    assert result["pages"] == 2
    assert any("missing.png" in error for error in result["errors"])
    assert any("#missing" in error for error in result["errors"])
    assert any("not indexed: reference.html" in error for error in result["errors"])
    (tmp_path / "missing.png").write_bytes(b"png")
    (tmp_path / "index.html").write_text('<h1 id="top">Docs</h1><a href="reference.html#api">API</a>')
    (tmp_path / "search.json").write_text(json.dumps([
        {"href": "index.html"}, {"href": "reference.html#api"}
    ]))
    assert not module.check_site(tmp_path)["errors"]
