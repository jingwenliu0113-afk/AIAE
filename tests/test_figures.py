"""The committed figures are current, and the check that says so means it.

``scripts/73_figures.py --check`` is what CI runs. It draws nothing, so it gives
the same answer on ubuntu as on the Mac that drew the PNGs -- which is the
point: drawing again and comparing bytes only works on the machine that drew
them, and on ubuntu-latest four unchanged figures came out different. These
tests hold the check to failing when it should, and hold the manifest's PNG
digests to being true where they were written.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def figures():
    spec = importlib.util.spec_from_file_location(
        "figures_generator", ROOT / "scripts" / "73_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed(figures):
    return json.loads((ROOT / "reports" / "figures" / figures.MANIFEST)
                      .read_text(encoding="utf-8"))


@pytest.fixture()
def tree(committed, tmp_path):
    """A copy of exactly what the check reads: the script, the reports, the figures."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts" / "73_figures.py", tmp_path / "scripts")
    for entry in committed["figures"].values():
        for rel in entry["sources"]:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / "reports" / "figures", tmp_path / "reports" / "figures")
    return tmp_path


# -- the committed tree ------------------------------------------------------


def test_the_committed_figures_are_current(figures):
    assert figures.check() == []


def test_the_command_ci_runs_exits_zero(figures, capsys):
    assert figures.main(["--check"]) == 0
    assert "figures are current" in capsys.readouterr().out


def test_every_figure_records_the_reports_it_read(figures, committed):
    """Recorded by load() as it opens them, so none can be left off by hand."""
    assert set(committed["figures"]) == set(figures.FIGURES)
    for name, entry in committed["figures"].items():
        assert entry["sources"], name
        assert all(rel.startswith("data/reports/") for rel in entry["sources"]), name


# -- the check fails when it should ------------------------------------------


def test_a_copy_of_the_tree_is_current(figures, tree):
    """The reverse tests below start from a tree the check accepts."""
    assert figures.check(root=tree) == []


def test_a_report_that_moved_without_a_redraw_is_caught(figures, committed, tree):
    rel = sorted(committed["figures"][figures.FIGURES[0]]["sources"])[0]
    (tree / rel).write_bytes((tree / rel).read_bytes() + b"\n")
    problems = figures.check(root=tree)
    assert f"{figures.FIGURES[0]}: {rel} changed since it was drawn" in problems


def test_a_generator_edited_without_a_redraw_is_caught(figures, tree):
    script = tree / "scripts" / "73_figures.py"
    script.write_text(script.read_text(encoding="utf-8") + "\n# edited\n",
                      encoding="utf-8")
    assert "scripts/73_figures.py changed since the figures were drawn" in \
        figures.check(root=tree)


def test_a_png_that_is_not_the_one_drawn_is_caught(figures, tree):
    """What the byte comparison used to guard against, asked without drawing."""
    name = figures.FIGURES[-1]
    png = tree / "reports" / "figures" / name
    data = bytearray(png.read_bytes())
    data[-1] ^= 1
    png.write_bytes(bytes(data))
    assert f"{name} is not the PNG {figures.MANIFEST} describes" in \
        figures.check(root=tree)


def test_a_missing_figure_is_caught(figures, tree):
    name = figures.FIGURES[1]
    (tree / "reports" / "figures" / name).unlink()
    assert f"{name} is missing" in figures.check(root=tree)


def test_a_missing_manifest_is_a_failure_not_a_pass(figures, tree):
    (tree / "reports" / "figures" / figures.MANIFEST).unlink()
    assert figures.check(root=tree) == [
        f"reports/figures/{figures.MANIFEST} does not exist"]


def test_a_manifest_that_forgets_a_figure_is_caught(figures, tree):
    path = tree / "reports" / "figures" / figures.MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["figures"][figures.FIGURES[2]]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(p.startswith(f"{figures.MANIFEST} names ")
               for p in figures.check(root=tree))


def test_the_cli_exits_nonzero_on_a_stale_tree(figures, tree, monkeypatch, capsys):
    (tree / "scripts" / "73_figures.py").write_text("# not the generator\n",
                                                    encoding="utf-8")
    monkeypatch.setattr(figures, "ROOT", tree)
    assert figures.main(["--check"]) == 1
    assert "STALE: scripts/73_figures.py changed" in capsys.readouterr().out


# -- drawing -----------------------------------------------------------------


def test_drawing_here_records_the_same_inputs_as_the_commit(figures, committed,
                                                            tmp_path):
    """Runs everywhere, CI included: the script draws on this platform, and
    what it records about its inputs does not depend on the platform even
    where the PNG bytes do."""
    drawn = figures.draw(tmp_path)
    assert drawn["generator"] == committed["generator"]
    for name in figures.FIGURES:
        assert (tmp_path / name).is_file(), name
        assert drawn["figures"][name]["sources"] == \
            committed["figures"][name]["sources"], name


def test_where_the_figures_were_drawn_the_bytes_are_the_committed_ones(
        figures, committed, tmp_path):
    """The manifest's PNG digests are a receipt, so something has to read it."""
    import matplotlib

    here = {"matplotlib": matplotlib.__version__,
            "platform": f"{platform.system()} {platform.machine()}"}
    if here != committed["drawn_with"]:
        pytest.skip(f"drawn with {committed['drawn_with']}; this is {here}")
    drawn = figures.draw(tmp_path)
    for name in figures.FIGURES:
        assert drawn["figures"][name]["png_sha256"] == \
            committed["figures"][name]["png_sha256"], name
