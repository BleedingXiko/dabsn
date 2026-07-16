from pathlib import Path


def test_forbidden_surfaces_absent():
    root = Path(__file__).resolve().parents[1]
    public = "\n".join(path.read_text(errors="ignore") for path in (root / "src" / "dabsn").rglob("*.py"))
    assert "import tla" not in public
    assert "from tla" not in public
    assert "absn_codec" not in public
    assert not (root / "src" / "dabsn" / "codec").exists()
