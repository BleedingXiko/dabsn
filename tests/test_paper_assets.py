import json
import py_compile
import subprocess
from pathlib import Path


def test_paper_source_and_generator_are_self_contained():
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper1"
    py_compile.compile(str(paper / "generate_figures.py"), doraise=True)
    assert (paper / "main.tex").is_file()
    assert (paper / "main.bbl").is_file()
    assert (paper / "main.pdf").is_file()
    assert len(list((paper / "figs").glob("fig*.pdf"))) == 3
    required = {
        "copy_length.csv",
        "mqar_length.csv",
        "keyvalue_length.csv",
        "ablation_knockout.csv",
        "a5_word_results.csv",
        "a5_seed1_linear_ablation.csv",
        "headline_results_paper1.csv",
    }
    assert required <= {path.name for path in (paper / "ancillary_results").glob("*.csv")}
    assert not (root / ".zenodo.json").exists()
    zenodo = json.loads((paper / "zenodo-metadata.json").read_text(encoding="utf-8"))
    assert zenodo["upload_type"] == "publication"
    assert zenodo["publication_type"] == "preprint"
    assert zenodo["title"].startswith("One Layer, Both Gaps:")
    repository = "https://github.com/BleedingXiko/dabsn"
    assert repository in (paper / "main.tex").read_text(encoding="utf-8")
    assert any(
        row["identifier"] == repository
        and row["relation"] == "isSupplementedBy"
        and row["resource_type"] == "software"
        for row in zenodo["related_identifiers"]
    )
    dry_run = subprocess.run(
        ["make", "--dry-run"],
        cwd=paper,
        text=True,
        capture_output=True,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
