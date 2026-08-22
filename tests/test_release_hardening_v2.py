import gzip
import io
import tarfile
from pathlib import Path

from scripts.build_release import _normalize_sdist_archive

ROOT = Path(__file__).resolve().parents[1]


def _write_test_sdist(path: Path, *, mtime: int, uid: int, owner: str) -> None:
    payload = b"canonical package payload\n"
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename=path.name, mode="wb", fileobj=raw, mtime=mtime) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        root = tarfile.TarInfo("dabsn-0.1.6")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        root.uid = uid
        root.gid = uid
        root.uname = owner
        root.gname = owner
        root.mtime = float(mtime) + 0.125
        root.pax_headers = {"mtime": str(root.mtime)}
        archive.addfile(root)

        member = tarfile.TarInfo("dabsn-0.1.6/payload.txt")
        member.size = len(payload)
        member.mode = 0o644
        member.uid = uid
        member.gid = uid
        member.uname = owner
        member.gname = owner
        member.mtime = float(mtime) + 0.875
        member.pax_headers = {"mtime": str(member.mtime)}
        archive.addfile(member, io.BytesIO(payload))


def test_release_builder_uses_tracked_archive_and_never_copies_user_artifacts():
    text = (ROOT / "scripts/build_release.py").read_text(encoding="utf-8")
    assert "TemporaryDirectory" in text
    assert '"archive", "--format=tar", "HEAD"' in text
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in text
    assert "completely clean tracked and untracked worktree" in text
    assert "copytree" not in text
    assert "rmtree" not in text
    assert "unlink" not in text
    assert "SOURCE_DATE_EPOCH" in text
    assert "range(2)" in text
    assert "byte-for-byte reproducible" in text
    assert "_normalize_sdist_archive" in text
    assert "dabsn-reproducible-build-record" in text


def test_sdist_normalization_is_byte_reproducible_and_preserves_payload(tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_test_sdist(first, mtime=100, uid=501, owner="first")
    _write_test_sdist(second, mtime=200, uid=1000, owner="second")

    canonical_mtime = 1_700_000_000
    _normalize_sdist_archive(first, mtime=canonical_mtime)
    _normalize_sdist_archive(second, mtime=canonical_mtime)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
        assert archive.extractfile("dabsn-0.1.6/payload.txt").read() == (
            b"canonical package payload\n"
        )
    assert {member.mtime for member in members} == {canonical_mtime}
    assert {member.uid for member in members} == {0}
    assert {member.gid for member in members} == {0}
    assert {member.uname for member in members} == {""}
    assert {member.gname for member in members} == {""}
    assert all(not member.pax_headers for member in members)


def test_blackwell_runner_uses_the_authoritative_safe_training_and_confidence_gate():
    text = (ROOT / "scripts/benchmark_dabsn2_release.py").read_text(encoding="utf-8")
    assert "model.forward_with_terms(ids)" in text
    assert "for term in result.loss_terms" in text
    assert "clip_grad_norm(model, 1.0)" in text
    assert "model.post_optimizer_step(step_applied=True)" in text
    assert "foreach=False" in text and "fused=False" in text
    assert "report.mean_seconds + report.ci95_seconds" in text
    assert '"conservative_95_tokens_per_second"' in text


def test_topology_restore_gate_explicitly_authorizes_resharding_without_rng_replay():
    text = (ROOT / "scripts/dcp_topology_restore_gate.py").read_text(encoding="utf-8")
    assert "restore_rng=False" in text
    assert "allow_topology_change=True" in text
