"""imdlib saves Tmax/Tmin as <year>.GRD; the downloader must accept and rename them."""
import sys

sys.path.insert(0, "scripts")
import download_imd as d  # noqa: E402


def test_uppercase_temperature_file_is_recognised_and_renamed(tmp_path):
    (tmp_path / "tmax").mkdir()
    upper = tmp_path / "tmax" / "1991.GRD"
    upper.write_bytes(b"\0" * d.expected_bytes("tmax", 1991))
    assert d.is_complete(str(tmp_path), "tmax", 1991)
    assert (tmp_path / "tmax" / "1991.grd").exists()


def test_short_file_is_not_complete(tmp_path):
    (tmp_path / "rain").mkdir()
    (tmp_path / "rain" / "2000.grd").write_bytes(b"\0" * 100)
    assert not d.is_complete(str(tmp_path), "rain", 2000)
