"""Tests for the standalone echo LVEF path (random-weight models: plumbing only)."""
import io
import json

import numpy as np
import pytest
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from imaging_models.echo_checkpoint import approve_echo_checkpoint, save_echo_checkpoint
from imaging_models.mobilenet_echo_model import MobileEchoNet

GOOD = {"n_studies": 150, "correlation": 0.93, "mae_lvef_points": 4.1, "reference": "cardiologist"}


@pytest.fixture()
def wdir(tmp_path):
    save_echo_checkpoint(MobileEchoNet(variant="small").eval(), str(tmp_path / "echo_lvef_v1.pt"), n_frames=8, img_size=64,
                         trained_on="synthetic smoke test", validation=GOOD)
    return tmp_path


def _set(wdir, **kw):
    mp = wdir / "echo_lvef_v1.json"; m = json.loads(mp.read_text()); m.update(kw); mp.write_text(json.dumps(m))


def test_approval_gates(wdir):
    p = str(wdir / "echo_lvef_v1.pt")
    with pytest.raises(ValueError, match="synthetic"):
        approve_echo_checkpoint(p, "x", "x")
    _set(wdir, trained_on="Mayo echo", validation={**GOOD, "correlation": 0.8})
    with pytest.raises(ValueError, match="correlation"):
        approve_echo_checkpoint(p, "x", "x")
    _set(wdir, validation={**GOOD, "reference": "auto-labels"})
    with pytest.raises(ValueError, match="cardiologist"):
        approve_echo_checkpoint(p, "x", "x")
    _set(wdir, validation={**GOOD, "n_studies": 20})
    with pytest.raises(ValueError, match="at least"):
        approve_echo_checkpoint(p, "x", "x")
    _set(wdir, validation=GOOD)
    assert approve_echo_checkpoint(p, "Dr Test", "ok")["approved_for_live"] is True


def test_runtime_unapproved_none_approved_works_and_is_not_fused(wdir):
    from inference.echo_runtime import build_echo_runtime
    assert build_echo_runtime(str(wdir)) is None
    _set(wdir, trained_on="Mayo echo"); approve_echo_checkpoint(str(wdir / "echo_lvef_v1.pt"), "Dr Test", "ok")
    run = build_echo_runtime(str(wdir))
    out = run(np.random.rand(12, 80, 90).astype(np.float32))
    assert out["status"] == "ok" and 5 <= out["lvef_percent"] <= 85 and out["fused_into_risk_score"] is False
    assert out["lvef_category"] in ("reduced_le40", "mildly_reduced_41_49", "preserved_ge50")
    short = run(np.random.rand(3, 80, 90))
    assert short["status"] == "not_run"


def test_registry_without_checkpoint_reports_awaiting(monkeypatch, tmp_path):
    monkeypatch.setattr("inference.echo_runtime.WEIGHTS_DIR", str(tmp_path))
    from inference.models import ModelRegistry
    out = ModelRegistry().predict_echo(np.random.rand(10, 32, 32))
    assert out["status"] == "awaiting_trained_model" and out["fused_into_risk_score"] is False


def _dcm_bytes(frames):
    m = FileMetaDataset(); m.TransferSyntaxUID = ExplicitVRLittleEndian
    d = Dataset(); d.file_meta = m; d.Modality = "US"; d.SOPInstanceUID = generate_uid()
    d.Rows, d.Columns = frames.shape[1:]; d.NumberOfFrames = str(frames.shape[0]) if frames.shape[0] > 1 else "1"
    d.BitsAllocated = 16; d.BitsStored = 16; d.HighBit = 15; d.PixelRepresentation = 0
    d.SamplesPerPixel = 1; d.PhotometricInterpretation = "MONOCHROME2"; d.PixelData = frames.astype("<u2").tobytes()
    buf = io.BytesIO(); d.save_as(buf, enforce_file_format=False) if hasattr(d, "save_as") else None
    return buf.getvalue()


def test_extract_frames_multiframe_and_single():
    from integrations.dicom import DICOMService
    svc = DICOMService()
    assert svc.extract_frames(_dcm_bytes((np.random.rand(10, 16, 16) * 1000))).shape == (10, 16, 16)
    assert svc.extract_frames(_dcm_bytes((np.random.rand(1, 16, 16) * 1000))) is None


def test_export_script(tmp_path):
    from scripts.export_echo_model import main
    r = main(["--random-init", "--out", str(tmp_path), "--img-size", "64"])
    assert (tmp_path / "echo_lvef.ts.pt").exists() and r["parameters"] > 1e6 and r["float_ms_cpu"] > 0
    assert torch.jit.load(str(tmp_path / "echo_lvef.ts.pt"))(torch.randn(1, 8, 1, 64, 64)).shape == (1, 1)
