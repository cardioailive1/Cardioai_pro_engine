"""Run: python -m training.mayo.test_lvef_dataset  (from backend/; needs torch + pydicom)"""
import os, tempfile
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from training.mayo.audit_gate import AutoAuditRequired, AuditedLabels
from training.mayo.lvef_dataset import build_lvef_examples, DicomCineLVEFDataset
from training.mayo.radiology_link import link_reports_to_images
from training.train_mobile_echo import train_mobile_echo_net


def write_dcm(path, sop, n, arr):
    m = FileMetaDataset(); m.TransferSyntaxUID = ExplicitVRLittleEndian
    m.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"; m.MediaStorageSOPInstanceUID = sop
    d = Dataset(); d.file_meta = m; d.SOPInstanceUID = sop; d.InstanceNumber = n
    d.Rows, d.Columns = arr.shape; d.SamplesPerPixel = 1; d.PhotometricInterpretation = "MONOCHROME2"
    d.BitsAllocated = 16; d.BitsStored = 16; d.HighBit = 15; d.PixelRepresentation = 0
    d.PixelData = arr.astype("<u2").tobytes(); d.save_as(path, enforce_file_format=True)


def t():
    tmp = tempfile.mkdtemp(); inv, reports, paths, reviewed = [], [], {}, {}
    rng = np.random.default_rng(0)
    for k in range(12):
        acc = f"ACC{k}"; ef = 30 + 3 * k
        reports.append({"ACCESSION_NBR": acc}); reviewed[acc] = {"lvef": ef, "category": "x"}
        for n in range(6):
            sop = generate_uid(); p = os.path.join(tmp, sop + ".dcm")
            img = (rng.random((48, 48)) * 1000 + ef * 10).astype(np.uint16)
            write_dcm(p, sop, n, img); paths[sop] = p
            inv.append({"DICOM_ACCESSION_NUMBER": acc, "STUDY_INSTANCE_UID": "S" + acc, "SERIES_INSTANCE_UID": "SE" + acc,
                        "SOP_INSTANCE_UID": sop, "IMAGE_DEID_STATUS": "Ready"})
    reviewed.pop("ACC11")  # unreviewed study must be excluded
    linked = link_reports_to_images(reports, inv)["linked"]
    try:                                    # labels that did not come from the audited pipeline are refused
        build_lvef_examples(linked, reviewed); assert False
    except AutoAuditRequired:
        pass
    ex = build_lvef_examples(linked, AuditedLabels(reviewed))
    assert len(ex) == 11 and all(len(e["sops"]) == 6 for e in ex)
    ds = DicomCineLVEFDataset(ex, lambda s: paths[s], n_frames=4, img_size=64)
    v, y = ds[0]
    assert v.shape == (4, 1, 64, 64) and 0 <= float(v.min()) and float(v.max()) <= 1 and abs(float(y) - 0.30) < 1e-6
    res = train_mobile_echo_net(n_epochs=1, batch_size=4, dataset=ds, n_val=2)
    assert len(res["train_losses"]) > 0
    print("all ok")

if __name__ == "__main__":
    t()
