from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200

    data = r.json()
    assert data["message"] == "CT inference service is running"
    assert (
        data["storage_mode"] == "no temp files, no cache files, no visualization files"
    )
    assert "/predict/path" in data["endpoints"]
    assert "/predict/dicom-dir" in data["endpoints"]
    assert "/health" in data["endpoints"]
    assert "model_ready" in data


def test_health():
    r = client.get("/health")
    assert r.status_code == 200

    data = r.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "generator_weights_exists" in data
    assert "classifier_weights_exists" in data
    assert "config_exists" in data
    assert "model_ready" in data
    assert "init_error" in data


def test_predict_path_missing_ct():
    r = client.post(
        "/predict/path",
        json={
            "ct_path": "/does/not/exist/sample.nii.gz",
        },
    )
    assert r.status_code == 404
    assert "CT path not found" in r.json()["detail"]


def test_predict_path_rejects_bad_extension(tmp_path):
    bad_file = tmp_path / "sample.txt"
    bad_file.write_text("not a medical volume")

    r = client.post(
        "/predict/path",
        json={
            "ct_path": str(bad_file),
        },
    )
    assert r.status_code == 400
    assert "Unsupported CT volume type" in r.json()["detail"]


def test_predict_path_missing_lung_mask(tmp_path):
    ct_file = tmp_path / "sample.nii.gz"
    ct_file.write_text("placeholder")

    missing_mask = tmp_path / "missing_mask.nii.gz"

    r = client.post(
        "/predict/path",
        json={
            "ct_path": str(ct_file),
            "lung_mask_path": str(missing_mask),
        },
    )
    assert r.status_code == 404
    assert "Lung mask path not found" in r.json()["detail"]


def test_predict_dicom_dir_missing_dir():
    r = client.post(
        "/predict/dicom-dir",
        json={
            "dicom_dir": "/does/not/exist/dicom_case",
        },
    )
    assert r.status_code == 404
    assert "DICOM directory not found" in r.json()["detail"]


def test_predict_dicom_dir_rejects_file_instead_of_dir(tmp_path):
    fake_file = tmp_path / "not_a_dir"
    fake_file.write_text("hello")

    r = client.post(
        "/predict/dicom-dir",
        json={
            "dicom_dir": str(fake_file),
        },
    )
    assert r.status_code == 400
    assert "is not a directory" in r.json()["detail"]


def test_predict_dicom_dir_missing_lung_mask(tmp_path):
    dicom_dir = tmp_path / "dicom_case"
    dicom_dir.mkdir()

    missing_mask = tmp_path / "missing_mask.nii.gz"

    r = client.post(
        "/predict/dicom-dir",
        json={
            "dicom_dir": str(dicom_dir),
            "lung_mask_path": str(missing_mask),
        },
    )
    assert r.status_code == 404
    assert "Lung mask path not found" in r.json()["detail"]
