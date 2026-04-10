from pathlib import Path
from typing import Dict, List, Optional

# just a comment

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.model_loader import (
    get_service_status,
    initialize_models,
    predict_from_dicom_dir,
    predict_from_uploaded_dicom_zip_bytes,
    predict_from_uploaded_volume_bytes,
    predict_from_uploaded_volume_zip_bytes,
    predict_from_volume_path,
)

app = FastAPI(title="CT Inference API", version="2.3.0")


class PathPredictRequest(BaseModel):
    ct_path: str
    lung_mask_path: Optional[str] = None
    annotations_world_xyz: Optional[List[List[float]]] = None
    annotation_diameters_mm: Optional[List[float]] = None
    seriesuid: Optional[str] = None


class DicomDirPredictRequest(BaseModel):
    dicom_dir: str
    lung_mask_path: Optional[str] = None
    annotations_world_xyz: Optional[List[List[float]]] = None
    annotation_diameters_mm: Optional[List[float]] = None
    seriesuid: Optional[str] = None


@app.on_event("startup")
def startup_event() -> None:
    initialize_models()


@app.get("/")
def root() -> Dict:
    status = get_service_status()
    return {
        "message": "CT inference service is running",
        "device": status["device"],
        "weights_dir": status["weights_dir"],
        "accepted_inputs": [
            "DICOM directory path",
            ".nii",
            ".nii.gz",
            ".mhd",
            ".mha",
            "uploaded volume file",
            "uploaded volume zip",
            "uploaded DICOM zip",
        ],
        "storage_mode": "temporary upload staging with automatic cleanup",
        "model_ready": status["model_ready"],
        "endpoints": [
            "/health",
            "/predict/path",
            "/predict/dicom-dir",
            "/predict/upload-volume",
            "/predict/upload-volume-zip",
            "/predict/upload-dicom-zip",
        ],
    }


@app.get("/health")
def health() -> Dict:
    return get_service_status()


@app.post("/predict/path")
def predict_path(req: PathPredictRequest) -> Dict:
    ct_path = Path(req.ct_path)
    if not ct_path.exists():
        raise HTTPException(status_code=404, detail=f"CT path not found: {req.ct_path}")

    if req.lung_mask_path:
        lung_mask_path = Path(req.lung_mask_path)
        if not lung_mask_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Lung mask path not found: {req.lung_mask_path}",
            )

    suffix = "".join(ct_path.suffixes).lower()
    allowed = {".nii", ".nii.gz", ".mha", ".mhd"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported CT volume type: {suffix}. Use one of {sorted(allowed)}",
        )

    return predict_from_volume_path(
        ct_path=req.ct_path,
        lung_mask_path=req.lung_mask_path,
        annotations_world_xyz=req.annotations_world_xyz,
        annotation_diameters_mm=req.annotation_diameters_mm,
        seriesuid=req.seriesuid,
    )


@app.post("/predict/dicom-dir")
def predict_dicom_dir(req: DicomDirPredictRequest) -> Dict:
    dicom_dir = Path(req.dicom_dir)
    if not dicom_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"DICOM directory not found: {req.dicom_dir}",
        )

    if not dicom_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Provided DICOM path is not a directory: {req.dicom_dir}",
        )

    if req.lung_mask_path:
        lung_mask_path = Path(req.lung_mask_path)
        if not lung_mask_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Lung mask path not found: {req.lung_mask_path}",
            )

    return predict_from_dicom_dir(
        dicom_dir=req.dicom_dir,
        lung_mask_path=req.lung_mask_path,
        annotations_world_xyz=req.annotations_world_xyz,
        annotation_diameters_mm=req.annotation_diameters_mm,
        seriesuid=req.seriesuid,
    )


@app.post("/predict/upload-volume")
async def predict_upload_volume(
    file: UploadFile = File(...),
    seriesuid: Optional[str] = Form(None),
    annotations_world_xyz: Optional[str] = Form(None),
    annotation_diameters_mm: Optional[str] = Form(None),
) -> Dict:
    filename = file.filename or "uploaded_volume"
    suffix = "".join(Path(filename).suffixes).lower()
    allowed = {".nii", ".nii.gz", ".mha", ".mhd"}

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported upload type: {suffix}. Use one of {sorted(allowed)}",
        )

    content = await file.read()

    parsed_annotations = None
    parsed_diameters = None

    if annotations_world_xyz:
        import json

        parsed_annotations = json.loads(annotations_world_xyz)

    if annotation_diameters_mm:
        import json

        parsed_diameters = json.loads(annotation_diameters_mm)

    return predict_from_uploaded_volume_bytes(
        file_bytes=content,
        filename=filename,
        annotations_world_xyz=parsed_annotations,
        annotation_diameters_mm=parsed_diameters,
        seriesuid=seriesuid,
    )


@app.post("/predict/upload-volume-zip")
async def predict_upload_volume_zip(
    file: UploadFile = File(...),
    seriesuid: Optional[str] = Form(None),
    annotations_world_xyz: Optional[str] = Form(None),
    annotation_diameters_mm: Optional[str] = Form(None),
) -> Dict:
    filename = file.filename or "uploaded_volume.zip"

    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Volume upload must be a .zip file")

    content = await file.read()

    parsed_annotations = None
    parsed_diameters = None

    if annotations_world_xyz:
        import json

        parsed_annotations = json.loads(annotations_world_xyz)

    if annotation_diameters_mm:
        import json

        parsed_diameters = json.loads(annotation_diameters_mm)

    return predict_from_uploaded_volume_zip_bytes(
        zip_bytes=content,
        filename=filename,
        annotations_world_xyz=parsed_annotations,
        annotation_diameters_mm=parsed_diameters,
        seriesuid=seriesuid,
    )


@app.post("/predict/upload-dicom-zip")
async def predict_upload_dicom_zip(
    file: UploadFile = File(...),
    seriesuid: Optional[str] = Form(None),
    annotations_world_xyz: Optional[str] = Form(None),
    annotation_diameters_mm: Optional[str] = Form(None),
) -> Dict:
    filename = file.filename or "uploaded_dicom.zip"

    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="DICOM upload must be a .zip file")

    content = await file.read()

    parsed_annotations = None
    parsed_diameters = None

    if annotations_world_xyz:
        import json

        parsed_annotations = json.loads(annotations_world_xyz)

    if annotation_diameters_mm:
        import json

        parsed_diameters = json.loads(annotation_diameters_mm)

    return predict_from_uploaded_dicom_zip_bytes(
        zip_bytes=content,
        filename=filename,
        annotations_world_xyz=parsed_annotations,
        annotation_diameters_mm=parsed_diameters,
        seriesuid=seriesuid,
    )
