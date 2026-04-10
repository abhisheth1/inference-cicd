from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.model_loader import (
    get_service_status,
    initialize_models,
    predict_from_dicom_dir,
    predict_from_volume_path,
)

app = FastAPI(title="CT Inference API", version="2.1.0")


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
        ],
        "storage_mode": "no temp files, no cache files, no visualization files",
        "model_ready": status["model_ready"],
        "endpoints": [
            "/health",
            "/predict/path",
            "/predict/dicom-dir",
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
