from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Normalized top-left X coordinate [0.0 - 1.0]")
    y_min: float = Field(..., description="Normalized top-left Y coordinate [0.0 - 1.0]")
    x_max: float = Field(..., description="Normalized bottom-right X coordinate [0.0 - 1.0]")
    y_max: float = Field(..., description="Normalized bottom-right Y coordinate [0.0 - 1.0]")
    confidence: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="Class identifier")
    class_name: str = Field(..., description="Human-readable class name (e.g., container, truck, crane, person)")


class ISOContainerCode(BaseModel):
    raw_code: str = Field(..., json_schema_extra={"example": "MSCU1234567"}, description="11-character ISO 6346 code")
    owner_code: str = Field(..., json_schema_extra={"example": "MSC"}, description="3-letter owner code")
    category_identifier: str = Field(..., json_schema_extra={"example": "U"}, description="U (freight container), J, or Z")
    serial_number: str = Field(..., json_schema_extra={"example": "123456"}, description="6-digit serial number")
    check_digit: int = Field(..., json_schema_extra={"example": 7}, description="Calculated 1-digit check number")
    is_checksum_valid: bool = Field(..., description="Whether ISO 6346 checksum verification passes")
    confidence: float = Field(..., description="Confidence score from OCR model")


class InspectionResult(BaseModel):
    camera_id: str = Field(..., description="Identifier of the querying quayside or gate camera")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of frame capture")
    detections: List[BoundingBox] = Field(default_factory=list)
    container_ocr: Optional[ISOContainerCode] = None
    structural_damage_detected: bool = Field(default=False)
    damage_types: List[str] = Field(default_factory=list, description="e.g., ['rust', 'dent', 'seal_broken']")
    ppe_compliance: bool = Field(default=True, description="Safety compliance indicator for personnel in frame")
    inference_latency_ms: float = Field(..., description="Pure GPU inference latency in milliseconds")
    end_to_end_latency_ms: float = Field(..., description="Total pipeline latency including network & pre/post processing")


class ServerHealthResponse(BaseModel):
    status: str
    triton_ready: bool
    gpu_available: bool
    active_models: List[str]
