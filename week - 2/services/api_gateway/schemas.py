from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# COCO 80-Class subset for Indian Traffic Object Detection (VisionPort)
COCO_TRAFFIC_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

# Extended Indian Traffic Categories (including local vehicle types)
INDIAN_TRAFFIC_CATEGORIES: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "auto_rickshaw",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]


class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Normalized top-left X coordinate [0.0 - 1.0]")
    y_min: float = Field(..., description="Normalized top-left Y coordinate [0.0 - 1.0]")
    x_max: float = Field(..., description="Normalized bottom-right X coordinate [0.0 - 1.0]")
    y_max: float = Field(..., description="Normalized bottom-right Y coordinate [0.0 - 1.0]")
    confidence: float = Field(..., description="Detection confidence score")
    class_id: int = Field(..., description="COCO or custom class identifier")
    class_name: str = Field(..., description="Class name (e.g., car, motorcycle, bus, truck, person, auto_rickshaw)")


class VehicleCounts(BaseModel):
    cars: int = Field(default=0, description="Count of passenger cars")
    motorcycles_two_wheelers: int = Field(default=0, description="Count of motorcycles and scooters")
    buses: int = Field(default=0, description="Count of public transit and private buses")
    trucks: int = Field(default=0, description="Count of freight and commercial trucks")
    auto_rickshaws: int = Field(default=0, description="Count of three-wheeler auto rickshaws")
    pedestrians: int = Field(default=0, description="Count of pedestrians / persons")
    bicycles: int = Field(default=0, description="Count of bicycles")
    total_vehicles: int = Field(default=0, description="Total vehicles detected")


class ISOContainerCode(BaseModel):
    raw_code: str = Field(default="MSCU1234567", description="11-character ISO 6346 code")
    owner_code: str = Field(default="MSC", description="3-letter owner code")
    category_identifier: str = Field(default="U", description="U (freight container), J, or Z")
    serial_number: str = Field(default="123456", description="6-digit serial number")
    check_digit: int = Field(default=7, description="Calculated 1-digit check number")
    is_checksum_valid: bool = Field(default=True, description="Whether ISO 6346 checksum verification passes")
    confidence: float = Field(default=0.98, description="Confidence score from OCR model")


class InspectionResult(BaseModel):
    camera_id: str = Field(..., description="Identifier of the querying traffic camera (e.g. INTERSECTION_MG_ROAD_01)")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of frame capture")
    detections: List[BoundingBox] = Field(default_factory=list)
    vehicle_counts: VehicleCounts = Field(default_factory=VehicleCounts)
    traffic_density: str = Field(default="MODERATE", description="Traffic congestion index: LOW, MODERATE, HIGH, CONGESTED")
    container_ocr: Optional[ISOContainerCode] = None
    structural_damage_detected: bool = Field(default=False)
    damage_types: List[str] = Field(default_factory=list)
    ppe_compliance: bool = Field(default=True)
    inference_latency_ms: float = Field(..., description="Pure GPU inference latency in milliseconds")
    end_to_end_latency_ms: float = Field(..., description="Total pipeline latency including network & pre/post processing")


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: str


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    bbox: List[float] = Field(..., description="COCO bounding box format [x, y, width, height]")
    area: float
    iscrowd: int = 0


class ServerHealthResponse(BaseModel):
    status: str
    triton_ready: bool
    gpu_available: bool
    active_models: List[str]

