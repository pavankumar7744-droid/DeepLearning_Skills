import cv2
import numpy as np
import pytest

from scripts.build_tensorrt_engine import build_engine
from services.api_gateway.main import (
    calculate_traffic_density,
    coco_bbox_to_xyxy,
    preprocess_image,
    validate_iso_6346_checksum,
)
from services.api_gateway.schemas import (
    COCO_TRAFFIC_CLASSES,
    BoundingBox,
    InspectionResult,
    ISOContainerCode,
    VehicleCounts,
)


def test_traffic_bounding_box_and_coco_categories():
    # Verify traffic classes mapped from COCO dataset
    assert COCO_TRAFFIC_CLASSES[0] == "person"
    assert COCO_TRAFFIC_CLASSES[2] == "car"
    assert COCO_TRAFFIC_CLASSES[3] == "motorcycle"
    assert COCO_TRAFFIC_CLASSES[5] == "bus"
    assert COCO_TRAFFIC_CLASSES[7] == "truck"

    # Verify bounding box creation for Indian traffic objects
    box = BoundingBox(
        x_min=0.15,
        y_min=0.25,
        x_max=0.65,
        y_max=0.75,
        confidence=0.94,
        class_id=4,
        class_name="auto_rickshaw",
    )
    assert box.class_name == "auto_rickshaw"
    assert box.confidence == 0.94
    assert 0.0 <= box.x_min < box.x_max <= 1.0


def test_coco_bbox_to_xyxy_conversion():
    # COCO format: [x, y, width, height]
    coco_box = [100.0, 150.0, 200.0, 300.0]
    img_w, img_h = 1000, 1000

    # Test normalized conversion
    x1, y1, x2, y2 = coco_bbox_to_xyxy(coco_box, img_w, img_h, normalize=True)
    assert x1 == 0.1
    assert y1 == 0.15
    assert x2 == 0.3
    assert y2 == 0.45

    # Test pixel-space conversion
    px1, py1, px2, py2 = coco_bbox_to_xyxy(coco_box, img_w, img_h, normalize=False)
    assert px1 == 100.0
    assert py1 == 150.0
    assert px2 == 300.0
    assert py2 == 450.0


def test_traffic_density_metrics():
    assert calculate_traffic_density(2) == "LOW"
    assert calculate_traffic_density(6) == "MODERATE"
    assert calculate_traffic_density(12) == "HIGH"
    assert calculate_traffic_density(25) == "CONGESTED"

    counts = VehicleCounts(cars=5, motorcycles_two_wheelers=8, auto_rickshaws=3, buses=2, total_vehicles=18)
    assert counts.total_vehicles == 18



def test_iso_6346_valid_checksum():
    # Valid real-world ISO 6346 container numbers
    assert validate_iso_6346_checksum("CSQU3054383") is True
    assert validate_iso_6346_checksum("MSCU7824106") is True


def test_iso_6346_invalid_checksum():
    assert validate_iso_6346_checksum("MSCU7824109") is False
    assert validate_iso_6346_checksum("INVALID123") is False
    assert validate_iso_6346_checksum("SHORT") is False


def test_bounding_box_schema():
    box = BoundingBox(
        x_min=0.1,
        y_min=0.2,
        x_max=0.8,
        y_max=0.9,
        confidence=0.95,
        class_id=0,
        class_name="shipping_container"
    )
    assert box.confidence == 0.95
    assert box.class_name == "shipping_container"


def test_image_preprocessing():
    # Create synthetic 720p image (1280x720)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    tensor, scale, (pad_x, pad_y) = preprocess_image(encoded.tobytes(), target_size=640)

    # Must be NCHW float32 normalized [0.0, 1.0]
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_tensorrt_engine_builder_simulation():
    ok = build_engine(
        onnx_file_path="models/test.onnx",
        engine_file_path="model_repository/container_detector/1/model.plan",
        fp16=True,
        mock=True,
    )
    assert ok is True

