import cv2
import numpy as np
import pytest

from scripts.build_tensorrt_engine import build_engine
from services.api_gateway.main import preprocess_image, validate_iso_6346_checksum
from services.api_gateway.schemas import BoundingBox, ISOContainerCode, InspectionResult


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

