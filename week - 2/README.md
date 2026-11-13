# VisionPort: High-Throughput Intelligent Port Vision Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA_Triton-24.08-76B900.svg)](https://developer.nvidia.com/triton-inference-server)
[![NVIDIA TensorRT](https://img.shields.io/badge/NVIDIA_TensorRT-10.x-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

VisionPort is an end-to-end, hardware-accelerated computer vision and video analytics platform engineered for automated container terminals, quayside cranes, automated gate systems (AGS), and yard logistics.

For the exhaustive technical specification, hardware mapping, and mathematical latency formulations, see [ARCHITECTURE.md](file:///c:/Users/raghu/Downloads/DLAC_Skill2/ARCHITECTURE.md).

---

## Architecture Summary

```
Edge 4K Cameras (RTSP/WebRTC)
           │
           ▼
FastAPI Gateway & Orchestrator (:8080)
           │  (Zero-Copy CUDA IPC / Asynchronous HTTP/2 gRPC)
           ▼
NVIDIA Triton Inference Server (:8001)
  ├── Dynamic Batching Scheduler (Window: 5ms, Max Batch: 32)
  ├── Concurrent Model Instance Groups (2x GPU streams)
  └── Ensemble Pipelines (Detector -> Crop -> OCR)
           │  (Direct GPU Memory Handoff)
           ▼
NVIDIA TensorRT Runtime Engine (.plan)
  ├── Layer & Tensor Fusions (Conv + Bias + Activation)
  ├── Auto-Tuned GPU Kernel Tactics
  └── Precision Quantization (FP16 / INT8)
           │
           ▼
NVIDIA GPU & Tensor Cores (HBM3 / GDDR6X, PCIe Gen5 DMA)
```

---

## Project Structure

```
DLAC_Skill2/
├── ARCHITECTURE.md                  # Comprehensive architectural specification
├── docker-compose.yml               # Triton + FastAPI + Prometheus orchestration
├── Dockerfile.gateway               # FastAPI production container build
├── requirements.txt                 # Python dependencies
├── model_repository/                # Triton model repository
│   ├── container_detector/
│   │   └── config.pbtxt             # TensorRT backend, dynamic batching & GPU instances
│   ├── container_ocr/
│   │   └── config.pbtxt             # INT8 OCR model configuration
│   └── visionport_pipeline/
│       └── config.pbtxt             # Triton Ensemble scheduler
├── services/
│   └── api_gateway/
│       ├── main.py                  # FastAPI application & WebSocket endpoints
│       ├── schemas.py               # Pydantic V2 schemas & ISO 6346 models
│       ├── triton_client.py         # Asynchronous gRPC Triton client
│       └── cuda_shm_client.py       # Zero-copy CUDA IPC shared memory client
├── scripts/
│   ├── build_tensorrt_engine.py     # ONNX to TensorRT engine compiler
│   └── benchmark_triton.py          # Latency percentiles (p50-p99) & FPS benchmark
└── tests/
    └── test_pipeline.py             # Validation & unit test suite
```

---

## Quickstart Guide

### 1. Install Local Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Unit Tests & Schema Verification
```powershell
pytest tests/test_pipeline.py -v
```

### 3. Run Inference Benchmark Simulation
```powershell
python scripts/benchmark_triton.py --concurrency 8 --frames 25 --batch-size 1
```

### 4. Deploy Full Production Stack (GPU Host)
```powershell
docker compose up -d
```
* **FastAPI Gateway**: `http://localhost:8080/docs`
* **Triton HTTP Server**: `http://localhost:8000/v2/health/ready`
* **Prometheus Metrics**: `http://localhost:9090`

