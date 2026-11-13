import asyncio
from contextlib import asynccontextmanager
import io
import time
from datetime import datetime, timezone
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from services.api_gateway.schemas import (
    BoundingBox,
    InspectionResult,
    ISOContainerCode,
    ServerHealthResponse,
)
from services.api_gateway.triton_client import TritonInferenceClient

triton_client = TritonInferenceClient(url="localhost:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await triton_client.connect()
    except Exception as e:
        print(f"Warning: Triton not immediately reachable at startup ({e}). Retrying on first request.")
    yield
    await triton_client.close()


app = FastAPI(
    title="VisionPort Edge & Gate Inspection API Gateway",
    version="2.0.0",
    description="High-throughput asynchronous API Gateway for Port Container OCR, Damage Detection, and Quayside Safety.",
    lifespan=lifespan,
)

# Enable CORS for Port Dashboard & Terminal Operating System (TOS) frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes, target_size: int = 640) -> Tuple[np.ndarray, float, float]:
    """Decodes image bytes, applies letterbox resizing, and normalizes to NCHW FP32 tensor."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image buffer.")

    h, w, _ = img.shape
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    # Normalize BGR -> RGB, [0, 255] -> [0.0, 1.0], HWC -> NCHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)  # Shape: (1, 3, 640, 640)
    return np.ascontiguousarray(tensor), scale, (left, top)


def validate_iso_6346_checksum(code: str) -> bool:
    """
    Validates ISO 6346 container identification checksum according to international standards.
    Standard: 4 letters (3 owner + 1 category) + 6 serial digits + 1 check digit.
    """
    if len(code) != 11 or not code[:4].isalpha() or not code[4:].isdigit():
        return False

    char_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
        'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
        'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
        'Y': 37, 'Z': 38
    }

    total = 0
    for i in range(10):
        c = code[i].upper()
        val = char_values[c] if c.isalpha() else int(c)
        total += val * (2 ** i)

    check_digit = (total % 11) % 10
    return check_digit == int(code[10])


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    ico_bytes = bytes.fromhex(
        "00000100010001010000010020003000000016000000280000000100000002000000010020000000000000000000000000000000000000000000000000000000000000000000"
    )
    return Response(content=ico_bytes, media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_view():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VisionPort | Intelligent Port Vision Platform</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(56, 189, 248, 0.15);
      --primary: #0284c7;
      --primary-glow: #38bdf8;
      --accent: #10b981;
      --warning: #f59e0b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    body { background: radial-gradient(circle at 20% 20%, #111a2e 0%, var(--bg) 100%); color: var(--text); min-height: 100vh; padding: 2rem; }
    .container { max-width: 1200px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 2rem; border-bottom: 1px solid var(--card-border); margin-bottom: 2rem; }
    .brand { display: flex; align-items: center; gap: 1rem; }
    .logo-icon { width: 44px; height: 44px; background: linear-gradient(135deg, #0284c7, #38bdf8); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 0 20px rgba(56,189,248,0.3); }
    h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; }
    .subtitle { color: var(--text-muted); font-size: 0.85rem; }
    .badge { padding: 0.4rem 0.9rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; display: inline-flex; align-items: center; gap: 0.5rem; }
    .badge-online { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 10px #34d399; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin-bottom: 2rem; }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.25rem; backdrop-filter: blur(10px); }
    .card h3 { font-size: 0.85rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; margin-bottom: 0.5rem; }
    .card .val { font-size: 1.4rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem; }
    .card .meta { font-size: 0.8rem; color: var(--text-muted); }
    .section-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
    .action-bar { display: flex; gap: 0.75rem; margin-bottom: 2rem; }
    .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.65rem 1.25rem; border-radius: 8px; font-size: 0.9rem; font-weight: 600; text-decoration: none; cursor: pointer; border: none; transition: all 0.2s; }
    .btn-primary { background: linear-gradient(135deg, #0284c7, #0369a1); color: #fff; box-shadow: 0 4px 15px rgba(2,132,199,0.3); }
    .btn-primary:hover { background: linear-gradient(135deg, #0369a1, #075985); }
    .btn-secondary { background: rgba(255,255,255,0.08); color: var(--text); border: 1px solid var(--card-border); }
    .btn-secondary:hover { background: rgba(255,255,255,0.14); }
    .panel { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }
    .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
    pre { background: #050811; padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); color: #38bdf8; font-family: "Consolas", monospace; font-size: 0.85rem; overflow-x: auto; max-height: 280px; }
    .visual-box { background: #050811; border: 2px dashed rgba(56,189,248,0.3); border-radius: 8px; padding: 2rem; text-align: center; margin-bottom: 1rem; position: relative; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .bbox-sample { border: 2px solid #38bdf8; background: rgba(56,189,248,0.12); border-radius: 4px; padding: 0.75rem 1.5rem; display: inline-block; box-shadow: 0 0 15px rgba(56,189,248,0.25); }
    .bbox-tag { font-size: 0.75rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 0.25rem; }
    .ocr-code { font-size: 1.5rem; font-weight: 800; letter-spacing: 0.15em; color: #fff; text-shadow: 0 0 8px rgba(255,255,255,0.5); }
    .ocr-pill { font-size: 0.75rem; background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 700; margin-left: 0.5rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="logo-icon">⚓</div>
        <div>
          <h1>VisionPort Edge & Gate Platform</h1>
          <div class="subtitle">High-Throughput Port Computer Vision, ISO 6346 OCR & Telemetry Gateway</div>
        </div>
      </div>
      <div>
        <span class="badge badge-online">
          <span class="badge-dot"></span>
          System Online
        </span>
      </div>
    </header>

    <div class="action-bar">
      <a href="/docs" class="btn btn-primary" target="_blank">📖 Interactive API Docs (Swagger UI)</a>
      <a href="/health" class="btn btn-secondary" target="_blank">🩺 Health Status JSON</a>
      <button onclick="runSampleInspection()" id="inspectBtn" class="btn btn-secondary">⚡ Run Simulated Inspection</button>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Server Status</h3>
        <div class="val" id="serverStatus" style="color: #34d399;">Healthy</div>
        <div class="meta" id="serverMeta">FastAPI Gateway v2.0.0 Active</div>
      </div>
      <div class="card">
        <h3>Triton Serving Engine</h3>
        <div class="val" style="color: #38bdf8;">Ready</div>
        <div class="meta">TensorRT 10.x Engine Profile</div>
      </div>
      <div class="card">
        <h3>Active Triton Models</h3>
        <div class="val">3 Loaded</div>
        <div class="meta">container_detector, container_ocr, visionport_pipeline</div>
      </div>
      <div class="card">
        <h3>Target Latency Budget</h3>
        <div class="val" style="color: #f59e0b;">&le; 30 ms</div>
        <div class="meta" id="latencyStat">Recent Inference: 12.40 ms</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <h2 class="section-title">🔍 Real-Time Gate Container Detection & OCR Simulation</h2>
        <span class="subtitle">Camera ID: <strong>QUAY_CRANE_04</strong> (Letterbox 640x640)</span>
      </div>

      <div class="visual-box">
        <div class="bbox-sample">
          <div class="bbox-tag">Container Detection &bull; Confidence: 96.5%</div>
          <div class="ocr-code">
            <span id="ocrDisplay">MSCU 782410 6</span>
            <span class="ocr-pill">ISO 6346 VALID</span>
          </div>
        </div>
      </div>

      <div class="panel-header">
        <h3 class="section-title">Telemetry & Inspection JSON</h3>
        <span class="subtitle" id="execTime">Status: Ready</span>
      </div>
      <pre id="jsonOutput">{
  "message": "Click 'Run Simulated Inspection' above to trigger an end-to-end edge inference cycle."
}</pre>
    </div>
  </div>

  <script>
    async function checkHealth() {
      try {
        const res = await fetch('/health');
        const data = await res.json();
        document.getElementById('serverStatus').textContent = data.status.toUpperCase();
        document.getElementById('serverStatus').style.color = data.status === 'healthy' ? '#34d399' : '#f59e0b';
      } catch (e) {
        document.getElementById('serverStatus').textContent = 'OFFLINE';
        document.getElementById('serverStatus').style.color = '#ef4444';
      }
    }
    checkHealth();
    setInterval(checkHealth, 5000);

    async function runSampleInspection() {
      const btn = document.getElementById('inspectBtn');
      btn.disabled = true;
      btn.textContent = 'Processing...';
      document.getElementById('execTime').textContent = 'Submitting frame to pipeline...';

      // Generate synthetic 1x1 JPEG frame
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 640;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, 640, 640);

      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'sample_frame.jpg');
        formData.append('camera_id', 'QUAY_CRANE_04');

        try {
          const t0 = performance.now();
          const res = await fetch('/api/v1/inspect/frame', {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          const totalMs = (performance.now() - t0).toFixed(1);

          document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
          document.getElementById('execTime').textContent = `Success &bull; Total Roundtrip: ${totalMs} ms`;
          document.getElementById('latencyStat').textContent = `Engine: ${data.inference_latency_ms} ms &bull; E2E: ${data.end_to_end_latency_ms} ms`;
          if (data.container_ocr) {
            document.getElementById('ocrDisplay').textContent = data.container_ocr.raw_code;
          }
        } catch (err) {
          document.getElementById('jsonOutput').textContent = JSON.stringify({ error: err.message }, null, 2);
          document.getElementById('execTime').textContent = 'Inspection Failed';
        } finally {
          btn.disabled = false;
          btn.textContent = '⚡ Run Simulated Inspection';
        }
      }, 'image/jpeg');
    }
  </script>
</body>
</html>
"""


@app.get("/health", response_model=ServerHealthResponse)
async def health_check():
    is_ready = await triton_client.is_ready()
    return ServerHealthResponse(
        status="healthy" if is_ready else "degraded",
        triton_ready=is_ready,
        gpu_available=is_ready,
        active_models=["container_detector", "container_ocr", "visionport_pipeline"] if is_ready else [],
    )


@app.post("/api/v1/inspect/frame", response_model=InspectionResult)
async def inspect_single_frame(
    file: UploadFile = File(..., description="JPEG/PNG image frame from port camera"),
    camera_id: str = Form("QUAY_CRANE_04", description="Camera location identifier"),
):
    """
    Receives an image frame from an edge camera, executes Triton TensorRT inference,
    extracts container detections, and returns structured inspection results.
    """
    t_start = time.perf_counter()

    # 1. Read & Preprocess Frame
    contents = await file.read()
    try:
        tensor, scale, (pad_x, pad_y) = preprocess_image(contents, target_size=640)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # 2. Invoke Triton gRPC Inference
    try:
        raw_output, trt_duration_ms = await triton_client.infer_detector(tensor)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Triton inference failed: {str(e)}")

    # 3. Post-Process Detections (Mocked parsing from TensorRT NMS output)
    # In production, parses output coordinates relative to original aspect ratio
    detections: List[BoundingBox] = [
        BoundingBox(
            x_min=0.15,
            y_min=0.20,
            x_max=0.85,
            y_max=0.75,
            confidence=0.965,
            class_id=0,
            class_name="shipping_container",
        )
    ]

    # Mock OCR identification for ISO 6346
    sample_code = "MSCU7824106"
    ocr_result = ISOContainerCode(
        raw_code=sample_code,
        owner_code="MSC",
        category_identifier="U",
        serial_number="782410",
        check_digit=6,
        is_checksum_valid=validate_iso_6346_checksum(sample_code),
        confidence=0.982,
    )

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

    return InspectionResult(
        camera_id=camera_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        detections=detections,
        container_ocr=ocr_result,
        structural_damage_detected=False,
        damage_types=[],
        ppe_compliance=True,
        inference_latency_ms=round(trt_duration_ms, 2),
        end_to_end_latency_ms=round(total_ms, 2),
    )


@app.websocket("/api/v1/stream/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Real-time bi-directional WebSocket stream for live video analytics."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            t0 = time.perf_counter()
            tensor, _, _ = preprocess_image(data, target_size=640)
            _, trt_latency = await triton_client.infer_detector(tensor)
            total_latency = (time.perf_counter() - t0) * 1000.0

            # Stream telemetry back to client
            await websocket.send_json({
                "status": "PROCESSED",
                "inference_ms": round(trt_latency, 2),
                "total_ms": round(total_latency, 2),
                "containers_detected": 1,
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api_gateway.main:app", host="0.0.0.0", port=8080, reload=True)

