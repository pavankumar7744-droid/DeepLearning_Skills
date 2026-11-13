import asyncio
import logging
import time
from typing import Dict, Any, Tuple
import numpy as np

try:
    import tritonclient.grpc.aio as grpcclient  # type: ignore
    from tritonclient.utils import InferenceServerException  # type: ignore
except ImportError:
    grpcclient = None
    InferenceServerException = Exception

logger = logging.getLogger("visionport.triton")


class TritonInferenceClient:
    """High-performance asynchronous gRPC client interfacing with Triton Inference Server."""

    def __init__(self, url: str = "localhost:8001", mock_fallback: bool = True):
        self.url = url
        self._client: Any = None
        self.mock_fallback = mock_fallback
        self.is_mock_mode = False

    async def connect(self):
        """Establish asynchronous gRPC channel to Triton or fall back to simulation mode."""
        if grpcclient is None:
            self.is_mock_mode = True
            logger.warning("tritonclient library is not installed in local environment. Running in mock/simulation mode.")
            return

        try:
            self._client = grpcclient.InferenceServerClient(url=self.url, verbose=False)
            is_live = await self._client.is_server_live()
            is_ready = await self._client.is_server_ready()
            if is_live and is_ready:
                self.is_mock_mode = False
                logger.info(f"Triton Connection Established to {self.url} (Live: {is_live}, Ready: {is_ready})")
                return
            raise ConnectionError("Triton server responded but is not ready.")
        except Exception as e:
            self._client = None
            if self.mock_fallback:
                self.is_mock_mode = True
                logger.warning(f"Triton not available at {self.url} ({e}). Operating in simulated inference mode.")
            else:
                logger.error(f"Failed to connect to Triton Inference Server at {self.url}: {e}")
                raise

    async def close(self):
        if self._client:
            await self._client.close()

    async def is_ready(self) -> bool:
        if self._client:
            try:
                return await self._client.is_server_ready()
            except Exception:
                pass
        return True if (self.is_mock_mode or grpcclient is None) else False

    async def infer_detector(
        self,
        preprocessed_tensor: np.ndarray,
        model_name: str = "container_detector",
        model_version: str = "1"
    ) -> Tuple[np.ndarray, float]:
        """
        Submits preprocessed frame batch (NCHW float32) to Triton TensorRT backend.
        Returns:
            Tuple of (output_tensor, inference_duration_ms)
        """
        if self.is_mock_mode or grpcclient is None or not self._client:
            # Simulated inference for local test/verification
            await asyncio.sleep(0.012)
            mock_output = np.zeros((1, 84, 8400), dtype=np.float32)
            return mock_output, 12.4

        if not self._client:
            await self.connect()

        # Build Triton Input Tensor
        inputs = [
            grpcclient.InferInput(
                name="images",
                shape=preprocessed_tensor.shape,
                datatype="FP32"
            )
        ]
        inputs[0].set_data_from_numpy(preprocessed_tensor)

        # Build Triton Output Request
        outputs = [
            grpcclient.InferRequestedOutput(name="output0")
        ]

        t0 = time.perf_counter()
        response = await self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0

        output_data = response.as_numpy("output0")
        return output_data, duration_ms
