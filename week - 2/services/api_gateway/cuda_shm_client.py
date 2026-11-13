"""
VisionPort CUDA Shared Memory (Zero-Copy) Client
Enables zero-copy tensor passing directly into NVIDIA GPU memory,
bypassing OS network socket serialization for high-throughput 4K feeds.
"""

import logging
from typing import Any
import numpy as np

try:
    import tritonclient.grpc.aio as grpcclient  # type: ignore
    import tritonclient.utils.cuda_shared_memory as cudashm  # type: ignore
    from tritonclient.utils import InferenceServerException  # type: ignore
except ImportError:
    grpcclient = None
    cudashm = None
    InferenceServerException = Exception

logger = logging.getLogger("visionport.cuda_shm")


class CudaSharedMemoryTritonClient:
    """
    Manages CUDA IPC Shared Memory regions for zero-copy Triton inference.
    
    Why this is critical for VisionPort:
    At 100 camera streams @ 30 FPS, raw video data reaches ~1.86 GB/s.
    Transferring this data over standard TCP/UDS sockets induces massive CPU context switching,
    memory duplication, and page cache churn. By allocating memory in GPU VRAM and sharing
    the CUDA IPC handle directly with Triton, data copy overhead is reduced to zero.
    """

    def __init__(self, triton_url: str = "localhost:8001", shm_byte_size: int = 64 * 1024 * 1024):
        self.triton_url = triton_url
        self.shm_byte_size = shm_byte_size
        self._client: Any = None
        self._shm_input_handle = None
        self._shm_output_handle = None
        self.input_shm_region_name = "visionport_input_cuda_shm"
        self.output_shm_region_name = "visionport_output_cuda_shm"

    async def initialize_shared_memory(self):
        """Allocates GPU CUDA memory and registers the regions with Triton."""
        if cudashm is None or grpcclient is None:
            logger.warning("CUDA Shared Memory client running in mock mode (libraries not installed).")
            return

        self._client = grpcclient.InferenceServerClient(url=self.triton_url)

        # 1. Allocate CUDA shared memory buffer for input frames
        self._shm_input_handle = cudashm.create_shared_memory_region(
            self.input_shm_region_name,
            byte_size=self.shm_byte_size,
            device_id=0
        )

        # 2. Register region with Triton
        await self._client.register_cuda_shared_memory(
            name=self.input_shm_region_name,
            cuda_shm_key=cudashm.get_raw_hash(self._shm_input_handle),
            byte_size=self.shm_byte_size,
            device_id=0
        )

        logger.info(f"Registered CUDA Shared Memory: {self.input_shm_region_name} ({self.shm_byte_size / (1024*1024):.1f} MB)")

    async def cleanup(self):
        """Unregisters and frees allocated CUDA memory handles."""
        if self._client and cudashm:
            try:
                await self._client.unregister_cuda_shared_memory(name=self.input_shm_region_name)
                cudashm.destroy_shared_memory_region(self._shm_input_handle)
            except Exception as e:
                logger.error(f"Error during CUDA shared memory cleanup: {e}")
            await self._client.close()
