"""
Data serialization implementations.
"""

import io
import pickle
import numpy as np
from typing import Any
from PIL import Image
from .base import StreamingSerializer

class PickleSerializer(StreamingSerializer):
    """Pickle serialization (for full PyVista datasets)."""
    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

class ImageSerializer(StreamingSerializer):
    """Image serialization (for pre-rendered views)."""
    def __init__(self, format: str = "PNG", quality: int = 95):
        self._format = format
        self._quality = quality
    
    def serialize(self, data: Any) -> bytes:
        if isinstance(data, np.ndarray):
            img = Image.fromarray(data)
        elif isinstance(data, Image.Image):
            img = data
        else:
            raise ValueError(f"Unsupported image type: {type(data)}")
        
        buffer = io.BytesIO()
        img.save(buffer, format=self._format, quality=self._quality)
        return buffer.getvalue()
    
    def deserialize(self, data: bytes) -> Any:
        return Image.open(io.BytesIO(data))
