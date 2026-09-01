"""
Contains abstract protocol definitions and other constants.
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Any


DEFAULT_STREAMING_PORT = 12345

class TransportProtocol(Enum):
    """
    Lists supported transport protocols.
    """
    TCP = "tcp"
    WEBSOCKET = "websocket"
    HTTP = "http"

class DataFormat(Enum):
    """
    Lists supported data formats.
    """
    PICKLE_DATASET = "pickle_dataset" # pyvista dataset, such as pickle.dumps(pv.RectilinearGrid) (desktop)
    RENDERED_IMAGE = "rendered_image" # pre-rendered image like png/jpeg (web)

class StreamingTransport(ABC):
    """
    Abstract class for transport protocols. Meant to be used for TCPSocketTransport, an object / field in StreamingClient.
    """
    @abstractmethod
    def send(self, data: bytes, metadata: Optional[dict] = None) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

class StreamingSerializer(ABC):
    """
    Abstract class for data serialization.
    """
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        ...

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        ...
