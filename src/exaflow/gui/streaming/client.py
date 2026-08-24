"""
Client for sending streaming data to GUI visualization server.
"""

import pyvista as pv
from typing import Any
from .base import (
    TransportProtocol, 
    DataFormat, 
    DEFAULT_STREAMING_ADDRESS, 
    DEFAULT_STREAMING_PORT
)
from .transports import TCPSocketTransport, WebSocketTransport
from .serializers import PickleSerializer, ImageSerializer

class StreamingClient:
    """
    Client for sending data to visualization server. Supports multiple transport protocols and data formats.
    """
    def __init__(
        self,
        transport: TransportProtocol = TransportProtocol.TCP,
        format: DataFormat = DataFormat.PICKLE_DATASET,
        address: str = DEFAULT_STREAMING_ADDRESS,
        port: int = DEFAULT_STREAMING_PORT,
        **kwargs
    ):
        self._transport_type = transport
        self._format_type = format
        
        # Create transport
        if transport == TransportProtocol.TCP:
            self._transport = TCPSocketTransport(address, port, **kwargs)
        elif transport == TransportProtocol.WEBSOCKET:
            raise NotImplementedError(f"Client transport {transport.value} is not implemented.")
        else:
            raise ValueError(f"Unsupported transport: {transport}")
        
        # Create serializer
        if format == DataFormat.PICKLE_DATASET:
            self._serializer = PickleSerializer()
        elif format == DataFormat.RENDERED_IMAGE:
            raise NotImplementedError(f"Client data format {format.value} is not implemented.")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def send_dataset(self, dataset: pv.DataSet) -> bool:
        data = self._serializer.serialize(dataset)
        return self._transport.send(data)
    
    def send_image(self, image: Any) -> bool:
        raise NotImplementedError("Image streaming is not implemented; use send_dataset.")
    
    def close(self):
        self._transport.close()
