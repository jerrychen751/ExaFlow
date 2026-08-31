"""
Implementation of socket transport protocol.
"""

import struct
import socket
from typing import Optional, override

from .base import StreamingTransport

class TCPSocketTransport(StreamingTransport):
    def __init__(self, address: str, port: int, timeout: float = 5.0) -> None:
        self._address = address
        self._port = port
        self._timeout = timeout
        self._socket: Optional[socket.socket] = None

    @override
    def send(self, data: bytes, metadata: Optional[dict] = None) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self._timeout)
                s.connect((self._address, self._port))
                length = len(data)
                # Send an 8-byte unsigned int. Have this as message prefix so that the server knows when data packets received are complete
                s.sendall(struct.pack('!Q', length))
                s.sendall(data)
                return True
        except OSError as e:
            print(f"TCP send error: {e}", flush=True)
            return False

    @override
    def close(self) -> None:
        if self._socket:
            self._socket.close()
