"""
Server for receiving streaming data from simulations. The rank 0 process first reconstructs the full domain from subdomain results, and then streams it via TCP (if desktop app) to the GUI (server).
"""

import struct
import pickle
from typing import Optional
import pyvista as pv
from PySide6 import QtNetwork, QtCore
from .base import TransportProtocol, DEFAULT_STREAMING_PORT

class StreamingServer(QtCore.QObject):
    """
    Server for receiving streaming data. Currently supports TCP.
    """
    data_received = QtCore.Signal(pv.DataSet)
    image_received = QtCore.Signal(bytes)
    
    def __init__(
        self,
        port: int = DEFAULT_STREAMING_PORT,
        protocol: TransportProtocol = TransportProtocol.TCP,
        parent: Optional[QtCore.QObject] = None
    ) -> None:
        super().__init__(parent)
        self._protocol = protocol
        self._port = port
        
        # See each setup method for fields declared
        if protocol == TransportProtocol.TCP:
            self._setup_tcp_server()
        else:
            raise ValueError(f"Server protocol {protocol} not yet implemented")
    
    # --- TCP Server Methods ---
    def _setup_tcp_server(self) -> None:
        """
        Initialize TCP server and connection state variables. Creates a QTcpServer instance and initializes all state variables needed for handling TCP connections and processing length-prefixed messages.

        Initializes the following instance variables:
        - _server: QTcpServer instance for accepting connections
        - _active_connection: Tracks the single active client connection
            (only one connection allowed at a time; rank 0 MPI process)
        - _buffer: QByteArray for accumulating incoming TCP stream data until
            complete messages are received
        - _expected_length: Expected message length extracted from the 8-byte
            length prefix, used to determine when a complete message has arrived
        """
        self._server = QtNetwork.QTcpServer(self)
        self._active_connection: Optional[QtNetwork.QTcpSocket] = None
        self._buffer = QtCore.QByteArray()
        self._expected_length: Optional[int] = None
        
        # Set up event listener for new connections
        self._server.newConnection.connect(self._on_new_connection)

        # Start server to listen for connetions on the loopback interface
        self._server.listen(QtNetwork.QHostAddress(QtNetwork.QHostAddress.SpecialAddress.LocalHost), self._port)
    
    def is_listening(self) -> bool:
        """Return whether the underlying TCP server is currently listening."""
        return self._server.isListening()

    def read_port(self) -> int:
        """Return the TCP port the server accepts connections on, which a caller that asked for port 0 needs in order to reach it."""
        return int(self._server.serverPort())

    def _on_new_connection(self) -> None:
        """
        Handle new client connection from QTcpServer. Called automatically when newConnection signal is emitted. Accepts the pending connection, closes any existing active connection, and sets up signal handlers for data reception and disconnection.
        """
        # Takes in the next queued connection (accepted/signaled by Qt but not yet processed). Should be empty most of the time
        client_socket = self._server.nextPendingConnection()
        if self._active_connection:
            self._active_connection.close()
        self._active_connection = client_socket

        # Set up listeners for signals from QtNetwork.QTcpSocket (client signals)
        client_socket.readyRead.connect(self._on_ready_read)
        client_socket.disconnected.connect(self._on_disconnected)
    
    def _on_ready_read(self) -> None:
        if not self._active_connection:
            return
        
        self._buffer.append(self._active_connection.readAll())
        
        # Message-parsing portion is protocol-agnostic
        while self._buffer.size() > 0:
            # Read full message
            if self._expected_length is None:
                if self._buffer.size() < 8:
                    return
                length_bytes = bytes(self._buffer.left(8))
                self._expected_length = struct.unpack('!Q', length_bytes)[0]
                self._buffer.remove(0, 8)
            
            if self._expected_length is None or self._buffer.size() < self._expected_length:
                return
            
            message_bytes = bytes(self._buffer.left(self._expected_length))
            self._buffer.remove(0, self._expected_length)

            try:
                # Deserialize the received message
                dataset = pickle.loads(message_bytes)
                self.data_received.emit(dataset)
            except Exception as e:
                print(f"Error unpickling data: {e}")
                self._buffer.clear()
                self._expected_length = None
                return
            
            self._expected_length = None
    
    def _on_disconnected(self) -> None:
        """
        Handle client disconnection. Called automatically when disconnected signal is emitted. Clears the active connection reference and resets message buffering state.
        """
        self._active_connection = None
        self._buffer.clear()
        self._expected_length = None
