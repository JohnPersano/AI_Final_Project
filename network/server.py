import socket

import settings


class NetworkServer:
    port = 5000
    buffer = 1024

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected_socket = None
        self.connected_socket_address = ""

    def start_listen(self):
        self.server_socket.bind((socket.gethostname(), self.port))
        self.server_socket.listen(5)

    def refresh_connected_socket(self):
        if self.connected_socket is not None:
            self.connected_socket.close()
        (self.connected_socket, self.connected_socket_address) = self.server_socket.accept()
        if settings.DEBUG:
            print("Connected to: {}".format(self.connected_socket_address))

    def send_message(self, message):
        if self.connected_socket is not None:
            self.connected_socket.send(message.encode())
            return
        print("NetworkServer: Must call refresh_connected_socket before trying to send a message")

    def get_response(self):
        return self.connected_socket.recv(self.buffer).decode()


