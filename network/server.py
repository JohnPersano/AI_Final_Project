"""
CSCI 6660 Final Project

Author: John Persano
Date:   04/27/2016
"""

import socket

import settings


class NetworkServer:
    """
    This class enables communication with Benson and will handle sending and receiving data

    Example usage:
        network_server = NetworkServer()
        network_server.listen()
        while True:
            network_server.get_data()
            network_server.send_data("This is a test")
    """
    buffer = 1024

    def __init__(self):
        self.socket = None
        self.connection = None
        self.host = socket.gethostname()
        self.port = 5555

    def listen(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)

    def get_data(self):
        self.connection, address = self.socket.accept()
        if settings.DEBUG:
            print('Connected to: {}'.format(address))

        data = self.connection.recv(1024)
        return data.decode()

    def send_data(self, message):
        self.connection.sendall(message.encode())
        self.connection.close()
