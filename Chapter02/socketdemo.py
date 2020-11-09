import socket
udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_client.bind(('127, 0, 0, 1', 9999))
udp_server.bind(('', 9999))


