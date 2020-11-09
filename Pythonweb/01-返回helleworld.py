import socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("", 8080))
server.listen(10)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
new_sever, ip_port = server.accept()