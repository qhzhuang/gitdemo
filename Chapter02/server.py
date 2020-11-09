import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定端口:
s.bind(('', 9990))
print("Start listening........")
s.listen(10)

while True:
    # s.send("Welcome!".encode())

    data, addr = s.accept()
    print("accept allowed")
    print(addr)
    data1 = data.recv(1024)
    # print(addr)
    print(data1.decode())
    # print("data acuqired")
    response_header = "HTTP/1.1 200 OK\r\n"
    response_sev = "Server: Apache 2.0\r\n"
    blank = "\r\n"
    body = "Helloworld"
    datatosend = (response_header + response_sev + blank + body).encode()
    data.send(datatosend)
    print("send successfully")
data.close()
