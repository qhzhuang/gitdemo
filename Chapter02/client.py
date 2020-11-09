# import socket
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(('www.sina.com.cn', 80))
# s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')
#
# buffer = []
# while True:
#     data = s.recv(1024)
#     if data:
#         buffer.append(data)
#     else:
#         break
# s.close()
# bindata = b''.join(buffer)
# header, html = bindata.split(b'\r\n\r\n',1)
# print(header.decode())
# with open ('sinahome.html', 'wb') as f:
#     f.write(html)



import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("connecting")
# try:
s.connect(('127.0.0.1', 9990))
# except(ConnectionResetError):
#     while True:
#         try:
#             s.connect(('127.0.0.1', 9990))
#         except(ConnectionResetError):
#             continue
#         break

print("connect success")
for data in [b'Michael', b'Tracy', b'Sarah']:
    # 发送数据:
    s.sendto(data, ('127.0.0.1', 9990))
    # 接收数据:
    print(s.recv(1024).decode())
# s.send(b'exit')
s.close()