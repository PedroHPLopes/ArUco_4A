import socket, pickle
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1214))

while True:
    msg = s.recv(1024)

    data = pickle.loads(msg)
    print(data)
