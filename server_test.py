import numpy as np
import socket, pickle, os, time, subprocess

"""
myIP = subprocess.check_output('hostname -I', shell=True).decode('utf-8')
myIP = myIP[:-2]
"""

print("[INFO] Server IP", myIP)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET=IPV4, SOCK_STREAM=TCP
s.bind((myIP, 1212))
s.listen(5) #accepting 5 connections max


while True:
	mtx = np.random.randint(low=0, high=1000, size=(4, 7), dtype=np.int16)
	print("\n", mtx)

	mtx_dump = pickle.dumps(mtx)

	clientsocket, address = s.accept()
	print("Connection from {} has been established!".format(address))
	clientsocket.send(mtx_dump)

	time.sleep(5)
	
