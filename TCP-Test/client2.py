import numpy as np
import socket, pickle, os, time, subprocess, select
import errno
import sys

my_username = "ROBO 2"
HEADER_LENGTH = 10

IP = "127.0.0.1"
PORT = 1234

'''
    mtx = np.random.randint(low=0, high=10, size=(4, 7), dtype=np.int16)
    mtx_dump = pickle.dumps(mtx)
    message = mtx_dump
'''
# Create a socket
# socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to a given ip and port
client_socket.connect((IP, PORT))

# Set connection to non-blocking state, so .recv() call won;t block, just return some exception we'll handle
client_socket.setblocking(False)

# Prepare username and header and send them
# We need to encode username to bytes, then count number of bytes and prepare header of fixed size, that we encode to bytes as well
username = my_username.encode('utf-8')
client_socket.send(username)

while True:
    try:
        mtx = np.random.randint(low=0, high=10, size=(4, 7), dtype=np.int16)
        mtx_d = pickle.dumps(mtx)
        client_socket.send(mtx_d)
        time.sleep(2) 
        while True:
            mtx_r = client_socket.recv(1024)
            try:
                mtx_a_r = pickle.loads(mtx_r)
                print("Robo 1:")
                print(mtx_a_r)
            except:
                print("Fail on pickle loads")


    except IOError as e:
            # This is normal on non blocking connections - when there are no incoming data error is going to be raised
            # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
            # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
            # If we got different error code - something happened
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()

            # We just did not receive anything
        continue

    except Exception as e:
            # Any other exception - something happened, exit
        print('Reading error: '.format(str(e)))
        sys.exit()
    except KeyboardInterrupt:
        client_socket.close()




            
