import numpy as np
import socket, pickle, os, time, subprocess, select
import errno
import sys


my_username = "ROBO 1"
IP = "127.0.0.1"
PORT = 1234

def connect(IP,PORT,my_username):
    """
    Tries to connect to the server. If connected, sends the username
    Returns client_socket
    """
    connected = False
    while not connected:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((IP, PORT))
            client_socket.setblocking(False)
            username = my_username.encode('utf-8')
            client_socket.send(username)
            connected = True
        except socket.error:
            continue
    return client_socket

client_socket = connect(IP,PORT,my_username)

while True:
    try:
        mtx = np.random.randint(low=0, high=10, size=(4, 7), dtype=np.int16)
        mtx_d = pickle.dumps(mtx)
        client_socket.send(mtx_d)
        time.sleep(0.1)
        # Sans ce time.sleep() le serveur tombe. Il nous donne cet erreur:
        #UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 6: invalid start byte
        # Je n'arrive pas a trouver une autre  solution                   
        while True:
            mtx_r = client_socket.recv(1024) 
            mtx_a_r = pickle.loads(mtx_r)
            print("Received from Robo 2:")
            print(mtx_a_r)
    except IOError as e:
            # This is normal on non blocking connections - when there are no incoming data error is going to be raised
            # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
            # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
            # If we got different error code - something happened
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            print('Lost connection, trying to reconnect')
            client_socket = connect(IP,PORT,my_username)


            # We just did not receive anything
        continue
    except KeyboardInterrupt as e:
        print('Keyboard Interrupr')
        sys.exit()
        
    except Exception as e:
            # Any other exception - something happened, exit
        print('Error: {}'.format(str(e)))
        print('Lost connection, trying to reconnect')
        client_socket = connect(IP,PORT,my_username)



            
