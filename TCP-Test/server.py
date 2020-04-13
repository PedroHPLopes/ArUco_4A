import numpy as np
import socket, pickle, os, time, subprocess, select
import sys

HEADER_LENGTH = 14

IP = "127.0.0.1"
PORT = 1234

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((IP, PORT))
server_socket.listen()
sockets_list = [server_socket]
clients = {}

print(f'Listening for connections on {IP}:{PORT}...')

def receive_message(client_socket):
    try:
        message = client_socket.recv(1024)
        if not len(message):
            return False
        try:     
            mtx = pickle.loads(message)
        except: 
            print("Fail on pickle loads")
            return False
        return mtx
    except: 
        return False



while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    if read_sockets is not None:
        for notified_socket in read_sockets:

                # If notified socket is a server socket - new connection, accept it
            if notified_socket == server_socket:

                    # Accept new connection
                    # That gives us new socket - client socket, connected to this given client only, it's unique for that client
                    # The other returned object is ip/port set
                client_socket, client_address = server_socket.accept()

                # Client should send his name right away, receive it
                user = client_socket.recv(1024)
                username = user.decode('utf-8')

                    # If False - client disconnected before he sent his name
                if user is False:
                    continue

                    # Add accepted socket to select.select() list
                sockets_list.append(client_socket)

                    # Also save username and username header
                clients[client_socket] = username

                print('Accepted new connection from')
                print(client_address)
                print('Username')
                print(username)
            else:
                mtx = receive_message(notified_socket)
                if mtx is False:
                    print('Closed connection from:')
                    print(clients[notified_socket])
                    sockets_list.remove(notified_socket)
                    del clients[notified_socket]
                    notified_socket.close()
                    continue
                else:
                    username = clients[notified_socket]
                    print(f"Received from: {username}:")
                    print(mtx)
                for client_socket in clients:
                    if client_socket != notified_socket:
                        try:
                            mtx_d = pickle.dumps(mtx)
                            client_socket.send(mtx_d)
                        except: 
                            print("Fail on sending message")
                            break           
        for notified_socket in exception_sockets:
            sockets_list.remove(notified_socket)
            del clients[notified_socket]
            notified_socket.close()
                


            