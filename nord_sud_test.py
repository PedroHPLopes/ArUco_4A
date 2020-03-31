from random import seed
from random import randint
import time, serial

sleep_time = 2
serial_port = "/dev/ttyS0"

def send_north():
    ser.write(1) #ou ser.write(ord('char')) qui donnne le nombre ASCII du char 
    print("Sent North ")

def send_south():
    ser.write(0)
    print("Sent South")

seed(1)
ser = serial.Serial(port=serial_port, baudrate=115200, parity=serial.PARITY_NONE, bytesize=serial.EIGHTBITS, timeout=10)

while True:
    if randint(0, 1): send_north()
    else: send_south()
    
    time.sleep(sleep_time)
