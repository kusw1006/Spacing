import os
import sys
import socket
import threading
import binascii

import io
from io import BytesIO
from scipy.io.wavfile import write

import base64

from pydub import AudioSegment
from pydub.playback import play

def write_to_fp(fp, input_wav):
    #input_wav = input_wav.encode('ascii')
    
    #decoded = binascii.b2a_base64(input_wav[16:])
    #decoded = input_wav[:16]+decoded
    #print(decoded)
    
    #decoded = base64.b64decode(as_bytes)
    
    fp.write(input_wav)


def recv(client_sock):
    fp = BytesIO()
    while True:
        recv = client_sock.recv(1024)  
        if len(recv):
            write_to_fp(fp, recv)
            print(len(recv))
        if fp.read and len(recv)==1:
            print(333333333333333333333333333333333)
            print(fp.read())
            
            fp.seek(0)
            #print(fp.read())
            song = AudioSegment.from_file(fp, format="wav")

            play(song)



#-------------이부분만 반복--------------#
import time


HOST = '114.70.22.237'
PORT = 5052


client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client_socket.connect((HOST,PORT))

thread1 = threading.Thread(target = recv, args=(client_socket, ))
thread1.start()
