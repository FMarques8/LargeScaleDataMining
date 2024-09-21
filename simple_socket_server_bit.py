import socket, time, argparse
from random import randint, uniform

parser = argparse.ArgumentParser()
parser.add_argument("--host", help = "server host", type = str, default = "localhost")
parser.add_argument("--port", help = "server port", type = int, default = 9999)
parser.add_argument("--duration", help = "duration of bit stream (minutes)", type = int, default = 5)
args = parser.parse_args()

if __name__ == "__main__":
    # Define the host and port
    HOST = args.host
    PORT = args.port

    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    server_socket.bind((HOST, PORT))

    # Listen for incoming connections
    server_socket.listen(1)
    print('Server is listening on {}:{}'.format(HOST, PORT))

    # Accept a connection from a client
    client_socket, client_address = server_socket.accept()
    print('Accepted connection from {}:{}'.format(client_address[0], client_address[1]))

    stamp = 1 # initial bit timestamp 
    # get starting time to limit while loop duration
    timeout = time.time() + 60 * args.duration # 'args.duration' duration, in minutes

    # Send data to client
    while True:
        bit = randint(0, 1)
        client_socket.send(f'{stamp},{bit}\n'.encode("utf-8"))
        stamp += 1
        # random sleep between 0 and 0.01 to simulate streaming data
        # and prevent spark driver clogging
        time.sleep(uniform(0, 0.01))
        
        if time.time() > timeout:
            break

    # Close the connection
    client_socket.close()
    server_socket.close()