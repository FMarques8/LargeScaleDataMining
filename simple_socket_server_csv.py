import socket, time, gzip, argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help = "location of input file", type = str)
parser.add_argument("--host", help = "server host", type = str, default = "localhost")
parser.add_argument("--port", help = "server port", type = int, default = 9999)
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

    # Open file and send bit decoded rows to client
    with gzip.open(args.file, 'r') as f:
        for row in f.readlines()[1:]:
            client_socket.send(f'{row.decode("utf-8")}'.encode("utf-8"))
            time.sleep(0.001) # prevent the cpu from clogging 

    # Close the connection
    client_socket.close()
    server_socket.close()