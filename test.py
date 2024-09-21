import gzip

with gzip.open("stream_data/stream-data.csv.gz", 'r') as file:
    for row in file.readlines():
        print(row.decode("utf-8"))
        # client_socket.send(f'{file}\n'.encode("utf-8"))
        # time.sleep(0.01) # prevent the cpu from clogging 