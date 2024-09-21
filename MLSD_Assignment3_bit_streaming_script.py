from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col
from dgim import *
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", help = "server host", type = str, default = "localhost", nargs = 1)
parser.add_argument("--port", help = "server port", type = int, default = 9999)
parser.add_argument("-w", "--window", help = "window size", type = int, default = 1000)
parser.add_argument("-r", "--recent", help = "most recent items", type = int, default = 300)
args = parser.parse_args()

if __name__ == "__main__":
    N = args.window
    k = args.recent

    # start Spark session
    spark = SparkSession.builder.appName("stream assignment").getOrCreate()

    # connect to socket server
    socket_df = (spark.readStream.format("socket") 
        .option("host", args.host) 
        .option("port", args.port)
        .load())

    split_df = (socket_df.withColumn("tmp", split(col("value"), ",")) # split values in column
            .withColumn("timestamp", col("tmp").getItem(0)) # time column 
            .withColumn("value", col("tmp").getItem(1)) # bit column
            .drop(col("tmp")))

    # print schema to dataframe
    split_df.printSchema()

    # Initialize DGIM object
    dgim = DGIM(N, k)

    def update_cumulative_sum(batch_df, batch_id):
        "Update DGIM object cumulative and real sum with each batch"
        
        print(f"Batch: {batch_id}\n")
        t_start = dgim.stream_timestamp # starting time of the batch
        rows = batch_df.toLocalIterator() # convert batch rows to generator
        
        for row in rows:
            bit = row["value"]
            dgim.update(bit)
        dgim.estimated_count += dgim.count()
        t_end = dgim.stream_timestamp # ending time of the batch
        
        new_df = spark.createDataFrame([(t_start, t_end, dgim.estimated_count, dgim.real_count)], 
                                    ['t_start', 't_end', 'estimated_sum', 'real_sum'])
        new_df.show()

    # apply update_cumulative_sum to each batch
    df_dgim = (split_df.writeStream
            .foreachBatch(update_cumulative_sum)
            .start())

    df_dgim.awaitTermination()
