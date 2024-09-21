from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, lit, window
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import round as spark_round
from pyspark.sql.types import TimestampType
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", help = "window size (seconds)", type = int, default = 10)
parser.add_argument("-s", "--slide", help = "window slide duration (seconds)", type = int, default = 1)
parser.add_argument("-d", "--decay", help = "decay constant", type = float, default = 0.1)
args = parser.parse_args()

if __name__ == "__main__":
    # variables
    window_size = args.window # window size
    slide = args.slide # window slide duration
    decay = args.slide # decay

    # start spark
    spark = SparkSession.builder.appName("stream assignment").getOrCreate()

    stream_df = (spark.readStream
                .format("socket")
                .option("host", "localhost")
                .option("port", 9999)
                .load())

    split_df = (stream_df.withColumn('tmp', split(stream_df.value, ','))
            .withColumn('event_time', col('tmp').getItem(0).cast(TimestampType()))
            .withColumn('event', col('tmp').getItem(1))
            .drop(col("tmp")).drop(col("value"))
            .withColumn("count", lit(1)))

    split_df.printSchema()

    windowed_df = (split_df.withColumn("decay_factor", lit(1 - decay)) 
        .withColumn("weighted_sum", (col("count") * col("decay_factor")) + 1) 
        .groupBy(window(col("event_time"), f"{window_size} seconds", slideDuration = f"{slide} seconds")
                .alias("window"), col("event")) 
        .agg(spark_sum(col("weighted_sum")).alias("total_weighted_sum"), col("event")) 
        .withColumn("start", col("window").getField("start"))
        .withColumn("end", col("window").getField("end"))
        .withColumn("total_weighted_sum", spark_round(col("total_weighted_sum"), 4))
        .select("event", "start", "end", "total_weighted_sum")\
        .orderBy(col("total_weighted_sum").desc(), col("window.start")))
        
    # print schema again
    windowed_df.printSchema()
    
    # output to console
    window_query = (windowed_df.writeStream
                    .outputMode("complete") 
                    .format("console") 
                    .start())

    # Wait for the queries to terminate
    window_query.awaitTermination()