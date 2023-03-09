# Modules

import os
os.environ['SPARK_HOME'] = r"C:\PySpark\spark-3.3.2-bin-hadoop3"

import findspark
findspark.init()

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local[*]").setAppName("abc").set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)

########
# some variables and load the file
support_threshold = 1000
k = [2, 3]
inputFile = 'conditions.csv'
file = sc.textFile(inputFile).sample(False, 0.01) # load file as RDD

######

def remove_duplicate_pairs(record):

    if(isinstance(record[0], tuple)):
        x1 = record[0]
        x2 = record[1]
    else:
        x1 = [record[0]]
        x2 = record[1]

    if(any(x == x2 for x in x1) == False):
        a = list(x1)
        a.append(x2)
        a.sort()
        result = tuple(a)
        return result 
    else:
        return x1


def get_frequent_itemsets(rdd, k):
    """Returns list with k most frequent itemsets"""
    
    
    full_log = (file.map(lambda line: line.split(','))
                .flatMap(lambda item: (item[2], item[-2])) # keeps PATIENT and CODE
                )
    
    patientRdd = (file.map(lambda line: line.split(','))
                .map(lambda item: (item[2], item[-2]))
                )

    unique_logs = patientRdd.distinct()
    
    freqRdd = (full_log.map(lambda item: (item, 1))
        .reduceByKey(lambda c1, c2: c1 + c2)
        .filter(lambda item: item[1] >= support_threshold)
        )

    freqRdd.top(k) # 10 most frequent items, for k = 1
    
    freqRdd = freqRdd.map(lambda item: item[0])
