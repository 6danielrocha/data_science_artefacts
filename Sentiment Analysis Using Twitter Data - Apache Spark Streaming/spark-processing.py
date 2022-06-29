from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from textblob import TextBlob

from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
from pyspark.sql.types import DoubleType
import sqlite3

import subprocess

def preprocessing(lines):
    words = lines.select(explode(split(lines.value, "_xxx_")).alias("word"), lines.timestamp)
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', F.regexp_replace('word', '#', ''))
    words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', F.regexp_replace('word', ':', ''))
    return words

# text classification
def polarity_detection(text):
    return TextBlob(text).sentiment.polarity

def subjectivity_detection(text):
    return TextBlob(text).sentiment.subjectivity

def text_classification(words):
    # polarity detection
    polarity_detection_udf = udf(polarity_detection, StringType())
    words = words.withColumn("polarity", polarity_detection_udf("word"))
    # subjectivity detection
    subjectivity_detection_udf = udf(subjectivity_detection, StringType())
    words = words.withColumn("subjectivity", subjectivity_detection_udf("word"))
    return words

def create_table():
    con = sqlite3.connect(r"dashboard-app/db/sentimento.db")

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS polaridade")
    cur.execute("CREATE TABLE polaridade (polaridade_catg, count)")

    # The qmark style used with executemany():
    lang_list = [
        ("Vazio", 0),
        ("Meio Vazio", 0),
        ("Meio Cheio", 0),
        ("Cheio", 0),
        ("Neutro", 0),
    ]
    cur.executemany("INSERT INTO polaridade VALUES (?, ?)", lang_list)
    con.commit()
    con.close()
    print("create table")

def update_table(lista):
    con = sqlite3.connect(r"dashboard-app/db/sentimento.db")
    cur = con.cursor()
    sql = "UPDATE polaridade SET count = ? WHERE polaridade_catg = ?"
    print(lista)
    cur.execute(sql, lista)
    con.commit()
    con.close()
    print("update table ")


def database_conn(dataframe):

    mydic = dataframe.asDict()

    key = mydic['polaridade_catg']
    value = mydic['count']
    # print(key+" "+str(value))
    lista = [value, key]
    update_table(lista)


if __name__ == "__main__":


    #subprocess.Popen(" python dashboard-app/app.py  1", shell=True)

    # INICIAR A APLICACAO TWITTER SOCKET
    subprocess.Popen(" python pt-twitter-api-socket-conn.py  1", shell=True)

    # INICIAR A APLICACAO QUER VAI CONSUMIR O SOCKET DO TWITTER
    conf1 = SparkConf().setMaster("local[*]").setAppName("TwitterSentimentAnalysis")

    # create Spark session
    spark = SparkSession.builder.config(conf=conf1).getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # read the tweet data from socket
    lines = spark.readStream.format("socket")\
        .option("host", "localhost")\
        .option("port", 9009)\
        .option("includeTimestamp", True)\
        .load()\


    # Returns True for DataFrames that have streaming sources
    # lines.isStreaming()

    lines = lines.withWatermark("timestamp", "2 minutes")

    #  |-- value: string (nullable = true)
    lines.printSchema()

    # Preprocess the data
    words = preprocessing(lines)

    # text classification to define polarity and subjectivity
    words = text_classification(words)

    words = words.repartition(1)

    # convert string to double
    words = words.withColumn("polarity",words.polarity.cast('double'))
    words = words.withColumn("subjectivity",words.subjectivity.cast('double'))
    words.printSchema()

    # Running count of the number of updates for each device type
    words.createOrReplaceTempView("temp_view")

    # use .outputMode("append")
    df = spark.sql(
        """SELECT polarity, subjectivity, timestamp,
                CASE
                   WHEN polarity < -0.5 THEN 'Vazio'
                   WHEN polarity >=-0.5 AND polarity < 0 THEN 'Meio Vazio'
                   WHEN polarity == 0 THEN 'Neutro'
                   WHEN polarity > 0 AND polarity <= 0.5 THEN 'Meio Cheio'
                   ELSE 'Cheio'
                END AS polaridade_catg,
                CASE
                    WHEN subjectivity > 0  AND subjectivity < 0.3 THEN 'Fato'
                    WHEN subjectivity >=0.3 AND subjectivity < 0.6 THEN 'Meio'
                    ELSE 'Opnião'
                END AS subjectivity_catg
           FROM temp_view""")


    # use .outputMode("complete")
    df = df.groupBy("polaridade_catg").count()
    df.printSchema()

    # Append output mode not supported when there are streaming aggregations
    # on streaming DataFrames/DataSets without watermark

    #query = df.writeStream.queryName("all_tweets")\
    #     .outputMode("complete") \
    #     .format("console") \
    #     .start()

    create_table()

    query = df.writeStream.queryName("all_tweets")\
         .foreach(database_conn)\
         .outputMode("complete")\
         .trigger(processingTime='1 seconds') \
         .start()


    query.awaitTermination()
