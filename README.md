sudo su
$SPARK_HOME/sbin/start-all.sh
$HADOOP_HOME/sbin/start-all.sh

wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz"
gunzip reviews_Kindle_Store_5.json.gz
hdfs dfs -put reviews_Kindle_Store_5.json /data/reviews_Kindle_Store_5.json
pyspark

>>> from pyspark.sql import SparkSession
>>> df = spark.read.json("hdfs:///data/reviews_Kindle_Store_5.json")
>>> df.write.parquet("hdfs:///data/kindle_store")

source mysparkenv/bin/activate
pip install textblob
pip install simplejson
