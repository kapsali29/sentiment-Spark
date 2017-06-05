from textblob import TextBlob
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
conf = SparkConf().setAppName("Sent Analysis Textblob")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)
df = sqlCtx.read.load("hdfs:///data/kindle_store")
reviews = df[["reviewText","overall"]]
reviews.show(20)
sqlCtx.registerDataFrameAsTable(reviews, "table2")
reviews1 = sqlCtx.sql("SELECT reviewText, overall from table2")
#positive->1
#neutral->0
#negative->2
def transform(star):
        if star >=3.0:
                return 1.0
        elif star == 3.0:
                return 0.0
        else:
                return 2.0
transformer = udf(transform)
df1 = reviews1.withColumn("label", transformer(reviews['overall']))
sqlCtx.registerDataFrameAsTable(df1, "table1")
df2 = sqlCtx.sql("SELECT reviewText, label from table1 WHERE reviewText != ''")
df2.show()
def apply_blob(sentence):
    temp = TextBlob(sentence).sentiment[0]
    if temp == 0.0:
        return 0.0
    elif temp >= 0.0:
        return 1.0
    else:
        return 2.0
predictions = udf(apply_blob)
blob_df = df2.withColumn("predicted", predictions(df2['reviewText']))
blob_df.show()

true_labels = [i.label for i in blob_df.select("label").collect()]
predicted_labels = [i.predicted for i in blob_df.select("predicted").collect()]
correct = 0
wrong = 0
for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
                correct +=1
        else:
                wrong +=1
print('Correct predictions: ', correct)
print('Wrong predictions: ', wrong)
print('Accuracy: ', correct/(correct+wrong))


