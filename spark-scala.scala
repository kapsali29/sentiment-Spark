import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.NaiveBayes

//Generate SQLContext using the following command. Here, sc means SparkContext object.
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

//load parquet file from hadoop file system
val parqfile = sqlContext.read.parquet("hdfs:///data/kindle_store")
//using sql on dataframe 
parqfile.registerTempTable("table1")
val reviews = sqlContext.sql("SELECT reviewText, overall FROM table1  WHERE reviewText != '' LIMIT 50000")
val reviewsdf = reviews.withColumn("label", when(col("overall") >= 3.0, 1.0).otherwise(0.0))

val regexTokenizer = new RegexTokenizer().setInputCol("reviewText").setOutputCol("words").setPattern("\\W")
val countTokens = udf { (words: Seq[String]) => words.length }
val regexTokenized = regexTokenizer.transform(reviewsdf)

regexTokenized.select("reviewText", "words").withColumn("tokens", countTokens(col("words"))).show(false)

val result = regexTokenized.select("label","words")
// split data to training and test 
val splits = result.randomSplit(Array(0.9, 0.1))
val (training, test) = (splits(0), splits(1))



val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features").setNumFeatures(1000)
val tf = hashingTF.transform(training)
val train_set = tf.select("label","features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
val pipeline = new Pipeline().setStages(Array(lr))

val model = pipeline.fit(train_set)

val tf_test = hashingTF.transform(test)
val test_set = tf_test.select("label","features")

val predictions = model.transform(test_set)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

//using Decision Trees Classifier
val nb  = new NaiveBayes()
val sec_model = nb.fit(train_set)

val sec_predict = sec_model.transform(test_set)

val evaluator2 = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator2.evaluate(sec_predict)
println("Test Error = " + (1.0 - accuracy))
