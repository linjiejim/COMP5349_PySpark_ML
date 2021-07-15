# Import all necessary libraries and setup the environment for matplotlib
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, IndexToString, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, Normalizer, Word2Vec
from pyspark.ml.recommendation import ALS


def extract_data(df_origin):
    return df_origin\
        .select('user_id', 'replyto_id', 'retweet_id')\
        .groupby("user_id")\
        .agg(
            F.collect_list("replyto_id").alias("rp_list"),
            F.collect_list("retweet_id").alias("rt_list"))\
        .withColumn('document_representation', F.concat_ws(' ', F.col('rp_list'), F.col('rt_list')))\
        .filter('document_representation != ""')\
        .select('user_id', 'document_representation')\
        .cache()


def tf_idf(df, user_id, num_show=5):
    # Define Transformers
    tokenizer = Tokenizer(
        inputCol="document_representation", outputCol="tokens")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(),
                          outputCol="raw_features")
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")
    normalizer = Normalizer(inputCol=idf.getOutputCol(), outputCol="norm")

    # Assemble Pipeline
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, normalizer])

    # Fit the model
    idfModel = pipeline.fit(df)

    # Transform the data
    data = idfModel.transform(df)
    data = data.select('user_id', 'features', 'norm')

    # Get user's features
    user = df.filter(df['user_id'] == user_id)
    user = idfModel.transform(user)
    user = user.select('user_id', 'features', 'norm')

    # Compute the cosine similarity
    func_dot = F.udf(lambda x, y: float(x.dot(y)), DoubleType())
    results = data.alias("data")\
        .join(user.alias("user"), F.col("data.user_id") < F.col("user.user_id"))\
        .select(
        F.col("data.user_id").alias("user_id"),
        func_dot("data.norm", "user.norm").alias("cos_similarity"))\
        .sort('cos_similarity', ascending=False)

    results.show(num_show)
    return


def word2vec(df, user_id, num_show=5):
    # Define Transformers
    tokenizer = Tokenizer(
        inputCol="document_representation", outputCol="tokens")
    word2Vec = Word2Vec(vectorSize=3, minCount=0,
                        inputCol=tokenizer.getOutputCol(), outputCol="features")
    normalizer = Normalizer(inputCol=word2Vec.getOutputCol(), outputCol="norm")

    # Assemble Pipeline
    pipeline = Pipeline(stages=[tokenizer, word2Vec, normalizer])

    # Fit the model
    w2vModel = pipeline.fit(df)

    # Transform the data
    data = w2vModel.transform(df)
    data = data.select('user_id', 'features', 'norm')

    # Get user's features
    user = df.filter(df['user_id'] == user_id)
    user = w2vModel.transform(user)
    user = user.select('user_id', 'features', 'norm')

    # Compute the cosine similarity
    func_dot = F.udf(lambda x, y: float(x.dot(y)), DoubleType())
    results = data.alias("data")\
        .join(user.alias("user"), F.col("data.user_id") < F.col("user.user_id"))\
        .select(
            F.col("data.user_id").alias("user_id"),
            func_dot("data.norm", "user.norm").alias("cos_similarity"))\
        .sort('cos_similarity', ascending=False)

    results.show(num_show)
    return


def extract_data_als(df_origin):
    df = df_origin\
        .filter("user_mentions is not null")\
        .select("user_id", "user_mentions.id")\

    df = df\
        .select('user_id', F.explode(df.id).alias('user_mentioned'))

    df = df.groupBy(df.columns)\
        .count()\
        .where(F.col('count') >= 1)\
        .cache()

    return df


def als(df):
    # Transform long -> int
    userIndexer = StringIndexer(inputCol="user_id", outputCol="_user_id")
    userIndexerModel = userIndexer.fit(df)
    data = userIndexerModel.transform(df)

    itemIndexer = StringIndexer(
        inputCol="user_mentioned", outputCol="_user_mentioned")
    itemIndexerModel = itemIndexer.fit(data)
    data = itemIndexerModel.transform(data)

    # Define the Model
    als = ALS(maxIter=5, regParam=0.01,
              userCol="_user_id",
              itemCol="_user_mentioned",
              ratingCol="count",
              coldStartStrategy="drop",
              nonnegative=True)

    # Fit
    alsModel = als.fit(data)

    # Recommendations
    userRecs = alsModel.recommendForAllUsers(5)

    # Results
    userRecs.show(truncate=False)
    return


if __name__ == "__main__":

    # Path to the dataset
    DATA_FILE = "tweets.json"

    # Create SS
    spark = SparkSession \
        .builder \
        .appName("Python Spark Assignment 2") \
        .getOrCreate()

    # Configuration
    spark.conf.set("spark.sql.shuffle.partitions", 5)

    # Read the dataset
    df_origin = spark.read.option('multiline', 'true').json(DATA_FILE)

    '''--- Q1 ---'''
    # Construct the document representation
    df = extract_data(df_origin)

    # Specify a user
    USER_ID = 867477000658841600

    # Running TF_IDF
    tf_idf(df, USER_ID)

    # Running Word2Vec
    word2vec(df, USER_ID)

    '''--- Q2 ---'''
    # Construct the martrix
    df = extract_data_als(df_origin)

    # Running ALS
    als(df)
