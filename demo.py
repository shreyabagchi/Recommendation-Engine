## Author: Shreya Bagchi

from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import rand
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType, DoubleType, FloatType
from pyspark.ml.recommendation import ALS
from flask import Flask, flash, redirect, render_template, request, url_for, session
from pyspark.sql.functions import explode
import pandas as pd
import random


spark = SparkSession.builder.master('yarn').appName('recommender').config("spark.default.parallelism",300).config("spark.sql.shuffle.partitions",300).getOrCreate()
df_mb = spark.read.json("gs://sh_books_bucket/metaBooks.json")
df_rb = spark.read.json("gs://sh_books_bucket/reviews_Books_5.json")

df_rb = df_rb.select('asin','reviewerID','overall')
df_mb = df_mb.select("asin","title")
df_mb = df_mb.fillna("Not Available",subset=['title'])

df_rb=df_rb.cache()
df_mb=df_mb.cache()

new_user_id = ''.join(random.choice('0123456789ABCDEF') for i in range(14))
new_user_ratings=[]
list_books = df_mb.select('asin').orderBy(rand()).limit(10).collect()
                                                    
for book in list_books:
    record = (new_user_id,book[0],float(random.randint(1,5)))
    new_user_ratings.append(record) 
    del record

                                                        
schema = StructType([
         StructField("reviewerID", StringType(), True),
         StructField("asin", StringType(), True),
         StructField("overall", FloatType(), True)])
                                                    
df_newuser = spark.createDataFrame(new_user_ratings, schema)


df_combined =df_rb.unionByName(df_newuser)

indexer1 = StringIndexer(inputCol='asin', outputCol='asin_Idx')
df_combined = indexer1.fit(df_combined).transform(df_combined)

indexer2 = StringIndexer(inputCol='reviewerID', outputCol='reviewerID_Idx')
df_combined = indexer2.fit(df_combined).transform(df_combined)

reviewerID_Idx = df_combined.where(df_combined.reviewerID==new_user_id).select('reviewerID_Idx').distinct()

als = ALS(userCol='reviewerID_Idx', itemCol='asin_Idx', ratingCol='overall',rank=5,
                         maxIter=10, regParam=.05, nonnegative=True,coldStartStrategy="drop", implicitPrefs=False)

new_als_model = als.fit(df_combined)



recommendations_Idx=new_als_model.recommendForUserSubset(reviewerID_Idx,10)

recommendationsDF = (recommendations_Idx
                      .select("reviewerID_Idx", explode("recommendations")
                      .alias("recommendation"))
                      .select("reviewerID_Idx", "recommendation.*")
                    )

recommendationsDF.createTempView("recommendations")
df_combined.createTempView("reviews")
df_mb.createTempView("books")


spark.sql("select distinct bks.asin,bks.title,rec.rating from recommendations rec JOIN reviews rev ON rec.asin_Idx=rev.asin_Idx JOIN books bks ON bks.asin=rev.asin ORDER BY rec.rating desc").show()


