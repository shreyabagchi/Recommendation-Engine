# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:50:28 2019

@author: Shreya
"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType,FloatType
from time import time
import logging
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import rand
#import random
from pyspark.sql.functions import explode
#import pandas as pd
from pyspark.ml import Pipeline



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Removing the spark instances wherever required

"""A Book recommendation engine"""
class RecommendationEngine:
    
    
    """Init the recommendation engine """
    def __init__(self):
        
        logger.info("Starting up the Recommendation Engine: ")
      
        self.spark = SparkSession.builder.master('yarn').appName('recommender').config("spark.default.parallelism",300).config("spark.sql.shuffle.partitions",300).getOrCreate()
        
        #df_reviews = self.spark.read.json("C:\MBS-Rutgers programs\Big Data Algo\Project\data\sampled_reviews_part.json")
        #df_books = self.spark.read.json("C:\MBS-Rutgers programs\Big Data Algo\Project\data\metaBooks.json")

        df_reviews = self.spark.read.json("gs://sh_books_bucket/reviews_Books_5.json")
        df_books = self.spark.read.json("gs://sh_books_bucket/metaBooks.json")
        
        self.df_sampled = df_reviews.select("reviewerID","asin","overall")
        #self.df_sampled = self.to_StringIndex('reviewerID',self.df_sampled)
        #self.df_sampled = self.to_StringIndex('asin',self.df_sampled)
        
        self.data_books = df_books.select("asin","title")
        
        # Imputing the null values in the Book Metadata with the string 'Not Available'
        self.data_books = self.data_books.fillna("Not Available",subset=['title'])
        
        #print("Splitting data for evaluation...")
        #(training_data, test_data) = self.df_sampled.randomSplit([0.8, 0.2],1234)
        #self.als_model = self.train_model(training_data,5,10,"reviewerID_Idx","asin_Idx","overall")
        #self.als_model.save("C:\MBS-Rutgers programs\Big Data Algo\Project\data\model")
        #self.evaluate_model(test_data)
        
 
    
    """Train the ALS model with the current dataset"""
    def train_model(self,training_data,rank,iterations,userCol,itemCol,ratingCol):
    
        logger.info("Training the ALS model...")
        als = ALS(userCol=userCol, itemCol=itemCol, ratingCol=ratingCol,rank=rank,
                         maxIter=iterations, regParam=.05, nonnegative=True,coldStartStrategy="drop", implicitPrefs=False)
        
        indexer1 = StringIndexer(inputCol='asin', outputCol='asin_Idx')
        training_data = indexer1.fit(training_data).transform(training_data)

        indexer2 = StringIndexer(inputCol='reviewerID', outputCol='reviewerID_Idx')
        training_data = indexer2.fit(training_data).transform(training_data)
        
        pipeline = Pipeline(stages=[indexer1,indexer2, als])

        t0 = time()
        als_model = pipeline.fit(training_data)
        tt = time() - t0
        
        logger.info("New model trained in %s seconds",round(tt,3))
        
        #als_model.save("gs://sh_books_bucket/code/model")
        return als_model
        
        
        
    def evaluate_model(self,test_data_df):
        
        logger.info("Evaluating Model...")

        test_predictions = self.als_model.transform(test_data_df)

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall",predictionCol="prediction")
        rmse = evaluator.evaluate(test_predictions)
             
        logger.info("RMSE:",rmse)
        return rmse
    
    def get_books(self,no_of_books):
        logger.info("Within get_books...")
        book_list=[]
        #no_of_books = 10
        list_books = self.data_books.select('asin','title').orderBy(rand()).limit(no_of_books).collect()
        for book in list_books:
            book_list.append(book[0])
        logger.info("List of books",book_list)
        return book_list
    
    
    """Add additional movie rating in the format (reviewer, asin, rating)"""
    def predict_ratings(self,new_user_id,pd_dataframe):
        
        # Creating a schema for the newuser dataframe.
        t0 = time()
        schema = StructType([
                 StructField("reviewerID", StringType(), True),
                 StructField("asin", StringType(), True),
                 StructField("overall", FloatType(), True)
                ])
        
        df_newuser = self.spark.createDataFrame(pd_dataframe, schema)
        
        logger.info("New dataframe created...",df_newuser.show())
        
        
        df_combined = self.df_sampled.unionByName(df_newuser)
                
        new_als_model = self.train_model(df_combined,20,10,'reviewerID_Idx','asin_Idx','overall')
        new_user = df_combined.where(df_combined.reviewerID==new_user_id).select('reviewerID_Idx').distinct()
        
        new_user_predictions=new_als_model.recommendForUserSubset(new_user,10)
        logger.info("Recommending for new user...")
        recommendationsDF = (new_user_predictions
                              .select("reviewerID_Idx", explode("recommendations")
                              .alias("recommendation"))
                              .select("reviewerID_Idx", "recommendation.*")
                            ) 
        
        logger.info("Creating views...")
        recommendationsDF.createTempView("recommendations")
        df_combined.createTempView("reviews")
        self.data_books.createTempView("books")

        logger.info("Running sql...")
        recom_df = self.spark.sql("select distinct bks.asin,bks.title,rec.rating from recommendations rec JOIN reviews rev ON rec.asin_Idx=rev.asin_Idx JOIN books bks ON bks.asin=rev.asin ORDER BY rec.rating desc").show()

        tt = time() - t0
        logger.info("prediction took...",round(tt,3))
        return recom_df
        
    
   

    
    def to_StringIndex(self,colname,dataframe):
        indexer = StringIndexer(inputCol=colname, outputCol=colname+'_Idx')
        dataframe = indexer.fit(dataframe).transform(dataframe)
        return dataframe
    


    
    
    
    

