#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:35:41 2019

@author: hwan
"""

import os
from pyspark.sql import SparkSession, SQLContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import rand
from flask import Flask, flash, redirect, render_template, request, url_for, session
import pandas as pd
import pyspark
#from Engine import RecommendationEngine


#sc = pyspark.SparkContext('local[1]')
#qlContext = SQLContext(sc)

VIZ_FOLDER = os.path.join('static', 'Viz')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = VIZ_FOLDER

#commented out the below two line if want to deploy on app engine
spark = SparkSession.builder.master('local').appName('recommender').getOrCreate()
df_mb = spark.read.json("Data/metaBooks.json")

# This line will be uncommented when we integrate main.py with the RecommendationEngine running on the cluster.
#r_engine = RecommendationEngine()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/explore3')
def explore():
    viz1 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating dist.png')
    viz2 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating by time.png')
    viz3 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating by sales rank.png')
    return render_template("explore3.html", viz1 = viz1, viz2=viz2, viz3=viz3)

#commented out the below two functions if want to deploy on app engine
@app.route('/recommendation')
def recommendation():
    book_list=[]
    #list_of_books = r_engine.get_books()
    list_books = df_mb.where(df_mb.title!='None').select(['asin','title']).orderBy(rand()).limit(10).collect()
    for book in list_books:
        book_list.append((book['asin'],book['title']))
    return render_template("recommendation.html",
                           book_id0 = book_list[0][0], book_title0=book_list[0][1],
                           book_id1 = book_list[1][0], book_title1=book_list[1][1],
                           book_id2 = book_list[2][0], book_title2=book_list[2][1],
                           book_id3 = book_list[3][0], book_title3=book_list[3][1],
                           book_id4 = book_list[4][0], book_title4=book_list[4][1],
                           book_id5 = book_list[5][0], book_title5=book_list[5][1],
                           book_id6 = book_list[6][0], book_title6=book_list[6][1],
                           book_id7 = book_list[7][0], book_title7=book_list[7][1],
                           book_id8 = book_list[8][0], book_title8=book_list[8][1],
                           book_id9 = book_list[9][0], book_title9=book_list[9][1])

@app.route('/result', methods=('GET','POST'))    
def result():
    select=[]
    for i in range(10):
        val = request.form.getlist('rating'+str(i))
        select.append(val)
    while ([] in select):
        select.remove([])
    book_rating=[val[0].split(' ') for val in select]
    df_ratings = pd.DataFrame(book_rating, 
               columns =['asin', 'overall'])
    df_ratings['overall']=df_ratings['overall'].astype(float)
    
    new_user_id = ''.join(random.choice('0123456789ABCDEF') for i in range(14))
    df_ratings['reviewerID']=new_user_id
    
    #new_user_recommendations = r_engine.predict_ratings(new_user_id,df_ratings)

   
    #return render_template('result.html',  tables=[new_user_predictions.to_html(classes='data', header="true")])
    return render_template('result.html', tables=[df_ratings.to_html(classes='data', header="true")])


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)
