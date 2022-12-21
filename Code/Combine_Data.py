import re
import sys
import os
import pyspark
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import split
import pyspark.sql.functions as F
from pyspark.sql.functions import *

N = 0
conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")
# user_name, user_location date text
df = spark.read.options(header = 'True').csv("./covid19_tweets.csv")
df.printSchema()

df_78 = df.drop("user_description", "user_created", "user_created", "user_followers", "user_favourites", "user_verified", "hashtags", "source", "is_retweet", "user_friends")
df_78.printSchema()
df_78 = df_78.withColumn('date', F.date_format(F.to_date('date', "yyyy-MM-dd hh:mm:ss"),'dd-MM-yyyy'))
df_78.show()

df_46 = spark.read.options(header = 'True').csv("./archive/Covid-19 Twitter Dataset (Apr-Jun 2020).csv")
df_46.printSchema()
df_46_test = df_46.drop("source", "original_text", "lang", "favorite_count", "retweet_count", "original_author", "hashtags", "user_mentions", "compound", "neg", "neu", "pos")
df_46 = df_46.drop("source", "original_text", "lang", "favorite_count", "retweet_count", "original_author", "hashtags", "user_mentions", "compound", "neg", "neu", "sentiment", "pos")


df_46_test.printSchema()
df_46_test = df_46_test.selectExpr("id as user_name", "created_at as date", "place as user_location", "clean_tweet as text", "sentiment as sentiment")
df_46_test = df_46_test.select("user_name", "user_location", "date", "text", "sentiment")
df_46_test = df_46_test.withColumn('mon', split(df_46_test['date'], ' ').getItem(1)).withColumn('day', split(df_46_test['date'], ' ').getItem(2)).withColumn('year', split(df_46_test['date'], ' ').getItem(5))
df_46_test = df_46_test.withColumn('date', F.concat(F.col('year'), F.lit('-'), F.col('mon')))
df_46_test = df_46_test.withColumn('date', F.concat(F.col('date'), F.lit('-'), F.col('day')))
df_46_test = df_46_test.withColumn('date', F.date_format(F.to_date('date', "yyyy-MMM-dd"),'dd-MM-yyyy'))
df_46_test = df_46_test.drop("mon", 'day', 'year')
######

df_46.printSchema()
df_46 = df_46.selectExpr("id as user_name", "created_at as date", "place as user_location", "clean_tweet as text")
df_46 = df_46.select("user_name", "user_location", "date", "text")
df_46 = df_46.withColumn('mon', split(df_46['date'], ' ').getItem(1)).withColumn('day', split(df_46['date'], ' ').getItem(2)).withColumn('year', split(df_46['date'], ' ').getItem(5))
df_46 = df_46.withColumn('date', F.concat(F.col('year'), F.lit('-'), F.col('mon')))
df_46 = df_46.withColumn('date', F.concat(F.col('date'), F.lit('-'), F.col('day')))
df_46 = df_46.withColumn('date', F.date_format(F.to_date('date', "yyyy-MMM-dd"),'dd-MM-yyyy'))
df_46 = df_46.drop("mon", 'day', 'year')
df_46.show()


df_89 = spark.read.options(header = 'True').csv("./archive/Covid-19 Twitter Dataset (Aug-Sep 2020).csv")
df_89.printSchema()
df_89_test = df_89.drop("source", "original_text", "lang", "favorite_count", "retweet_count", "original_author", "hashtags", "user_mentions", "compound", "neg", "neu")
df_89 = df_89.drop("source", "original_text", "lang", "favorite_count", "retweet_count", "original_author", "hashtags", "user_mentions", "compound", "neg", "neu", "sentiment")
df_89.printSchema()

df_89_test = df_89_test.selectExpr("id as user_name", "created_at as date", "place as user_location", "clean_tweet as text", "sentiment as sentiment")
df_89_test = df_89_test.select("user_name", "user_location", "date", "text", "sentiment")
df_89_test = df_89_test.withColumn('mon', split(df_89_test['date'], ' ').getItem(1)).withColumn('day', split(df_89_test['date'], ' ').getItem(2)).withColumn('year', split(df_89_test['date'], ' ').getItem(5))
df_89_test = df_89_test.withColumn('date', F.concat(F.col('year'), F.lit('-'), F.col('mon')))
df_89_test = df_89_test.withColumn('date', F.concat(F.col('date'), F.lit('-'), F.col('day')))
df_89_test = df_89_test.withColumn('date', F.date_format(F.to_date('date', "yyyy-MMM-dd"),'dd-MM-yyyy'))
df_89_test = df_89_test.drop("mon", 'day', 'year')
df_89_test.show()

df_89 = df_89.selectExpr("id as user_name", "created_at as date", "place as user_location", "clean_tweet as text")
df_89 = df_89.select("user_name", "user_location", "date", "text")
df_89 = df_89.withColumn('mon', split(df_89['date'], ' ').getItem(1)).withColumn('day', split(df_89['date'], ' ').getItem(2)).withColumn('year', split(df_89['date'], ' ').getItem(5))
df_89 = df_89.withColumn('date', F.concat(F.col('year'), F.lit('-'), F.col('mon')))
df_89 = df_89.withColumn('date', F.concat(F.col('date'), F.lit('-'), F.col('day')))
df_89 = df_89.withColumn('date', F.date_format(F.to_date('date', "yyyy-MMM-dd"),'dd-MM-yyyy'))
df_89 = df_89.drop("mon", 'day', 'year')
df_89.show()


df_3 = spark.read.options(header = 'True').csv("./Corona_NLP_train.csv")
df_3.printSchema()
df_3 = df_3.drop("ScreenName", "Sentiment")
df_3 = df_3.selectExpr("UserName as user_name", "Location as user_location", "TweetAt as date", "OriginalTweet as text")
df_3.show()

# 1163998
unionDF = df_3.union(df_46).union(df_78).union(df_89)
unionDF.show()

unionDF = unionDF.na.drop(subset=["text"])
unionDF.show()
# Find time 
df_date = unionDF.select("date")
df_date.show()
df_date = df_date.dropDuplicates()
pattern = "yyyy-MM-dd"
df_date = df_date.select(F.col("date"), to_date(F.col('date'), "dd-MM-yyyy").alias('date_new')).na.drop(subset = ['date_new'])
df_date = df_date.orderBy(unix_timestamp(F.col("date_new"), pattern).cast("timestamp"))
df_date.show()
df_date.coalesce(1).write.option("header", True).csv('date')
# unionDF_label = df_46_test.union(df_89_test)
# unionDF_label = unionDF_label.na.drop(subset = ["text"]) 
# unionDF_label = unionDF_label.na.drop(subset = ["sentiment"])
# unionDF_label.show()
# unionDF_label.coalesce(1).write.option("header", True).csv('data_label.csv')

# # drop null location
# #unionDF = unionDF.drop("user_location")
# unionDF = unionDF.na.drop(subset = ["date"]) 
# unionDF = unionDF.na.drop(subset = ["user_location"])
# r = "\A[\pL\s]+\z"
# unionDF = unionDF.withColumn("user_location", when(F.col("user_location").rlike(r), F.col('user_location')).otherwise(None))
# #unionDF = unionDF.where(np.char.isalpha(i) for i in unionDF.withColumn('user_location'))
# unionDF = unionDF.na.drop(subset = ["user_location"])
# unionDF = unionDF.select("*", F.lower("user_location"))
# unionDF = unionDF.drop('user_location').select(F.col('user_name'), F.col('date'), F.col('text'), F.col('lower(user_location)').alias('user_location'))
# unionDF = unionDF.withColumn("country", F.lit(None))
# unionDF = unionDF.withColumn("country", when(unionDF.user_location == "india", F.lit("India"))
# .when(unionDF.user_location == "delhi", F.lit("India"))
# .when(unionDF.user_location == "munbai", F.lit("India"))
# .when(unionDF.user_location == "bengaluru", F.lit("India"))
# .when(unionDF.user_location == "bangalore", F.lit("India"))
# .when(unionDF.user_location == "bhubaneswar", F.lit("India"))
# .when(unionDF.user_location == "hyderabad", F.lit("India"))
# .when(unionDF.user_location == "china", F.lit("China"))
# .when(unionDF.user_location == "beijing", F.lit("China"))
# .when(unionDF.user_location == "hong kong", F.lit("Hong Kong"))
# .when(unionDF.user_location == "singapore", F.lit("Singapore"))
# .when(unionDF.user_location == "australia", F.lit("Australia"))
# .when(unionDF.user_location == "melbourne", F.lit("Australia"))
# .when(unionDF.user_location == "sydney", F.lit("Australia"))
# .when(unionDF.user_location == "canada", F.lit("Canada"))
# .when(unionDF.user_location == "africa", F.lit("Africa"))
# .when(unionDF.user_location == "england", F.lit("UK"))
# .when(unionDF.user_location == "united kingdom", F.lit("UK"))
# .when(unionDF.user_location == "london", F.lit("UK"))
# .when(unionDF.user_location == "uk", F.lit("UK"))
# .when(unionDF.user_location == "us", F.lit("US"))
# .when(unionDF.user_location == "united states", F.lit("US"))
# .when(unionDF.user_location == "usa", F.lit("US"))
# .when(unionDF.user_location == "washington", F.lit("US"))
# .when(unionDF.user_location == "new york", F.lit("US"))
# .when(unionDF.user_location == "angeles", F.lit("US"))
# .when(unionDF.user_location == "la", F.lit("US"))
# .when(unionDF.user_location == "north america", F.lit("US"))
# .when(unionDF.user_location == "atlanta", F.lit("US"))
# .when(unionDF.user_location == "california", F.lit("US"))
# .when(unionDF.user_location == "houston", F.lit("US"))
# .when(unionDF.user_location == "chicago", F.lit("US"))
# .when(unionDF.user_location == "boston", F.lit("US"))
# .when(unionDF.user_location == "philadelphia", F.lit("US"))
# .when(unionDF.user_location == "diego", F.lit("US"))
# .when(unionDF.user_location == "seattle", F.lit("US"))
# .when(unionDF.user_location == "texas", F.lit("US"))
# .when(unionDF.user_location == "nyc", F.lit("US"))
# .when(unionDF.user_location == "vegas", F.lit("US"))
# .when(unionDF.user_location == "francisco", F.lit("US"))
# .when(unionDF.user_location == "florida", F.lit("US"))
# .when(unionDF.user_location == "dallas", F.lit("US"))
# .when(unionDF.user_location == "denver", F.lit("US"))
# .when(unionDF.user_location == "worldwide", F.lit("NoCountry"))
# .when(unionDF.user_location == "global", F.lit("NoCountry"))
# .when(unionDF.user_location == "earth", F.lit("NoCountry"))
# .when(unionDF.user_location == "everywhere", F.lit("NoCountry"))
# .when(unionDF.user_location == "nigeria", F.lit("Nigeria"))
# .when(unionDF.user_location == "kenya", F.lit("Kenya"))
# .when(unionDF.user_location == "switzerland", F.lit("Switzerland"))
# .when(unionDF.user_location == "ireland", F.lit("Ireland"))
# .when(unionDF.user_location == "canada", F.lit("Canada"))
# .when(unionDF.user_location == "toronto", F.lit("Canada"))
# .when(unionDF.user_location == "philippines", F.lit("Philippines"))
# .when(unionDF.user_location == "malaysia", F.lit("Malaysia")))

# unionDF = unionDF.na.drop(subset = ['country'])

# id = unionDF.where(unionDF.country == 'India').count()
# cn = unionDF.where(unionDF.country == 'China').count()
# hk = unionDF.where(unionDF.country == 'Hong Kong').count()
# sg = unionDF.where(unionDF.country == 'Singapore').count()
# aus = unionDF.where(unionDF.country == 'Australia').count()
# ca = unionDF.where(unionDF.country == 'Canada').count()
# uk = unionDF.where(unionDF.country == 'UK').count()
# us = unionDF.where(unionDF.country == 'US').count()
# ng = unionDF.where(unionDF.country == 'Nigeria').count()
# ky = unionDF.where(unionDF.country == 'Kenya').count()
# sz = unionDF.where(unionDF.country == 'Switzerland').count()
# il = unionDF.where(unionDF.country == 'Ireland').count()
# pl = unionDF.where(unionDF.country == 'Philippines').count()
# ms = unionDF.where(unionDF.country == 'Malaysia').count()
# print(id, cn, hk, sg, aus, ca, uk, us, ng, ky, sz, il, pl, ms)
# result = [id, cn, hk, sg, aus, ca, uk, us, ng, ky, sz, il, pl, ms]

# result.write.txt('loc')
# name_list = unionDF.user_location
#unionDF.show()

#unionDF.coalesce(1).write.option("header", True).csv('location.csv')
# #location = unionDF.select('user_location').distinct().collect()
# #print(location)
# unionDF.coalesce(1).write.csv('data.csv')