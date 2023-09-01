# Databricks notebook source
pip install nltk

# COMMAND ----------

df = spark.read.option("header","true").option("inferSchema","true").csv("dbfs:/FileStore/tables/Language_Detection.csv")

# COMMAND ----------

from pyspark.sql.functions import col
listOfLanguages = ['English','French','Spanish','Portugeese','Italian']
 
df = df.filter(col("Language").isin(listOfLanguages))

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import math
 
def clean_func(Text):
  Text = re.sub(r'[\([{})\]!@#$,"%^*?:;~`0-9]', ' ', Text)   # removing the symbols and numbers
  Text = Text.lower()   # converting the text to lower case
  Text = re.sub('#\S+', '', Text)  # remove hashtags
 
  return Text
df1 = df.rdd.map(lambda x:(clean_func(x[0]),x[1])).toDF().withColumnRenamed("_1", "Text").withColumnRenamed("_2", "Language")
df1.collect()
regexTokenizer = RegexTokenizer(inputCol="Text", outputCol="words", pattern="\\W+")
df1 = regexTokenizer.transform(df1)
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
df1 = stopwordsRemover.transform(df1)
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return lemmatized_text
 
# Convert the lemmatization function to a UDF
lemmatize_udf = udf(lemmatize_text, StringType())
df1.withColumn("lemmatized_text", lemmatize_udf("filtered"))

# COMMAND ----------

train_ratio = 0.7  # Training data ratio
test_ratio = 0.3   # Testing data ratio
train_df, test_df = df1.randomSplit([train_ratio, test_ratio], seed=42)

# COMMAND ----------

from math import log

# COMMAND ----------

display(train_df)

# COMMAND ----------

display(train_df)

# COMMAND ----------

total_samples = train_df.count()

# COMMAND ----------

total_samples

# COMMAND ----------

class_counts = train_df.groupBy("Language").count()

# COMMAND ----------

display(class_counts)

# COMMAND ----------

class_counts_dictionary = dict(class_counts.rdd.map(lambda x: (x[0], x[1])).collect())

# COMMAND ----------

# Function to define prior probability of each class
def calculate_prior_probability(class_counts, total_num_samples):
    prior_probabilities = {}
    for class_label, count in class_counts.items():
        prior_probabilities[class_label] = count / total_num_samples
    return prior_probabilities

# COMMAND ----------

prior_probabilities = calculate_prior_probability(class_counts_dictionary, total_samples)

# COMMAND ----------

prior_probabilities

# COMMAND ----------

class_counts.show()

# COMMAND ----------

def calculate_log_likelihood(words_list, language_label, class_word_dictionary, vocabulary_dictionary):
    log_likelihood_value = 0.0
    for word in words_list:
        number_of_words = vocabulary_dictionary.get((language_label, word), 0.0)
        word_probability = (number_of_words + 1) / (class_word_dictionary[language_label] + len(class_word_dictionary))
        log_likelihood_value += math.log(word_probability)
    return log_likelihood_value

# COMMAND ----------

def naive_bayes_fit(train_df):
    class_word_dictionary = {}
    vocabulary_dictionary = {}
    for language_label in prior_probabilities:
        text_data = train_df.filter(train_df.Language == language_label).select("words").rdd.flatMap(lambda word: word).collect()
        distinct_words = set()
        for word_list in text_data:
            for word in word_list:
                distinct_words.add(word)
                vocabulary_dictionary[(language_label, word)] = vocabulary_dictionary.get((language_label, word), 0.0) + 1.0
        class_word_dictionary[language_label] = len(distinct_words)
    return  class_word_dictionary, vocabulary_dictionary

# COMMAND ----------

class_word_dictionary, vocabulary_dictionary = naive_bayes_fit(train_df)

# COMMAND ----------

class_word_dictionary 

# COMMAND ----------

vocabulary_dictionary 

# COMMAND ----------

total_samples_test = test_df.count()

# COMMAND ----------

total_samples_test

# COMMAND ----------

number_of_predictions = 0
output = []
for document in test_df.rdd.collect():
    language_label = document["Language"]
    words_list = document["words"]
    predicted_label = None
    predicted_probability = float("-inf")
    for current_language_label in prior_probabilities:
        likelihood_value = calculate_log_likelihood(words_list, current_language_label, class_word_dictionary, vocabulary_dictionary)
        probability = math.log(prior_probabilities[current_language_label]) + likelihood_value
        if probability > predicted_probability:
            predicted_probability = probability
            predicted_label = current_language_label
    output.append((language_label, predicted_label))
    if predicted_label == language_label:
        number_of_predictions += 1

# COMMAND ----------

predictions_accuracy = number_of_predictions / total_samples_test

# COMMAND ----------

print("Accuracy of the Predictions : " + str(predictions_accuracy))

# COMMAND ----------

print("Prior probabilities : ", prior_probabilities)
