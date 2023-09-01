# Language-Classification-Naive-Bayes

Build a naive bayes classifier using map reduce to predict the language of an extracted word from a document dataset. 

Pseudo Code of the algorithm implemented: 

Step 1: Data Preparation
- Download the CSV file from the provided URL
- Read the CSV file into a Spark DataFrame
- Filter the DataFrame to keep only the desired languages . Here we considered, ['English','French','Spanish','Portugeese','Italian'].

- Import all the required libraries for  text preprocessing tasks (cleaning, tokenization, stop word removal, lemmatization) on the text data

Step 2: Split the data into Training dataset and Testing dataset. Using randomSplit.

Step 3: Training Phase
- Calculate the number of words  for each language from the training data, using count().
- Calculate the prior probabilities of each language based on the class counts
  def calculate_prior_probability(class_counts, total_num_samples):

- Build the class-word dictionary and vocabulary dictionary by iterating over the training data
  def calculate_log_likelihood(words_list, language_label, class_word_dictionary, vocabulary_dictionary):



Step 4: Testing Phase
- Iterate over document in the test dataset
  - Extract the language label and words list from the document
  - Initialize variables for predicted label and probability
  - Iterate over each language label in prior probabilities
    - Calculate the log likelihood value for the words list and current language label
    - Calculate the probability using the log prior probability and log likelihood value
    - Update the predicted label and probability if the current probability is higher
  - Store the true and predicted labels in an output list
  - Increment the count of correct predictions if the predicted label matches the true label

Step 5: Calculate Accuracy
- Calculate the accuracy of the predictions by dividing the number of correct predictions by the total number of test samples
