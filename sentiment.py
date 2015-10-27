import nltk

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive'),]

neg_tweets = [("I do not like this car", "negative"),
              ("This view is horrible", "negative"),
              ("I feel tired this morning", "negative"),
              ("I am not looking forward to the concert", "negative"),
              ("He is my enemy", "negative"),
              ("You are an idiot", "negative")]

test_tweets = [("I feel happy this morning", "positive"),
               ("Larry is my friend", "positive"),
               ("I do not like that man", "negative"),
               ("My house is not great", "negative"),
               ("Your song is annoying", "negative")]

def filter_tweets(list_of_tuples):
  r = []
  for (words, sentiment) in list_of_tuples:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    r.append((words_filtered, sentiment))

  return r

tweets = filter_tweets(pos_tweets + neg_tweets)
test_tweets = filter_tweets(test_tweets)

def get_words_in_tweets(tweets):
  all_words = []
  for (words, sentiment) in tweets:
    all_words.extend(words)
  return all_words

def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist).most_common()
  return [w[0] for w in wordlist]


word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
  document_words = set(document)
  features = {}
  for word in word_features:
    features["contains(%s)" % word] = (word in document_words)
  return features

training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

tweet = "Larry is an idiot"

print classifier.classify(extract_features(tweet.split()))
