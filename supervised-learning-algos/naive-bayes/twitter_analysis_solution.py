# built-in python library for interacting with .csv files
import csv
# natural language toolkit is a great python library for natural language processing
import nltk
# built-in python library for utility functions that introduce randomness
import random
# built-in python library for measuring time-related things
import time


def get_length_bucket(tweet_length):
    """
    buckets the tweet length into either short / medium / long
    """
    if tweet_length < 20:
        return "short"
    elif tweet_length < 70:
        return "medium"
    else:
        return "long"

def contains_exclamation(tweet):
    if ("!" in tweet):
        return True
    else:
        return False

def find_sentiment(tweet):
    sad_words = ["sad", "sorry", "oops", "disappoint", "oops"]
    
    if ("sad" in tweet):
        return "negative"
    if ("happy" in tweet):
        return "positive"
    else:
        return "unknown"
    
def twitter_features(tweet):
    """
     Returns a dictionary of the features of the tweet we want our model
    to be based on, e.g. tweet_length.

    So if the tweet was "Hey!", the output of this function would be
    {
        "length": "short"
    }

    If the tweet was "Hey this is a really great idea and I think that we should totally implement this technique",
    then the output would be
    {
        "length": "medium"
    }
    """
    return {
        "length": get_length_bucket(len(tweet)),
        "exclamation": "!" in tweet,
        "sad" : "sad" in tweet,
        "happy" : "happy" in tweet,
        "wow" : ("wow" in tweet or "Wow" in tweet),
        "thanks" : ("thanks" in tweet or "Thanks" in tweet),
        "sorry" : ("sorry" in tweet or "Sorry" in tweet),
        "great" : ("great" in tweet or "Great" in tweet),
        "good" : ("good" in tweet or "Good" in tweet),
        "cool" : ("cool" in tweet or "Cool" in tweet),
        "oops" : ("oops" in tweet or "Oops" in tweet),
        "excite" : ("excite" in tweet or "Excite" in tweet),
        "loved" : ("loved" in tweet or "Loved" in tweet),
        "try" : ("try" in tweet or "Try" in tweet),
        #"disappoint" : ("disappoint" in tweet or "Disappoint" in tweet),
        #"check" : ("check" in tweet or "Check" in tweet),
        #"best" : ("best" in tweet or "Best" in tweet),
    }


def get_feature_sets():
    """
    # Step 1: This reads in the rows from the csv file which look like this:
    0, I'm so sad
    1, Happy!

    where the first row is the label; 0=negative, 1=positive
    and the second row is the body of the tweet

    # Step 2: Turn the csv rows into feature dictionaries using `twitter_features` function above.

    The output of this function run on the example in Step 1 will look like this:
    [
        ({"length": "short"}, 0), # this corresponds to 0, I'm so sad
        ({"length": "short"}, 1) # this corresponds to 1, Happy!
    ]

    You can think about this more abstractly as this:
    [
        (feature_dictionary, label), # corresponding to row 0
        ... # corresponding to row 1 ... n
    ]
    """
    # open the file, which we've placed at /home/vagrant/repos/datasets/clean_twitter_data.csv
    # 'rb' means read-only mode and binary encoding
    f = open('/home/vagrant/repos/datasets/clean_twitter_data.csv', 'rb')

    # let's read in the rows from the csv file
    rows = []
    for row in csv.reader(f):
        rows.append(row)

    # now let's generate the output that we specified in the comments above
    output_data = []

    # let's just run it on 100,000 rows first, instead of all 1.5 million rows
    # when you experiment with the `twitter_features` function to improve accuracy
    # feel free to get rid of the row limit and just run it on the whole set
    for row in rows[:200000]:
        # Remember that row[0] is the label, either 0 or 1
        # and row[1] is the tweet body

        # get the label
        label = row[0]

        # get the tweet body and compute the feature dictionary
        feature_dict = twitter_features(row[1])

        # add the tuple of feature_dict, label to output_data
        data = (feature_dict, label)
        output_data.append(data)

    # close the file
    f.close()
    return output_data


def get_training_and_validation_sets(feature_sets):
    """
    This takes the output of `get_feature_sets`, randomly shuffles it to ensure we're
    taking an unbiased sample, and then splits the set of features into
    a training set and a validation set.
    """
    # randomly shuffle the feature sets
    random.shuffle(feature_sets)

    # get the number of data points that we have
    count = len(feature_sets)
    # 20% of the set, also called "corpus", should be training, as a rule of thumb, but not gospel.

    # we'll slice this list 20% the way through
    slicing_point = int(.20 * count)

    # the training set will be the first segment
    training_set = feature_sets[:slicing_point]

    # the validation set will be the second segment
    validation_set = feature_sets[slicing_point:]
    return training_set, validation_set


def run_classification(training_set, validation_set):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # let's see how accurate it was
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

def predict(classifier, new_tweet):
    """
    Given a trained classifier and a fresh data point (a tweet),
    this will predict its label, either 0 or 1.
    """
    return classifier.classify(twitter_features(new_tweet))


# Now let's use the above functions to run our program
start_time = time.time()

print "Let's use Naive Bayes!"

our_feature_sets = get_feature_sets()
our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
print "Size of our data set: {}".format(len(our_feature_sets))

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)
classifier.show_most_informative_features()

end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)
