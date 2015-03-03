# built-in python library for interacting with .csv files
import csv
# natural language toolkit is a great python library for natural language processing
import nltk
# built-in python library for utility functions that introduce randomness
import random
# built-in python library for measuring time-related things
import time

def contains_exclamation(sms):
    if ("!" in sms):
        return True
    else:
        return False
    
def sms_features(sms):
     
    sms = sms.lower()
    return {
    	"exclamation" : contains_exclamation(sms),
        "sms" : "sms" in sms,
        "free" : "free" in sms,
        "claim" : "claim" in sms,
        "win" : ("win" in sms or "won" in sms or "winner" in sms),
        "txt" : "txt" in sms,
        "award" : "award" in sms,
        "www" : "www" in sms,
        "stop" : "stop" in sms,
        "sex" : "sex" in sms,
        "call" : "call" in sms,
	"unsubscribe" : "unsubscribe" in sms,
        "reply" : "reply" in sms,
        "gay" : "gay" in sms,
    }


def get_feature_sets():
    # open the file, which we've placed at /home/vagrant/repos/datasets/clean_twitter_data.csv
    # 'rb' means read-only mode and binary encoding
    f = open('/home/vagrant/repos/datasets/sms_spam_or_ham.csv', 'rb')

    # let's read in the rows from the csv file
    rows = []
    for row in csv.reader(f):
        rows.append(row)

    # now let's generate the output that we specified in the comments above
    output_data = []

    for row in rows:
        # Remember that row[0] is the label, either 0 or 1
        # and row[1] is the sms body

        # get the label
        label = row[0]

        # compute the feature dictionary for sms
        feature_dict = sms_features(row[1])

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
    # find the accuracy of classifier
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

def predict(classifier, new_sms):
    """
    Given a trained classifier and a fresh data point (a sms),
    this will predict its label, either 0 or 1.
    """
    return classifier.classify(sms_features(new_sms))


# Now let's use the above functions to run our program
start_time = time.time()

print "Let's use Naive Bayes!"

our_feature_sets = get_feature_sets()
our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
print "Size of our data set: {}".format(len(our_feature_sets))

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)
classifier.show_most_informative_features(20)

end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)

