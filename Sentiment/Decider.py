import random
import nltk
from nltk.corpus import names
from nltk import word_tokenize

class Decider:

    def __init__(self,name):
        self.name = name
        self.classifier=None

    def sent(self,doc):
        return {'word':doc}

    def build_model(self):
        print("Built\n")
        labeled_names = ([(name, 'positive') for name in word_tokenize(open('positive.txt').read())] + [(name, 'negative') for name in word_tokenize(open('negative.txt').read())]+ [(name, 'neutral') for name in word_tokenize(open('neutral.txt').read())])
        random.shuffle(labeled_names)
        featuresets = [(self.sent(n), gender) for (n, gender) in labeled_names]
        train_set, test_set = featuresets[50:], featuresets[:50]
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)

    def features(self):
        self.classifier.show_most_informative_features()

    def load(self):
        print("Loaded "+self.name)

    def predict(self, input):
        print(self.classifier.classify(self.sent((input))))

    def test(self):
        print("positive")

