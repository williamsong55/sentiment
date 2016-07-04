import random
import nltk
from nltk.corpus import names
from nltk import word_tokenize

class Decider:

    def __init__(self,name):
        self.name = name
        self.classifier=None
        self.all = list(word_tokenize(open('all.txt').read()))
        self.build_model()
        self.set = None

    def sent(self,doc):
        low = [w.lower() for w in word_tokenize(doc)]
        words = set(low)
        feats = {}
        for word in self.all:
            feats['contains({})'.format(word)] = (word in words)
        return feats

    def build_model(self):
        print("Built Model")
        labeled_sents = ([(txt, 'positive') for txt in word_tokenize(open('positive.txt').read())] + [(txt, 'negative') for txt in word_tokenize(open('negative.txt').read())]+ [(txt, 'neutral') for txt in word_tokenize(open('neutral.txt').read())])
        random.shuffle(labeled_sents)
        featuresets = [(self.sent(txt), emotion) for (txt, emotion) in labeled_sents]
        self.set = featuresets
        #train_set, test_set = featuresets[50:], featuresets[:50]
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)
        #print(nltk.classify.accuracy(self.classifier, self.set))


    def features(self):
        self.classifier.show_most_informative_features()

    def load(self):
        print("Loaded "+self.name)

    def predict(self, input):
        print(self.classifier.classify(self.sent((input))))

    def test(self):
        print("positive")

