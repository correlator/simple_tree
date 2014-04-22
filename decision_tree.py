import collections as ct
import numpy as np

class DecisionTree:
    def __init__(self, classifiers, collection):
        '''
        collection is an array of attributes belonging to an object
        each row in the matrix describes an object.  Classifiers is
        an array of outcomes for each object.  If the object in the
        ith row of collection is a 'no' decision, the ith element in
        classifiers is 'no'
        '''
        self.classifiers = np.array(classifiers)
        self.collection = np.array(collection)
        self.total_entropy = self.calculate_entropy(self.classifiers)

    def calculate_entropy(self, classifiers):
        '''
        Classifiers is an array of all decisions made on the training
        set [yes, no, no, yes, maybe,...].  Returns the Shannon entropy
        of the set of possible outcomes.
        sum[-p(classifier) log_2(p(classifier))]
        '''
        classifier_hist = ct.Counter(classifiers)
        total_states = sum(classifier_hist.values())
        entropy = 0
        for classifier, count in classifier_hist.iteritems():
            p_classifier = float(count)/total_states
            entropy += -p_classifier * np.log2(p_classifier)
        return entropy

    def information_gain(self, attribute_index):
        '''
        Calculates the reduced entropy of classifiers
        after the collection is split by values in the
        attribute_index column of the collection
        '''
        attribute_hist = ct.Counter(self.collection[:,attribute_index])
        total_states = sum(attribute_hist.values())
        entropy = 0
        for attribute_value, count in attribute_hist.iteritems():
            split_classifier = []
            for row in range(np.shape(self.collection)[0]):
                if self.collection[row][attribute_index] == attribute_value:
                    split_classifier.append(self.classifiers[row])
            reduced_entropy = self.calculate_entropy(split_classifier)
            p_attribute_value = float(count)/total_states
            entropy += reduced_entropy * p_attribute_value
        return self.total_entropy


