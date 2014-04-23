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
        self.attributes = range(np.shape(self.collection)[1])

    def split_on_best_attribute(self, collection, classifiers):
        '''
        collection and classifiers are subsets of the data that the class was
        initialized with.  This method finds gets the attribute that contains
        the most information and splits collection and classifiers by this
        attribute.  It returns the attribute, an array of the first set of
        classifiers and its collection, as well as the second set of
        classifiers and its collection
        '''
        split_attribute = self.find_best_attribute(collection, classifiers)
        classifier1, classifier2 = [], []
        collection1, collection2 = [], []
        for row in range(np.shape(collection)[0]):
            if collection[row][split_attribute] == True:
                classifier1.append(classifiers[row])
                collection1.append(collection[row])
            else:
                classifier2.append(classifiers[row])
                collection2.append(collection[row])
            right_node = [np.array(collection1), classifier1]
            left_node = [np.array(collection2), classifier2]
        return split_attribute, right_node, left_node


    def find_best_attribute(self, collection, classifiers):
        '''
        collection and classifiers are subsets of the data that the class was
        initialized with.  This method loops through all attributes and finds
        the one that provides the greatest information gain.  It then removes
        the attribute from the instance variable list of attributes
        '''
        best_attribute = 0
        score = 0
        for attribute in self.attributes:
            information_gain = self.information_gain(collection, classifiers, attribute)
            if information_gain > score:
                score = information_gain
                best_attribute = attribute
        self.attributes.remove(best_attribute)
        return best_attribute

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

    def information_gain(self, collection, classifiers,  attribute_index):
        '''
        Calculates the entropy reduction of classifiers
        after the collection is split by values in the
        attribute_index column of the collection
        '''
        attribute_hist = ct.Counter(collection[:,attribute_index])
        total_states = sum(attribute_hist.values())
        entropy = 0
        for attribute_value, count in attribute_hist.iteritems():
            split_classifier = []
            for row in range(np.shape(collection)[0]):
                if collection[row][attribute_index] == attribute_value:
                    split_classifier.append(classifiers[row])
            reduced_entropy = self.calculate_entropy(split_classifier)
            p_attribute_value = float(count)/total_states
            entropy += reduced_entropy * p_attribute_value
        return self.total_entropy - entropy


