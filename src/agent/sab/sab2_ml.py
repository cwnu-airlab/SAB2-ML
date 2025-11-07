import numpy as np
import math
from collections import Counter
from .minibatch import *
from .sample import *
import torch

class Quantizer(object):
    def __init__(self, num_steps, max_uncertainty):
        self.num_steps = num_steps
        self.max_uncertainty = max_uncertainty
        self.step_size = self.max_uncertainty / float(num_steps)

    def quantizer_func_for_boudnary(self, uncertainty):
        return int(math.ceil(uncertainty / self.step_size))

class ProbTable(object):
    def __init__(self, size_of_data, num_of_classes, s_es, epochs):

        # Local Variables
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.max_uncertainty= 1.0
        self.epochs = epochs
        self.s_es = s_es
        self.fixed_term = math.exp(math.log(s_es[0]) / self.size_of_data)
        self.table = np.ones(self.size_of_data, dtype=float)

        self.before_se = None
        self.now_se = None

        # Quantizer Module
        self.quantizer = Quantizer(self.size_of_data, self.max_uncertainty)
        self.quantizer_func = self.quantizer.quantizer_func_for_boudnary
        

        # Initialize Table: equal probability being selectedupdate_sampling_probability
        for i in range(self.size_of_data):
            self.table[i] = math.pow(self.fixed_term, 1)

    def compute_s_e(self, entropy_0, entropy) :
        return self.s_es[0] * math.pow(math.exp(math.log(1/self.s_es[0])/(entropy_0*self.size_of_data)),(entropy_0-entropy)*self.size_of_data)

    def get_sampling_probability(self, quantization_index):
        return 1.0 / math.pow(self.fixed_term, quantization_index)



    def update_p_table(self, distances, normalize=False, entropy_0=1.0, entropy=1.0):
        # compute_s_e
        s_e=self.compute_s_e(entropy_0, entropy)
        self.before_se = self.now_se
        self.now_se = s_e
        # update fixed term
        for i in range(self.size_of_data):
            if distances[i] < 0 :
                self.fixed_term = math.exp(math.log(s_e) / (1* self.size_of_data))
            else :
                self.fixed_term = math.exp(math.log(s_e) / (1*self.size_of_data))

            self.table[i] = self.get_sampling_probability(self.quantizer_func(np.fabs(distances[i])))


        if normalize:
            total_sum = np.sum(self.table)
            self.table = self.table / total_sum
    def set_epoch(self, epoch):
        self.epochs[0] = epoch


class Sampler(object):
    def __init__(self, size_of_data, num_of_classes, queue_size, s_es, epochs, label_data):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.queue_size = queue_size
        self.label_data = label_data
        self.prob_table = ProbTable(self.size_of_data, self.num_of_classes, s_es, epochs)
        self.epochs = epochs

        # prediction histories of samples
        self.all_predictions = {}
        for i in range(size_of_data):
            #self.all_predictions[i] = np.zeros(queue_size, dtype=int)
            self.all_predictions[i] = [None for i in range(self.queue_size)]

        self.max_certainty = -np.log(1.0 / float(self.num_of_classes))
        self.update_counters = np.zeros(size_of_data, dtype=int)

        # distances
        self.distances = np.zeros(self.size_of_data, dtype=float)

        # entropy
        self.first_entropy = None
        self.now_entropy = None
    def get_se_table(self):
        ori_se, se = self.prob_table.get_se()
        return ori_se, se
        

    def update_queue_epoch_size(self, queue_size):
        self.queue_size = queue_size
        self.prob_table.set_epoch(queue_size)

    def async_update_prediction_matrix(self, ids, softmax_matrix):
        for i in range(len(ids)):
            id = ids[i]
            predicted_label = np.argsort(softmax_matrix[i])[::-1][:len(self.label_data[id])].tolist()

            # append the predicted label to the prediction matrix
            cur_index = self.update_counters[id] % self.queue_size
            self.all_predictions[id][cur_index] = predicted_label
            self.update_counters[id] += 1

    def update_all_uncertainties(self):
        total_uncertainty = 0.0
        processed = 0
        max_uncertainty = -np.log(1.0 / float(self.num_of_classes))

        for idx in range(self.size_of_data):
            labels = set(self.label_data[idx])
            counter = Counter()

            for prediction in self.all_predictions[idx][:self.queue_size]:
                if prediction is None:
                    continue
                if len(set(prediction) & labels) != len(labels):
                    counter.update(prediction)

            denom = float(self.queue_size) * len(labels)
            p_dict = {label: count / denom for label, count in counter.items()}

            negative_entropy = sum(value * np.log(value) for value in p_dict.values() if value)
            uncertainty = -negative_entropy / max_uncertainty

            self.distances[idx] = 1.0 - uncertainty
            total_uncertainty += uncertainty
            processed += 1

        average_entropy_score = total_uncertainty / max(processed, 1)

        if self.first_entropy is None:
            self.first_entropy = average_entropy_score
        self.now_entropy = average_entropy_score


            

    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.queue_size, dtype=int)

    def update_sampling_probability(self, normalize=False):
        self.update_all_uncertainties()
        self.prob_table.update_p_table(self.distances, normalize=normalize, entropy_0 = self.first_entropy,entropy = self.now_entropy)
