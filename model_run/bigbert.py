"""
Author: Mike Sosa
Date  : November 2018
File  : bigberg.py

Same as candidate generation except for bert
"""


# Imports
import os
import math 
from collections import defaultdict

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

import re 
import pickle 
import multiprocessing as mp
from multiprocess import Pool
import argparse
from Bert import Bert
# --------------------------------------------------

# Args
parser = argparse.ArgumentParser()
parser.add_argument("num_cand_labels")
parser.add_argument("data")
parser.add_argument("outputfile_candidates")
parser.add_argument("bert_model")
parser.add_argument("bert_indices")
args = parser.parse_args()

"""
Pickle file of indices
"""
with open(args.bert_indices, 'rb') as m:
    b_indices = pickle.load(m)

# load the model
model = Bert(args.bert_model)
model.load()
print( "Models Loaded" )

# Loading the data file
topics = pd.read_csv(args.data)
try:
    new_frame= topics.drop('domain',1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
print( "Data Gathered" )

b_indices = list(set(b_indices))

# Models normalised in unit vectors from the indices given above in pickle files.
model1.syn0norm = (model1.syn0 / sqrt((model1.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model1.docvecs.doctag_syn0norm =  (model1.docvecs.doctag_syn0 / sqrt((model1.docvecs.doctag_syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)[d_indices]
print( "doc2vec normalized" )

model2.syn0norm = (model2.syn0 / sqrt((model2.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model3 = model2.syn0norm[w_indices]
print( "word2vec normalized" )
