"""
Author:         Alex Yoo
Date:           November 2018
File:           unsupervised_labels_ft.py

This file will take candidate labels and give the best labels from them using unsupervised way which is just going
to be based on dbpedia with graph centrality ranking. 
"""

import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
import argparse

import networkx as nx

# The Arguments which were giben in get_labels.py file.
parser = argparse.ArgumentParser()
parser.add_argument("num_unsup_labels") # The number of unsupervised labels.
parser.add_argument("data") # The topic data file. It contains topic terms.
parser.add_argument("output_candidates") # The file which contains candidate labels.
parser.add_argument("output_unsupervised") # The file in which output is written
args = parser.parse_args()


# Get the candidate labels form candidate labels generated by cand-generation(get_labels -cg mode)
label_list =[]
with open(args.output_candidates,'r') as k:
    for line in k:
        labels = line.split()
        label_list.append(labels[1:])

# Just get the number of labels per topic.
test_chunk_size = len(label_list[0])


# Number of Unupervised labels needed should not be less than the number of candidate labels
if test_chunk_size < int(args.num_unsup_labels):
    print ("\n")
    print ("Error")
    print ("You cannot extract more labels than present in input file")
    sys.exit()

# Reading in the topic terms from the topics file.
topics = pd.read_csv(args.data)
try:
    new_frame= topics.drop('domain',1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
print ("Data Gathered for unsupervised model")
print ("\n")

"""
This method will be used to extract the graph and choose best label based on graph centrality measure
"""

def extract_ranked_cands(seed_cands, top_links):
    """
    Extract the generated candidates' rank out from the raw order after
    DBpedia exploration.

    :param seed_cands: list of generated candidates
    :param top_links: list of (link, score) tuples 
    """
    cand_pool = set(seed_cands)
    top_cands = []
    for raw_link, _ in top_links:
        last_sep = raw_link.rfind("/")
        raw_title = raw_link[last_sep + 1:]
        title = raw_title.lower()        
        if title in cand_pool:
            top_cands.append(title)
    return top_cands

def get_best_label(label_list,num):
    # update to incorporate other graph centrality measures
    fname = "Topics/topic" +str(num)+ "G"
    G = nx.read_gml(fname)
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    
    centrality_measure = nx.betweenness_centrality(Gc)
    top_links = sorted(centrality_measure.items(), key=lambda x: x[1], reverse=True)
    cands_ranks = extract_ranked_cands(label_list, top_links)

    return cands_ranks[:int(args.num_unsup_labels)]

unsup_output = []


for j in range(len(topic_list)):
    if j % 10 == 0:
        print ("Topic " + str(j) + " has been processed")
    unsup_output.append(get_best_label(label_list[j],j))


# printing the top unsupervised labels.
print ("Printing labels for unsupervised model")
print ("\n")
g = open(args.output_unsupervised,'w')
for i,item in enumerate(unsup_output):
    print ("Top " +args.num_unsup_labels+ " labels for topic " +str(i) +" are:")
    g.write("Top " +args.num_unsup_labels+ " labels for topic " +str(i) +" are:" +"\n")
    for elem in item:
        print (elem)
        g.write(elem +"\n")
    print ("\n")
    g.write("\n")
g.close()


