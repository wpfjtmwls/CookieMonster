from csv import DictReader
from collections import defaultdict
import re, math

import os
import argparse
parser = argparse.ArgumentParser()

# parameter for gold standard labels (top N)
N = 5
output_gs_path = "../model_run/output_goldstandard"

# model parameters
parser.add_argument("-us", "--unsupervised", help ="evaluate unsupervised model", action = "store_true")
parser.add_argument("-usft", "--unsupervised_ft", help ="evaluate unsupervised fasttext model", action = "store_true")
parser.add_argument("-usdb", "--unsupervised_db", help ="evaluate unsupervised dbpedia model", action = "store_true")
parser.add_argument("-s", "--supervised", help ="evaluate supervised model", action = "store_true")

# number of topics per corpus
N_blogs = 45
N_books = 38
N_news = 60
N_pubmed = 85

gold_topic_labels = defaultdict(lambda: {})
with open("annotated_dataset.csv", "r") as f:
    reader = DictReader(f, delimiter="\t")
    for row in reader:
        annotator_scores = [float(score) for field, score in row.items() \
            if "annotator" in field and score != ""]
        avg_score = sum(annotator_scores)/len(annotator_scores)
        topic_id = int(float(row["topic_id"]))
        gold_topic_labels[topic_id][row["label"].replace(" ", "_")] = avg_score

def generate_gs(n):
    g = open(output_gs_path , 'w')
    for topic_id, lbl_scores in gold_topic_labels.items():
        top_p = sorted(lbl_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        g.write("Top " +str(n)+ " labels for topic " +str(topic_id) +" are:" +"\n")
        for lbl, score in top_p:
            g.write(lbl + " " + str(score) +"\n")
        g.write("\n")
    g.close()

# generate gold standard labels for top N labels
print ("Generating gold standard labels")
generate_gs(N)

def parse_output_file(fname):
    """Parses labels for each topic from the output files generated by the Learn2Rank SVM. """
    d = defaultdict(lambda: [])
    with open(fname, "r") as f:
        curr_topic = None
        expected_count = 0
        curr_count = 0
        for line in f:
            line = line.strip()
            if curr_count < expected_count and curr_topic is not None:
                d[curr_topic].append(line)
                curr_count += 1
            elif curr_topic is None:
                m = re.match(r"Top ([0-9]+) labels for topic ([0-9]+) are:", line)
                if m:
                    expected_count = int(m.group(1))
                    curr_topic = int(m.group(2))
                    curr_count = 0
            else:
                curr_count = 0
                expected_count = 0
                curr_topic = None
    return d

def DCG_p(results, topic, p):
    """Computes DCG@p for given results and topic"""
    rel = lambda label: gold_topic_labels[topic][label]
    top_p = results[:p]
    dcg = 0
    for idx, label in enumerate(top_p):
        rank = idx + 1
        if idx == 0:
            dcg += rel(label)
            continue
        dcg += rel(label)/ math.log(rank,2)
    return dcg

def IDCG_p(topic, p):
    """ Computes the Idealized-DCG@p for a given topic. Based on gold standard rankings """
    lbl_scores = gold_topic_labels[topic].items() # (label, score) list
    top_p = sorted(lbl_scores, key=lambda x: x[1], reverse=True)[:p]
    idcg = 0
    for idx, (label, rel) in enumerate(top_p):
        rank = idx + 1
        if idx == 0:
            idcg += rel
            continue
        idcg += rel/ math.log(rank,2)
    return idcg

def nDCG_p(results, topic, p):
    """
    Computes the normalized DCG@p (DCG scaled down by the truths)
    Source: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    nDCG_p = DCG_p / IDCG_p
    """
    return DCG_p(results, topic, p) / IDCG_p(topic, p)

def avg_nDCG_p(model_topic_labels, p):
    """ Compute the average nDCG@p from all the topics in model_topic_labels."""
    blog_model_dcg_sum = 0
    book_model_dcg_sum = 0
    news_model_dcg_sum = 0
    pubmed_model_dcg_sum = 0

    for topic_id, labels in model_topic_labels.items():
        if topic_id < N_blogs:
            blog_model_dcg_sum += nDCG_p(labels, topic_id, p)
        elif topic_id < N_blogs + N_books:
            book_model_dcg_sum += nDCG_p(labels, topic_id, p)
        elif topic_id < N_blogs + N_books + N_news:
            news_model_dcg_sum += nDCG_p(labels, topic_id, p)
        else:
            pubmed_model_dcg_sum += nDCG_p(labels, topic_id, p)

    return (blog_model_dcg_sum / N_blogs, book_model_dcg_sum / N_books, \
            news_model_dcg_sum / N_news, pubmed_model_dcg_sum / N_pubmed)

def top_1_avg(model_topic_labels):
    """ Compute the average top_1 score from all the topics in model_topic_labels."""
    blog_model_sum = 0
    book_model_sum = 0
    news_model_sum = 0
    pubmed_model_sum = 0
    blog_gold_sum = 0
    book_gold_sum = 0
    news_gold_sum = 0
    pubmed_gold_sum = 0

    for topic_id, labels in model_topic_labels.items():
        top_model_label = labels[0]
        model_score = gold_topic_labels[topic_id][top_model_label]
        gold_score = max(gold_topic_labels[topic_id].values())
        if topic_id < N_blogs:
            blog_model_sum += model_score
            blog_gold_sum += gold_score
        elif topic_id < N_blogs + N_books:
            book_model_sum += model_score
            book_gold_sum += gold_score
        elif topic_id < N_blogs + N_books + N_news:
            news_model_sum += model_score
            news_gold_sum += gold_score
        else:
            pubmed_model_sum += model_score
            pubmed_gold_sum += gold_score

    top1avg_blogs, upper_bound_blogs = blog_model_sum / N_blogs, blog_gold_sum / N_blogs
    top1avg_books, upper_bound_books = book_model_sum / N_books, book_gold_sum / N_books
    top1avg_news, upper_bound_news = news_model_sum / N_news, news_gold_sum / N_news
    top1avg_pubmed, upper_bound_pubmed = pubmed_model_sum / N_pubmed, pubmed_gold_sum / N_pubmed

    return (top1avg_blogs, upper_bound_blogs, top1avg_books, upper_bound_books, \
    top1avg_news, upper_bound_news, top1avg_pubmed, upper_bound_pubmed)

def evaluate_model(model_topic_labels):

    nDCG_1_blogs, nDCG_1_books, nDCG_1_news, nDCG_1_pubmed = avg_nDCG_p(model_topic_labels, 1)
    nDCG_3_blogs, nDCG_3_books, nDCG_3_news, nDCG_3_pubmed = avg_nDCG_p(model_topic_labels, 3)
    nDCG_5_blogs, nDCG_5_books, nDCG_5_news, nDCG_5_pubmed = avg_nDCG_p(model_topic_labels, 5)

    top1avg_blogs, upper_bound_blogs, top1avg_books, upper_bound_books, \
    top1avg_news, upper_bound_news, top1avg_pubmed, upper_bound_pubmed = top_1_avg(model_topic_labels)

    print ("\nnDCG_1_blogs : %.2f" % nDCG_1_blogs)
    print ("nDCG_1_books : %.2f" % nDCG_1_books)
    print ("nDCG_1_news : %.2f" % nDCG_1_news)
    print ("nDCG_1_pubmed : %.2f \n" % nDCG_1_pubmed)

    print ("nDCG_3_blogs : %.2f" % nDCG_3_blogs)
    print ("nDCG_3_books : %.2f" % nDCG_3_books)
    print ("nDCG_3_news : %.2f" % nDCG_3_news)
    print ("nDCG_3_pubmed : %.2f \n" % nDCG_3_pubmed)

    print ("nDCG_5_blogs : %.2f" % nDCG_5_blogs)
    print ("nDCG_5_books : %.2f" % nDCG_5_books)
    print ("nDCG_5_news : %.2f" % nDCG_5_news)
    print ("nDCG_5_pubmed : %.2f \n" % nDCG_5_pubmed)

    print ("\nTop-1 Avg blogs : %.2f" % top1avg_blogs)
    print ("Upper bound blogs : %.2f \n" % upper_bound_blogs)
    print ("Top-1 Avg books : %.2f" % top1avg_books)
    print ("Upper bound books : %.2f \n" % upper_bound_books)
    print ("Top-1 Avg news : %.2f" % top1avg_news)
    print ("Upper bound news : %.2f \n" % upper_bound_news)
    print ("Top-1 Avg pubmed : %.2f" % top1avg_pubmed)
    print ("Upper bound pubmed : %.2f \n" % upper_bound_pubmed)

args = parser.parse_args()
if args.unsupervised:  
    print ("\nEvaluting Unsupervised Model")
    fname = ("../model_run/output_unsupervised")
    d = parse_output_file(fname)
    evaluate_model(d)

elif args.unsupervised_ft:
    print ("\nEvaluating Unsupervised Model using fasttext")
    fname = ("../model_run/output_unsupervised_ft")
    d = parse_output_file(fname)
    evaluate_model(d)

elif args.unsupervised_db:  
    print ("\nEvluating Unsupervised Model using dbpedia")
    fname = ("../model_run/output_unsupervised_db")
    d = parse_output_file(fname)
    evaluate_model(d)

elif args.supervised:  
    print ("\nEvluating Supervised Model")
    fname = ("../model_run/output_supervised")
    d = parse_output_file(fname)
    evaluate_model(d)

else:
    print ("Invalid model evaluation")