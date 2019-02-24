import csv, string, math, copy
from twitter_specials import *
import numpy as np


def label_count(file):
    label_proportion = {'positive':0, 'negative':0, 'neutral':0, 'irrelevant':0}
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        curr_line = list(row)
        if curr_line[1] == 'positive':
            label_proportion['positive']+=1
        elif curr_line[1] == 'negative':
            label_proportion['negative']+=1
        elif curr_line[1] == 'neutral':
            label_proportion['neutral']+=1
        else:
            label_proportion['irrelevant']+=1
    return label_proportion


def count_work_prop():
    word_counts = dict()
    exclude = set(string.punctuation)
    with open("labeled_corpus.tsv", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            curr_line = list(row)
            tweet = curr_line[0]
            tweet = clean_tweet(tweet, emo_repl_order, emo_repl, re_repl)
            words = tweet.split()
            for w in words:
                if '#' not in w and '@' not in w and '/' not in w and w != 'rt' and w != 'RT':
                    w = ''.join(ch for ch in w if ch not in exclude)
                    if w in word_counts:
                        if curr_line[1] == 'positive':
                            word_counts[w]['positive'] += 1
                        elif curr_line[1] == 'negative':
                            word_counts[w]['negative'] += 1
                        elif curr_line[1] == 'neutral':
                            word_counts[w]['neutral'] += 1
                        else:
                            word_counts[w]['irrelevant'] += 1
                    else:
                        word_counts[w] = {'positive':0, 'negative':0, 'neutral':0, 'irrelevant':0}
                        if curr_line[1] == 'positive':
                            word_counts[w]['positive'] += 1
                        elif curr_line[1] == 'negative':
                            word_counts[w]['negative'] += 1
                        elif curr_line[1] == 'neutral':
                            word_counts[w]['neutral'] += 1
                        else:
                            word_counts[w]['irrelevant'] += 1
    return word_counts


def calc_cond_prob(dict, probs):
    for key in dict.keys():
        dict[key]['positive'] = dict[key]['positive'] / probs.get('positive')
        dict[key]['negative'] = dict[key]['negative'] / probs.get('negative')
        dict[key]['neutral'] = dict[key]['neutral'] / probs.get('neutral')
        dict[key]['irrelevant'] = dict[key]['irrelevant'] / probs.get('irrelevant')


def classify_tweet(file, dict, probs):
    prediction = []
    exclude = set(string.punctuation)
    reader = csv.reader((x.replace('\0', '') for x in file), delimiter='\t')
    # reader = csv.reader(file, delimiter='\t')
    for row in reader:
        curr_line = list(row)
        curr_prediction = [curr_line[0], curr_line[1]]
        tweet = curr_line[2]
        tweet = clean_tweet(tweet, emo_repl_order, emo_repl, re_repl)
        words = tweet.split()
        curr_prob = [math.log(probs[0]), math.log(probs[1]), math.log(
            probs[2]), math.log(probs[3])]
        for w in words:
            if '#' not in w and '@' not in w and '/' not in w and w != 'rt' and w != 'RT':
                w = ''.join(ch for ch in w if ch not in exclude)
                if w in dict:
                    try:
                        curr_prob[0] += math.log(dict[w]['positive'])
                        curr_prob[1] += math.log(dict[w]['negative'])
                        curr_prob[2] += math.log(dict[w]['neutral'])
                        curr_prob[3] += math.log(dict[w]['irrelevant'])
                    except:
                        continue
                else:
                    continue
        if np.argmax(curr_prob) == 0:
            curr_prediction.append('positive')
            prediction.append(curr_prediction)
        elif np.argmax(curr_prob) == 1:
            curr_prediction.append('negative')
            prediction.append(curr_prediction)
        elif np.argmax(curr_prob) == 2:
            curr_prediction.append('neutral')
            prediction.append(curr_prediction)
        elif np.argmax(curr_prob) == 3:
            curr_prediction.append('irrelevant')
            prediction.append(curr_prediction)

    return prediction


def area_score(prediction):
    score = []
    curr_square = []
    for row in range(len(prediction)):
        curr_dict = {}
        pos, neg = 0.0, 0.0
        if not curr_square:
            curr_square.append(prediction[row])
            continue
        else:
            if prediction[row][0] == curr_square[0][0]:
                if prediction[row][1] == curr_square[0][1]:
                    curr_square.append(prediction[row])
                    continue
        for curr in range(len(curr_square)):
            if curr_square[curr][2] == 'positive':
                pos += 1
                continue
            elif curr_square[curr][2] == 'negative':
                neg += 1
                continue
        curr_score = pos/len(curr_square) - neg/len(curr_square)
        curr_score = (curr_score + 1)/2
        curr_square = [curr_square[0][0], curr_square[0][1], curr_score]
        square_to_add = copy.deepcopy(curr_square)
        curr_dict["score"] = square_to_add[2]
        curr_dict['g'] = float(square_to_add[1]) + (0.05/2)
        curr_dict['t'] = float(square_to_add[0]) + (0.05/2)
        score.append(curr_dict)
        curr_square.clear()

    return score


