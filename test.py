import build, csv, json
'''Importing dataset'''
data = open('labeled_corpus.tsv', encoding='utf-8')

'''Calculating probabilities'''
labels = build.label_count(data)
total_count_labels = sum(labels.values())
prob_positive = labels.get('positive') / total_count_labels
prob_negative = labels.get('negative') / total_count_labels
prob_neutral = labels.get('neutral') / total_count_labels
prob_irrelevant = labels.get('irrelevant') / total_count_labels
prob_list = [prob_positive, prob_negative, prob_neutral, prob_irrelevant]

'''Creating dictionary (nested) for word category counts'''
word_prob = build.count_work_prop()

'''Calculate conditional probabilities'''
for key in word_prob.keys():
    word_prob[key]['positive'] = word_prob[key]['positive'] / labels.get('positive')
    word_prob[key]['negative'] = word_prob[key]['negative'] / labels.get('negative')
    word_prob[key]['neutral'] = word_prob[key]['neutral'] / labels.get('neutral')
    word_prob[key]['irrelevant'] = word_prob[key]['irrelevant'] / labels.get('irrelevant')

'''Reading in geo_twits_squares'''
geo_tweets = open('geo_twits_squares.tsv', encoding='utf-8')
predictions = build.classify_tweet(geo_tweets, word_prob, prob_list)

'''Output predictions'''
with open('locations_classified.tsv', 'w', newline='\n') as output:
    tsv_output = csv.writer(output, delimiter='\t')
    for row in range(len(predictions)):
        tsv_output.writerow(predictions[row])

'''Assign score to each area'''
area_score = build.area_score(predictions)

'''Create JS variable'''
json_area_score = json.dumps(area_score)
with open('public_html/data.js', 'w') as output:
    output.write("var data =")
    output.write(json_area_score)

