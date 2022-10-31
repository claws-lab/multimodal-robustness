import os, sys, re
import itertools
import pickle as pkl
import nltk
from nltk.translate.bleu_score import SmoothingFunction

ngram = 2

with open('text_for_comparison_with_dumb.pkl', 'rb') as f: 
	text_data = pkl.load(f)

models = ['original_text', 'caption_text', 'bert_text', 'bert_adv_text', 'bert_kw_adv_text', 'gpt_ft_text', 'gpt_text', 'mismatch_t', 'text_kw_t', 'image_kw_t', 'image_text_kw_t', 'random_url']
weight = tuple((1. / ngram for _ in range(ngram)))

def computeSelfBleu(example):
	example = example.replace('.', '. ')
	example = example.replace('?', '. ')
	example = example.replace('!', '. ')
	sentences = example.split('. ')
	sentences = [x.strip() for x in sentences if x != '']
	tokenized_sentences = [x.split(' ') for x in sentences]
	tokenized_sentences = [x for x in tokenized_sentences if len(x) >= 2]
	if len(tokenized_sentences) <= 1:
		return -1.0
	count = 0
	curr_SB = 0
	for target_sentence in tokenized_sentences:
		try:
			BLEUscore = nltk.translate.bleu_score.sentence_bleu([x for x in tokenized_sentences if x!= target_sentence], target_sentence, weight, smoothing_function=SmoothingFunction().method1)
			count += 1
		except KeyError:
			print("key error occurred")
			continue
		curr_SB += BLEUscore
	if count == 0:
		return -1.0
	return curr_SB/float(count)

for model in models:
	all_examples = []
	scores = []
	for key in text_data:
		point = text_data[key]
		all_examples.append(point[model])

	for example in all_examples:
		# print(example)
		scores.append(computeSelfBleu(example))
	scores = [x for x in scores if x > 0]
	if len(scores) == 0:
		print(model + ' : ' + str(0))
		continue
	print(model + ' : ' + str(sum(scores)/float(len(scores))))
