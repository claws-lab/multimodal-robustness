import gensim
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import os, sys, re
import pickle as pkl
from math import log2

num_topics = 20 # these are the numbe of topics for the model

# loading all the data here
with open('text_for_comparison_with_dumb.pkl', 'rb') as f: 
	text_data = pkl.load(f)
models = ['original_text', 'caption_text', 'bert_text', 'bert_adv_text', 'bert_kw_adv_text', 'gpt_ft_text', 'gpt_text', 'mismatch_t', 'text_kw_t', 'image_kw_t', 'image_text_kw_t', 'random_url']

with open('../task_humanitarian_text_img_train.pkl', 'rb') as f: 
	full_data = pkl.load(f)
all_examples = []

for point in full_data:
	all_examples.append(list(gensim.utils.tokenize(point['tweet_text'])))

# train the LDA model here
common_dictionary = Dictionary(all_examples)
common_corpus = [common_dictionary.doc2bow(text) for text in all_examples]
lda = LdaModel(common_corpus, num_topics=num_topics)

ref_examples = []
for model in models:
	if model == 'original_text':
		for key in text_data:
			point = text_data[key]
			ref_examples.append(list(gensim.utils.tokenize(point[model])))
ref_corpus = [common_dictionary.doc2bow(text) for text in ref_examples]
common_corpus = ref_corpus

for model in models:
	curr_examples = []
	print(model)
	for key in text_data:
		point = text_data[key]
		curr_examples.append(list(gensim.utils.tokenize(point[model])))
	curr_corpus = [common_dictionary.doc2bow(text) for text in curr_examples]
	assert len(curr_corpus) == len(common_corpus)
	kl_divergence_scores = []
	for var in range(len(curr_corpus)):
		curr_doc = curr_corpus[var]
		ref_doc = common_corpus[var]
		curr_vector = lda[curr_doc]
		curr_vector_final = [0 for x in range(num_topics)]
		for temp in curr_vector:
			curr_vector_final[temp[0]] = temp[1]
		ref_vector = lda[ref_doc]
		ref_vector_final = [0 for x in range(num_topics)]
		for temp in ref_vector:
			ref_vector_final[temp[0]] = temp[1]
		curr_vector = [0.00001 if x == 0 else x for x in curr_vector_final]
		ref_vector = [0.00001 if x == 0 else x for x in ref_vector_final]
		kl_divergence_scores.append(sum([ref_vector[i] * log2(ref_vector[i]/curr_vector[i]) for i in range(len(ref_vector))]))
	print(sum(kl_divergence_scores)/float(len(kl_divergence_scores)))
