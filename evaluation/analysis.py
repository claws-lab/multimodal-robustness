import os, sys, re
import pickle as pkl
import numpy as np
import pandas as pd
import preprocessor as tp
import ftfy
from argparse import ArgumentParser
from pathlib import Path
import logging
import json
import random
from collections import namedtuple
from tempfile import TemporaryDirectory
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForMaskedLM
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from transformers import DistilBertModel, DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def FormatData(filename):
	data_list = []
	data = pd.read_csv(filename, sep = '\t')

	# Clean the "tweet_text" column
	tp.set_options(tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.SMILEY, tp.OPT.RESERVED)
	data["tweet_text"] = data["tweet_text"].apply(lambda x: tp.clean(x))
	data["tweet_text"] = data["tweet_text"].apply(lambda x : ftfy.fix_text(x))
	data["tweet_text"] = data["tweet_text"].str.replace(r'\\n',' ', regex=True) 
	data["tweet_text"] = data["tweet_text"].str.replace(r"\'t", " not")
	data["tweet_text"] = data["tweet_text"].str.strip()
	data["tweet_text"] = data["tweet_text"].str.replace("#","")
	data["tweet_text"] = data["tweet_text"].str.replace("@","")
	tweet_id = data['tweet_id'].to_list()
	image_id = data['image_id'].to_list()
	image_path = data['image'].to_list()
	tweet_text = data['tweet_text'].to_list()
	tweet_text = [str(x) for x in tweet_text]
	image_path = ['./CrisisMMD/' + x for x in image_path]

	label = data['label'].to_list()
	alignment = data['label_text_image'].to_list()
	for a_var in range(len(tweet_id)):
		data_point = {}
		if alignment[a_var] == 'Positive':
			data_point['tweet_id'] = tweet_id[a_var]
			data_point['image_id'] = image_id[a_var]
			data_point['image_path'] = image_path[a_var]
			data_point['tweet_text'] = tweet_text[a_var].lower()
			data_point['label'] = label[a_var] 
			data_list.append(data_point)
	return data_list

# load the original descriptions
folderpath = '../CrisisMMD/'
filenames = ['task_humanitarian_text_img_test.tsv']

# load the captions
with open('../final_captions_all.pkl', 'rb') as f:
	captions = pkl.load(f)

# load the BERT descriptions
with open('../adv_eval_files/ids.txt', 'r') as f:
	ids = f.readlines()
	ids = [x.strip() for x in ids]
with open('../adv_eval_files/keywords..greedy.txt', 'r') as f:
	bert_desc = f.readlines()
	bert_desc = [x.strip() for x in bert_desc]

# load the BERT descriptions with adversarial loss
with open('../adv_eval_files/keywords_adv_0.01..greedy.txt', 'r') as f:
	bert_adv_desc = f.readlines()
	bert_adv_desc = [x.strip() for x in bert_adv_desc]

# most similar image caption's baseline
with open('./most_sim_image_text.pkl', 'rb') as f:
	mismatch_points = pkl.load(f)
mismatch_points_ids = [str(x['tweet_id']).strip() for x in mismatch_points]

with open('./image_kw.pkl', 'rb') as f:
	image_kw = pkl.load(f)
image_kw_ids = [str(x['tweet_id']).strip() for x in image_kw]

with open('./random_urls.pkl', 'rb') as f:
	random_urls = pkl.load(f)
random_urls_ids = [str(x['tweet_id']).strip() for x in random_urls]

with open('./text_kw.pkl', 'rb') as f:
	text_kw = pkl.load(f)
text_kw_ids = [str(x['tweet_id']).strip() for x in text_kw]

with open('./image_text_kw,pkl', 'rb') as f:
	image_text_kw = pkl.load(f)
image_text_kw_ids = [str(x['tweet_id']).strip() for x in image_text_kw]

# load the keyword-infused BERT descriptions with adversarial loss
with open('../adv_eval_files/infused_kw_0.01..greedy.txt', 'r') as f:
	bert_kw_adv_desc = f.readlines()
	bert_kw_adv_desc = [x.strip() for x in bert_kw_adv_desc]

# load the GPT fine-tuned descriptions with loss
with open('prompt_finetuned_test.pkl', 'rb') as f:
	test_gpt_ft_data = pkl.load(f)

# load the GPT normal descriptions with loss
with open('prompt_normal_test.pkl', 'rb') as f:
	test_gpt_data = pkl.load(f)

for a_file in filenames:
	data = FormatData(folderpath + a_file)

# This is where we collate all the generations from all the models (baseline as well proposed)
tweet_id = []
original_only = []
original_and_caption = []
original_and_bert = []
original_and_bert_adv = []
caption_only = []
bert_only = []
bert_adv_only = []
bert_kw_adv_only = []
original_and_gpt_ft = []
gpt_ft_only = []
original_and_gpt = []
gpt_only = []
mismatched_text = []
image_kw_text = []
text_kw_text = []
image_text_kw_text = []
random_url_text = []

for point in data:
	if str(point['tweet_id']) not in ids:
		continue
	if str(point['tweet_id']) not in [str(elt[1]) for elt in test_gpt_ft_data]:
		continue
	if str(point['tweet_id']) not in [str(elt) for elt in mismatch_points_ids]:
		continue
	tweet_id.append(point['tweet_id'])
	original_only.append(point['tweet_text'])

	try:
		first_caption = captions[point['image_id'] + '.jpg'].split('<br>')[0]
	except KeyError:
		first_caption = captions[point['image_id'] + '.png'].split('<br>')[0]
	original_and_caption.append(point['tweet_text'] + ' ' + first_caption)
	caption_only.append(first_caption)

	original_and_bert.append(point['tweet_text'] + ' ' + bert_desc[ids.index(str(point['tweet_id']))])
	bert_only.append(bert_desc[ids.index(str(point['tweet_id']))])

	original_and_bert_adv.append(point['tweet_text'] + ' ' + bert_adv_desc[ids.index(str(point['tweet_id']))])
	bert_adv_only.append(bert_adv_desc[ids.index(str(point['tweet_id']))])
	bert_kw_adv_only.append(bert_kw_adv_desc[ids.index(str(point['tweet_id']))])

	random_url_text.append(random_urls[random_urls_ids.index(str(point['tweet_id']))]['text'])
	mismatched_text.append(mismatch_points[mismatch_points_ids.index(mismatch_points[mismatch_points_ids.index(str(point['tweet_id']))]['matched_id'])]['text'])
	image_kw_text.append(image_kw[image_kw_ids.index(str(point['tweet_id']))]['text'])
	text_kw_text.append(text_kw[text_kw_ids.index(str(point['tweet_id']))]['text'])
	image_text_kw_text.append(image_text_kw[image_text_kw_ids.index(str(point['tweet_id']))]['text'])

	for elt in test_gpt_ft_data:
		if str(elt[1]).strip() == str(point['tweet_id']):
			original_and_gpt_ft.append(point['tweet_text'] + ' ' +elt[0])
			gpt_ft_only.append(elt[0].replace(point['tweet_text'], ""))
			break

	for var in test_gpt_data:
		if str(var[1]).strip() == str(point['tweet_id']):
			original_and_gpt.append(point['tweet_text'] + ' ' + var[0])
			gpt_only.append(var[0].replace(point['tweet_text'], ""))
			break
	if str(point['tweet_id']) == '917868035423080449':
		print(image_text_kw_text[-1])
		print(bert_adv_only[-1])
		print("******")
		print(caption_only[-1])
		sys.exit()

# Prepare fine-tuned text embedding model, this is required for computing embedding-based evaluation metrics
my_tokenizer = DistilBertTokenizer.from_pretrained('../multimodal_model/trained_models', local_files_only = True)
my_bert = DistilBertForSequenceClassification.from_pretrained('../multimodal_model/trained_models', local_files_only = True)

def getBERTEmbedding(sentence):
	text_input = torch.tensor(my_tokenizer.encode(sentence)).unsqueeze(0)
	bert_output = my_bert(text_input)
	bert_embedding = torch.mean(bert_output.hidden_states[-1], 1, True)
	bert_embedding = bert_embedding.detach().numpy()
	bert_embedding = np.reshape(bert_embedding, (bert_embedding.shape[-1],))
	return bert_embedding

print(len(original_only))
print(len(gpt_ft_only))
print(len(original_and_gpt_ft))
print(len(gpt_only))

vectors_for_comparison = {}
text_for_comparison = {}

# compute embeddings and store text in dictionary here
for var in range(len(original_only)):
	text = original_only[var]
	demarcated_text = text.split('. ')

	original = original_only[var]
	caption_text = caption_only[var]
	org_caption = original_and_caption[var]
	bert_text = bert_only[var]
	org_bert = original_and_bert[var]
	bert_adv_text = bert_adv_only[var]
	bert_kw_adv_text = bert_kw_adv_only[var]
	org_bert_adv = original_and_bert_adv[var]

	mismatch_t = mismatched_text[var]
	image_kw_t = image_kw_text[var]
	text_kw_t = text_kw_text[var]
	image_text_kw_t = image_text_kw_text[var]
	random_urls_t = random_url_text[var]
	gpt_ft_text = gpt_ft_only[var]
	org_gpt_ft = original_and_gpt_ft[var]
	gpt_text = gpt_only[var]
	org_gpt = original_and_gpt[var]

	vectors_for_comparison[count] = {'original_text': getBERTEmbedding(original_only[var]),\
									 'caption_text': getBERTEmbedding(caption_text),\
									 'bert_text': getBERTEmbedding(bert_text),\
									 'bert_adv_text': getBERTEmbedding(bert_adv_text),\
									 'bert_kw_adv_text': getBERTEmbedding(bert_kw_adv_text),\
									 'org_caption': getBERTEmbedding(org_caption),\
									 'org_bert': getBERTEmbedding(org_bert),\
									 'org_bert_adv': getBERTEmbedding(org_bert_adv),\
									 'gpt_ft_text': getBERTEmbedding(gpt_ft_text),\
									 'org_gpt_ft': getBERTEmbedding(org_gpt_ft),\
									 'gpt_text': getBERTEmbedding(gpt_text),\
									 'mismatch_t': getBERTEmbedding(mismatch_t),\
									 'image_kw_t': getBERTEmbedding(image_kw_t),\
									 'text_kw_t': getBERTEmbedding(text_kw_t),\
									 'image_text_kw_t': getBERTEmbedding(image_text_kw_t),\
									 'org_gpt': getBERTEmbedding(org_gpt),\
									 'random_url':getBERTEmbedding(random_urls_t)
									 }
	text_for_comparison[count] = {	'tweet_id': tweet_id[var],\
									 'original_text': original_only[var],\
									 'caption_text': caption_text,\
									 'bert_text': bert_text,\
									 'bert_adv_text': bert_adv_text,\
									 'bert_kw_adv_text': bert_kw_adv_text,\
									 'org_caption': org_caption,\
									 'org_bert': org_bert,\
									 'org_bert_adv': org_bert_adv,\
									 'gpt_ft_text': gpt_ft_text,\
									 'org_gpt_ft': org_gpt_ft,\
									 'gpt_text': gpt_text,\
									 'org_gpt': org_gpt,\
									 'mismatch_t': mismatch_t,\
									 'image_kw_t': image_kw_t,\
									 'text_kw_t': text_kw_t,\
									 'image_text_kw_t': image_text_kw_t,\
									 'random_url': random_urls_t
									 }

with open('vectors_for_comparison_with_baselines.pkl', 'wb') as f:
	pkl.dump(vectors_for_comparison, f)

with open('text_for_comparison_with_baselines.pkl', 'wb') as f:
	pkl.dump(text_for_comparison, f)


