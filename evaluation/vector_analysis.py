import os, sys, re
import pickle as pkl
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy import stats

with open('vectors_for_comparison_with_dumb.pkl', 'rb') as f:
	vectors = pkl.load(f)

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h

def plotHistogram(list1, list2, variable1, variable2):
	similarities = []
	assert len(list1) == len(list2)
	print(len(list1))
	for var in range(len(list1)):
		cos_sim = np.dot(list1[var], list2[var])/(np.linalg.norm(list1[var])*np.linalg.norm(list2[var]))
		similarities.append(cos_sim)
	similarities = np.array(similarities)
	mean, low_ci, high_ci = mean_confidence_interval(similarities)
	stdev = np.std(similarities)
	plt.hist(similarities, density=False, bins=20)
	plt.ylabel('count')
	plt.xlabel('Similarity')
	plt.title("Similarity between " + variable1 + ' & ' + variable2 + "\n Mean: %.2f; STDEV: %.2f (95 pct CI: %.2f, %.2f)" % (mean, stdev, low_ci, high_ci))
	plt.show()
	return similarities

variable1, variable2 = 'original_text', 'gpt_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'gpt_ft_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'caption_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'bert_adv_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'bert_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'bert_kw_adv_text'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'mismatch_t'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'image_kw_t'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'text_kw_t'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'image_text_kw_t'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

variable1, variable2 = 'original_text', 'random_url'
list1, list2 = [], []
for count in vectors:
	list1.append(vectors[count][variable1])
	list2.append(vectors[count][variable2])

sim_second_caption = plotHistogram(list1, list2, variable1, variable2)

all_text_id = ['original_text',
'first_sentence',
'second_sentence',
'caption_text',
'bert_text',
'bert_adv_text',
'bert_kw_adv_text',
'org_caption',
'org_bert',
'org_bert_adv',
'gpt_ft_text',
'org_gpt_ft',
'gpt_text',
'org_gpt',
'mismatch_t',
'text_kw_t',
'image_kw_t',
'image_text_kw_t',
'random_url'
]

all_text = [[] for x in all_text_id]
with open('text_for_comparison_with_dumb.pkl', 'rb') as f: 
	text_data = pkl.load(f)

print(text_data[list(text_data.keys())[0]])

for key in text_data.keys():
	for subkey in text_data[key]:
		if subkey == 'tweet_id':
			continue
		print(subkey)
		all_text[all_text_id.index(subkey)].append(text_data[key][subkey])
	# sys.exit()

num_words = [[] for x in all_text_id]
num_new_words = [[] for x in all_text_id]

for curr_id in all_text_id:
	for point in all_text[all_text_id.index(curr_id)]:
		num_words[all_text_id.index(curr_id)].append(len(point.split(' ')))

for curr_id in all_text_id:
	for var in range(len(all_text[all_text_id.index(curr_id)])):
		point = all_text[all_text_id.index(curr_id)][var]
		num_new_words[all_text_id.index(curr_id)].append(len(list(set(point.split(' ')).difference(set(all_text[all_text_id.index('original_text')][var])))))

count = 0
for curr_list in num_new_words:
	print("****\n")
	print(all_text_id[count])
	count += 1
	curr_list = np.array(curr_list)
	print("mean: ", np.mean(curr_list))
	print("median: ", np.median(curr_list))
	print("stdev: ", np.std(curr_list))
