import os, sys, re
import pickle as pkl
import random, string
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

# load image embeddings
train_image_data = np.load('../multimodal_model/img_embs/image_embedding_256_train.npz')
test_image_data = np.load('../multimodal_model/img_embs/image_embedding_256_test.npz')
val_image_data = np.load('../multimodal_model/img_embs/image_embedding_256_dev.npz')

train_img_emb, train_img_id = train_image_data['image_embedding'], train_image_data['image_tweetID']
val_img_emb, val_img_id = val_image_data['image_embedding'], val_image_data['image_tweetID']
test_img_emb, test_img_id = test_image_data['image_embedding'], test_image_data['image_tweetID']

print(len(test_img_id))
print(type(test_img_emb))
print(type(test_img_id))

with open('text_for_comparison.pkl', 'rb') as f:
	data = pkl.load(f)


my_test_ids = []

for key in data:
	my_test_ids.append(int(data[key]['tweet_id']))

print(len(my_test_ids))

# compute cosine similarity between two image embeddings
def cosine_sim(a, b):
	return 1.0 - spatial.distance.cosine(a, b)

# Find the most similar image to the currrent image here
count = 0
matching_id = []
matching_score = []
for i in my_test_ids:
	for j in range(test_img_id.shape[0]):
		if int(test_img_id[j]) == int(i):
			count += 1
			print(count)
			temp_id = []
			temp_score = []
			for j_in in range(test_img_id.shape[0]):
				if int(test_img_id[j_in]) in [int(x) for x in my_test_ids]:
					if int(test_img_id[j_in]) == int(test_img_id[j]):
						continue
					temp_id.append(int(test_img_id[j_in]))
					temp_score.append(cosine_sim(test_img_emb[j_in], test_img_emb[j]))
			index_oi = temp_score.index(max(temp_score))
			matching_id.append(str(temp_id[index_oi]))
			matching_score.append(temp_score[index_oi])

print(len(matching_id))
print(len(matching_score))

# Let's visualize how the image similarities in the corpus are distributed using a histogram
plt.hist(matching_score, density=False, bins=30)  # density=False would make counts
plt.ylabel('Count')
plt.xlabel('Similarity')
plt.show()

import statistics
print(sum(matching_score)/len(matching_score))
print(statistics.stdev(matching_score))

with open('../adv_eval_files/test_id2kw.pkl', 'rb') as f:
	[test_ids, test_kw] = pkl.load(f)
test_kw = [' '.join(x) for x in test_kw]

with open('../adv_eval_files/ids.txt', 'r') as f:
	ids_from_file = f.readlines()
ids_from_file = [x.strip() for x in ids_from_file]

with open('../adv_eval_files/keywords.txt', 'r') as f:
	image_kw_from_file = f.readlines()

# get text with various baselines here: random URLs, image kw, text kw, and image+text kw
random_urls = []
image_kw = []
text_kw = []
image_text_kw = []
for var in range(len(my_test_ids)):
	an_id = my_test_ids[var]
	for elt in data:
		if data[elt]['tweet_id'] == an_id:
			rdm_url = 'https://t.co/' + ''.join(random.choices(string.ascii_letters + string.digits, k=6))
			print(rdm_url)
			curr_text_kw = test_kw[test_ids.index(str(an_id))].strip()
			curr_image_kw = image_kw_from_file[ids_from_file.index(str(an_id))].strip()
			print(curr_image_kw)
			print(curr_text_kw)
			print(curr_image_kw + ' ' + curr_text_kw)
			text_kw.append({'tweet_id': an_id, 'text': curr_text_kw})
			image_kw.append({'tweet_id': an_id, 'text': curr_image_kw})
			image_text_kw.append({'tweet_id': an_id, 'text': (curr_image_kw + ' ' + curr_text_kw).strip()})
			random_urls.append({'tweet_id': an_id, 'text': rdm_url})

# save the generations here
with open('random_urls.pkl', 'wb') as f:
	pkl.dump(random_urls, f)

with open('image_kw.pkl', 'wb') as f:
	pkl.dump(image_kw, f)

with open('text_kw.pkl', 'wb') as f:
	pkl.dump(text_kw, f)

with open('image_text_kw,pkl', 'wb') as f:
	pkl.dump(image_text_kw, f)

