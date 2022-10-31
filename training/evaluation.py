import os, sys, re
import pickle as pkl

subsets = ['001/', '002/', '003/', '004/']
directory = '../SceneGraph/output_sg_'


# get the keywords we care about:
kw_of_concern = []
with open('./multimodal_model/task_humanitarian_text_img_dev.pkl', 'rb') as f:
	data = pkl.load(f)
	for elt in data:
		kw_of_concern.append(elt['tweet_id'])
with open('./multimodal_model/task_humanitarian_text_img_train.pkl', 'rb') as f:
	data = pkl.load(f)
	for elt in data:
		kw_of_concern.append(elt['tweet_id'])
with open('./multimodal_model/task_humanitarian_text_img_test.pkl', 'rb') as f:
	data = pkl.load(f)
	for elt in data:
		kw_of_concern.append(elt['tweet_id'])
kw_of_concern = list(set(kw_of_concern))
kw_of_concern = [str(x) for x in kw_of_concern]
print(kw_of_concern)
print(len(kw_of_concern))


all_keywords = {}
for subset in subsets:
	with open(directory + subset + 'final_keywords.pkl', 'rb') as f:
		curr_data = pkl.load(f)
	for id in curr_data.keys():
		list_objs = curr_data[id]['objs']
		tweet_id = id.split('/')[-1].split('_')[0]
		if tweet_id not in kw_of_concern:
			continue
		if tweet_id not in all_keywords:
			all_keywords[tweet_id] = list_objs

with open('./evaluation/ids.txt', 'w') as id_file:
	with open('./evaluation/keywords.txt', 'w') as kw_file:
		for key in all_keywords.keys():
			id_file.write(key + '\n')
			kw_file.write(' '.join(all_keywords[key]) + '\n')

