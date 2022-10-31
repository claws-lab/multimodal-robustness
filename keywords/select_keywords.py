import os, sys, re
import json
from tqdm import tqdm
import pickle as pkl

# Specify the path of the  output of the Scene Graph Generator
K_size = 20 # maximum number of objects from an image
bbox_score_threshold= 0.1
rel_score_threshold= 0.3
directory = './output_sg_004/'

def load_data(foldername):
	# Load the metainfo
	filenames = []
	classes = []
	predicates = []
	print("[status]	metainfo file is being loaded...")
	with open(foldername + 'custom_data_info.json', 'r') as f:
		metainfo_dict = json.loads(f.read())
	for key in metainfo_dict:
		if key == 'idx_to_files':
			for filename in metainfo_dict[key]:
				filenames.append(filename)
		elif key == 'ind_to_classes':
			for a_class in metainfo_dict[key]:
				classes.append(a_class)
		elif key == 'ind_to_predicates':
			for predicate in metainfo_dict[key]:
				predicates.append(predicate)
	print("[info]	number of files: ", len(filenames))
	print("[info]	number of classes: ", len(classes))
	print("[info]	number of predicates: ", len(predicates))
	print("[status]	metainfo file has been loaded")

	# print(filenames[0])
	# Load the predictions file
	print("[status]	predictions file is being loaded. this will take some time...")
	with open(foldername + 'custom_prediction.json', 'r') as f:
		predictions_dict = json.loads(f.read())

	final_keywords = {}
	for key in tqdm(predictions_dict):

		# Get all the bboxes with score greater than pre-specified threshold
		bbox_scores = predictions_dict[key]['bbox_scores']
		indices_top_K_score = []
		for var in range(len(bbox_scores)):
			if bbox_scores[var] >= bbox_score_threshold:
				indices_top_K_score.append(var)

		# Get top K based on bounding box sizes
		bbox_size = []
		for coords in predictions_dict[key]['bbox']:
			size = abs(coords[0] - coords[2]) * abs(coords[1] - coords[3])
			bbox_size.append(size)
		indices_top_K_size = sorted(range(len(bbox_size)), key=lambda x: bbox_size[x])[-K_size:]
		
		# Find intersection of both top Ks
		intersecting_indices = list(set(indices_top_K_size) & set(indices_top_K_score))
		final_objs = []
		final_objs_scores = []
		for index in intersecting_indices:
			if predictions_dict[key]['bbox_labels'][index] not in final_objs:
				final_objs.append(predictions_dict[key]['bbox_labels'][index])
				final_objs_scores.append(predictions_dict[key]['bbox_scores'][index])
		
		final_rel_pairs = []
		final_rel_labels = []
		final_rel_scores = []
		for var in range(len(predictions_dict[key]['rel_pairs'])):
			pair = predictions_dict[key]['rel_pairs'][var]
			label = predictions_dict[key]['rel_labels'][var]
			score = predictions_dict[key]['rel_scores'][var]
			if predictions_dict[key]['rel_scores'][var] >= rel_score_threshold:
				if pair[0] in final_objs and pair[1] in final_objs:
					final_rel_pairs.append(pair)
					final_rel_labels.append(label)
					final_rel_scores.append(score)

		# convert indices to words:
		final_objs = [classes[x] for x in final_objs]
		final_rels = []
		rels_scores = []
		for val in range(len(final_rel_pairs)):
			obj1 = '(' + classes[final_rel_pairs[val][0]] + ')'
			obj2 = '(' + classes[final_rel_pairs[val][1]] + ')'
			rel = ' ' + predicates[final_rel_labels[val]] + ' '
			if obj2 + rel + obj1 in final_rels:
				continue
			final_rels.append(obj1 + rel + obj2)
			rels_scores.append(final_rel_scores[val])


		final_keywords[filenames[int(key)]] = {'objs': final_objs, 'objs_scores': final_objs_scores, 'rels': final_rels, 'rels_scores': rels_scores}
		#print(filenames[int(key)])
		#print(final_keywords[filenames[int(key)]])
		#print("____________")
	
	return final_keywords


final_keywords = load_data(directory)

with open(directory + 'final_keywords.pkl', 'wb') as f:
	pkl.dump(final_keywords, f)
