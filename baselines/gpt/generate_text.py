import os, sys, re
import pickle as pkl
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

filenames = ['test']

for a_file in filenames:
	with open('../data/crisis_' + a_file + '_id.txt', 'r') as f:
		data = f.readlines()
		data = [x for x in data if x.strip() != '']
		print(a_file)
		print(len(data))
		data = [[x.split('\t')[0].strip(), x.split('\t')[1].strip()] for x in data]

		gpt_full_text = []
		# work with the GPT model here
		for entry in data:
			#print(entry)
			generation = generator(entry[0])[0]['generated_text'].replace("\n", "")
			generation = generation.replace(entry[0], "")
			generation = generation.strip().replace("\\", "")
			print(generation)
			gpt_full_text.append([generation, entry[1]])


		with open('./prompt_normal_' + a_file + '.pkl', 'wb') as f:
			pkl.dump(gpt_full_text, f)
