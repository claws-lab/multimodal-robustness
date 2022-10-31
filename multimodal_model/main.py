import os, sys, re
import pickle as pkl
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import torch

NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 16

def AlignFormatLabels(label_dict, labels):
	updated_dict = {}
	for tweet_id in label_dict.keys():
		if label_dict[tweet_id] == 'vehicle_damage':
			updated_dict[tweet_id] = 'infrastructure_and_utility_damage'
		elif label_dict[tweet_id] == 'missing_or_found_people' or label_dict[tweet_id] == 'injured_or_dead_people':
			updated_dict[tweet_id] = 'affected_individuals'
		else:
			updated_dict[tweet_id] = label_dict[tweet_id]
	return updated_dict

def GetLabels(folder):
	filenames_prefix = 'task_humanitarian_text_img_'
	with open(folder + filenames_prefix + 'train.pkl', 'rb') as f:
		train_data = pkl.load(f)
	with open(folder + filenames_prefix + 'dev.pkl', 'rb') as f:
		val_data = pkl.load(f)
	with open(folder + filenames_prefix + 'test.pkl', 'rb') as f:
		test_data = pkl.load(f)

	id2label = {}
	for a_point in train_data:
		id2label[a_point['tweet_id']] = a_point['label']
	for a_point in val_data:
		id2label[a_point['tweet_id']] = a_point['label']
	for a_point in test_data:
		id2label[a_point['tweet_id']] = a_point['label']
	my_labels = ['affected_individuals', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort']

	id2label = AlignFormatLabels(id2label, my_labels)
	return id2label

class DatasetFormatting(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

def compute_metrics(preds, labels):
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
	acc = accuracy_score(labels, preds)
	return {
		'accuracy': acc,
		'f1': f1,
		'precision': precision,
		'recall': recall
	}

# load image embeddings
train_image_data = np.load('./img_embs/image_embedding_256_train.npz')
test_image_data = np.load('./img_embs/image_embedding_256_test.npz')
val_image_data = np.load('./img_embs/image_embedding_256_dev.npz')

train_img_emb, train_img_id = train_image_data['image_embedding'], train_image_data['image_tweetID']
val_img_emb, val_img_id = val_image_data['image_embedding'], val_image_data['image_tweetID']
test_img_emb, test_img_id = test_image_data['image_embedding'], test_image_data['image_tweetID']

# load non-English language embeddings
LANGUAGE = 'eng'

print("[INFO]: Working for language: " + LANGUAGE)

# load English language embeddings
with open('./english/embeddings_train.pickle', 'rb') as handle:
    embeddings_train_dict = pkl.load(handle)
with open('./english/embeddings_val.pickle', 'rb') as handle:
    embeddings_val_dict = pkl.load(handle)
with open('./english/embeddings_test.pickle', 'rb') as handle:
    embeddings_test_dict = pkl.load(handle)

train_text_emb, train_text_id = [], []
for tweet_id in embeddings_train_dict.keys():
	train_text_id.append(tweet_id)
	train_text_emb.append(embeddings_train_dict[tweet_id])

test_text_emb, test_text_id = [], []
for tweet_id in embeddings_test_dict.keys():
	test_text_id.append(tweet_id)
	test_text_emb.append(embeddings_test_dict[tweet_id])

val_text_emb, val_text_id = [], []
for tweet_id in embeddings_val_dict.keys():
	val_text_id.append(tweet_id)
	val_text_emb.append(embeddings_val_dict[tweet_id])

train_text_emb = np.array(train_text_emb)
train_text_emb = train_text_emb.reshape((train_text_emb.shape[0], train_text_emb.shape[-1]))
test_text_emb = np.array(test_text_emb)
test_text_emb = test_text_emb.reshape((test_text_emb.shape[0], test_text_emb.shape[-1]))
val_text_emb = np.array(val_text_emb)
val_text_emb = val_text_emb.reshape((val_text_emb.shape[0], val_text_emb.shape[-1]))


# load labels
pickle_folder_path = './'
id2label_dict = GetLabels(pickle_folder_path)
my_labels = ['affected_individuals', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort']

train_ids = []
for tweet_id in train_img_id:
	if tweet_id in train_text_id:
		train_ids.append(tweet_id)
test_ids = []
for tweet_id in test_img_id:
	if tweet_id in test_text_id:
		test_ids.append(tweet_id)
val_ids = []
for tweet_id in val_img_id:
	if tweet_id in val_text_id:
		val_ids.append(tweet_id)

train_ids = list(set(train_ids))
val_ids = list(set(val_ids))
test_ids = list(set(test_ids))
print("length train set: ", len(train_ids))
print("length val set: ", len(val_ids))
print("length test set: ", len(test_ids))

# create train, val, and test data
train_x = np.zeros((len(train_ids), train_img_emb.shape[1] + train_text_emb.shape[1]))
train_y = np.zeros((len(train_ids),))
val_x = np.zeros((len(val_ids), train_img_emb.shape[1] + train_text_emb.shape[1]))
val_y = np.zeros((len(val_ids),))
test_x = np.zeros((len(test_ids), train_img_emb.shape[1] + train_text_emb.shape[1]))
test_y = np.zeros((len(test_ids),))

for var in range(len(train_ids)):
	tweet_id = train_ids[var]
	for elt in range(train_img_emb.shape[0]):
		if train_img_id[elt] == tweet_id:
			train_x[var, : train_img_emb.shape[1]] = train_img_emb[elt]
	for elt in range(train_text_emb.shape[0]):
		if train_text_id[elt] == tweet_id:
			train_x[var, train_img_emb.shape[1]:] = train_text_emb[elt]
	train_y[var] = my_labels.index(id2label_dict[tweet_id])

for var in range(len(val_ids)):
	tweet_id = val_ids[var]
	for elt in range(val_img_emb.shape[0]):
		if val_img_id[elt] == tweet_id:
			val_x[var, : val_img_emb.shape[1]] = val_img_emb[elt]
	for elt in range(val_text_emb.shape[0]):
		if val_text_id[elt] == tweet_id:
			val_x[var, val_img_emb.shape[1]:] = val_text_emb[elt]
	val_y[var] = my_labels.index(id2label_dict[tweet_id])

for var in range(len(test_ids)):
	tweet_id = test_ids[var]
	for elt in range(test_img_emb.shape[0]):
		if test_img_id[elt] == tweet_id:
			test_x[var, : test_img_emb.shape[1]] = test_img_emb[elt]
	for elt in range(test_text_emb.shape[0]):
		if test_text_id[elt] == tweet_id:
			test_x[var, test_img_emb.shape[1]:] = test_text_emb[elt]
	test_y[var] = my_labels.index(id2label_dict[tweet_id])

train_y = np.array(train_y)
val_y = np.array(val_y)
test_y = np.array(test_y)

train_x, train_y = torch.Tensor(train_x), torch.tensor(train_y, dtype=torch.long)
val_x, val_y = torch.Tensor(val_x), torch.Tensor(val_y)
test_x, test_y = torch.Tensor(test_x), torch.Tensor(test_y)

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)

INPUT_SIZE = train_img_emb.shape[1] + train_text_emb.shape[1]


## Define the NN architecture
class MultiLayerPerceptron(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(INPUT_SIZE, 512),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(32, 5)
		)
	def forward(self, x):
		return self.layers(x)

mlp = MultiLayerPerceptron()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

if __name__ == "__main__":
	for epoch in range(NUM_EPOCHS):
		print(f'Starting epoch {epoch+1}')
		current_loss = 0.0
		for i, data in enumerate(trainloader):
			inputs, targets = data
			optimizer.zero_grad()
			outputs = mlp(inputs)
			loss = loss_function(outputs, targets)
			loss.backward()
			optimizer.step()
			current_loss += loss.item()
			if i % 500 == 499:
				print('Training loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
				current_loss = 0.0

	PATH = './multimodal_classifier.pth'
	torch.save(mlp.state_dict(), PATH)

	mlp = MultiLayerPerceptron()
	mlp.load_state_dict(torch.load('multimodal_classifier.pth'))
	mlp.eval()
	with torch.no_grad():
		print("Validation set results")
		ground_truth = []
		predictions = []
		for var in range(val_x.shape[0]):
			data, true_label = val_dataset[var]
			ground_truth.append(true_label.item())
			curr_pred =  np.argmax(mlp(data)).item()
			predictions.append(curr_pred)
		print(compute_metrics(predictions, ground_truth))

		# Evaluation on test set
		print("Test set results")
		ground_truth = []
		predictions = []
		for var in range(test_x.shape[0]):
			data, true_label = test_dataset[var]
			ground_truth.append(true_label.item())
			curr_pred =  np.argmax(mlp(data)).item()
			predictions.append(curr_pred)
		print(compute_metrics(predictions, ground_truth))

# precision, recall, f1, _ = precision_recall_fscore_support(test_y, predicted_labels, average='macro')
# acc = accuracy_score(test_y, predicted_labels)

# print('accuracy', acc)
# print('f1', f1)
# print('precision', precision)
# print('recall', recall)


