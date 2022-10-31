from argparse import ArgumentParser
from pathlib import Path
import os,sys
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory
import pickle as pkl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import pdb

import torch.nn.functional as F
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForMaskedLM
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from transformers import DistilBertModel, DistilBertConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

#import torch.nn as nn
#import torch

NUM_PAD = 3

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids example_id ")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)

sys.path.append("/pytorch_code")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def convert_example_to_features(example, tokenizer, max_seq_length, args=None):
    tokens = example["tokens"]
    lm_label_tokens = example["lm_label_tokens"]
    example_id_curr = int(example["example_id"]) 

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
        lm_label_tokens = lm_label_tokens[:max_seq_length]

    assert len(tokens) == len(lm_label_tokens) <= max_seq_length  # The preprocessed data should be already truncated
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    lm_label_ids = tokenizer.convert_tokens_to_ids(lm_label_tokens)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[:min(len(lm_label_ids) + NUM_PAD,max_seq_length) ] = 0
    lm_label_array[:len(lm_label_ids)] = lm_label_ids

    if args.wp:
        cls_pos = tokens.index('[CLS]')
        lm_label_array[:cls_pos] = -1

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             example_id=example_id_curr
                             )
    return features



def AlignFormatLabels(label_dict, labels):
    updated_dict = {}
    for tweet_id in label_dict.keys():
        if label_dict[tweet_id] == 'vehicle_damage':
            updated_dict[tweet_id] = 'infrastructure_and_utility_damage'
        elif label_dict[tweet_id] == 'missing_or_found_people' or label_dict[tweet_id] == 'injured_or_dead_people':
            updated_dict[tweet_id] = 'affected_individuals'
        else:
            updated_dict[tweet_id] = label_dict[tweet_id]
    formatted_dict = {}
    for key in updated_dict.keys():
        formatted_dict[np.int64(key)] = updated_dict[key]
    return formatted_dict


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

# id to original tweet text
with open('./multimodal_model/task_humanitarian_text_img_train.pkl', 'rb') as f:
   temp_data = pkl.load(f)
id2text = {}
for elt in temp_data:
    if np.int64(int(elt['tweet_id'])) not in id2text:
        id2text[np.int64(int(elt['tweet_id']))] = elt['tweet_text']

# load image embeddings
train_image_data = np.load('./multimodal_model/img_embs/image_embedding_256_train.npz')
test_image_data = np.load('./multimodal_model/img_embs/image_embedding_256_test.npz')
val_image_data = np.load('./multimodal_model/img_embs/image_embedding_256_dev.npz')

train_img_emb, train_img_id = train_image_data['image_embedding'], train_image_data['image_tweetID']
val_img_emb, val_img_id = val_image_data['image_embedding'], val_image_data['image_tweetID']
test_img_emb, test_img_id = test_image_data['image_embedding'], test_image_data['image_tweetID']

# load labels
pickle_folder_path = './multimodal_model/'
id2label_dict = GetLabels(pickle_folder_path)
my_labels = ['affected_individuals', 'infrastructure_and_utility_damage', 'not_humanitarian', 'other_relevant_information', 'rescue_volunteering_or_donation_effort']

all_image_ids = []
for tweet_id in train_img_id:
    all_image_ids.append(tweet_id)
all_image_ids = list(set(all_image_ids))

print(type(tweet_id))
print(tweet_id)
print(len(all_image_ids))

img_embeddings = np.zeros((len(all_image_ids), train_img_emb.shape[1]))
class_labels = np.zeros((len(all_image_ids),))

#print(len(list(id2label_dict.keys())))
#print(id2label_dict)

print(type(list(id2label_dict.keys())[0]))
print(list(id2label_dict.keys())[0])
print(len(list(id2label_dict.keys())))

for var in range(len(all_image_ids)):
    tweet_id = all_image_ids[var]
    #print(tweet_id)
    #print(type(tweet_id))
    #print(type(list(id2label_dict.keys())[0]))
    #print(list(id2label_dict.keys())[0])
    for elt in range(train_img_emb.shape[0]):
        if train_img_id[elt] == tweet_id:
            img_embeddings[var, :] = train_img_emb[elt]
    try:
        class_labels[var] = my_labels.index(id2label_dict[tweet_id])
    except KeyError:
        continue

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, args=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            example_ids = np.memmap(filenames=self.working_dir/'example_ids.memmap',
                                    shape=(num_samples,), mode='w+', dtype=np.int32)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            example_ids = np.empty(shape=(num_samples,), dtype=np.int64)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len, args=args)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                example_ids[i] = features.example_id
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.example_ids = example_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                self.example_ids[item].astype(np.int64),
                )

# The architecture of the multimodal classifier
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
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

def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    # For quicker experimentation, and if working with CPU-only machines you may want to use DistillBERT models. 
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--wp", type=bool, default=False, help="if train on wp")
    parser.add_argument('--from_scratch', action='store_true', help='do not load prtrain model, only random initialize')
    parser.add_argument("--output_step", type=int, default=100000, help="Number of step to save model")

    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"


    samples_per_epoch = []
    num_data_epochs = args.epochs
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    args.output_mode = "classification"

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
  
    while True:
        try:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
            if tokenizer._noi_token is None:
                tokenizer._noi_token = '[NOI]'
                if args.bert_model == 'bert-base-uncased' or 'bert-large-uncased' :
                    tokenizer.vocab['[NOI]'] = tokenizer.vocab.pop('[unused0]')
                elif args.bert_model == 'bert-base-cased':
                    tokenizer.vocab['[NOI]'] = tokenizer.vocab.pop('[unused1]')
                else:
                    raise ValueError("No clear choice for insert NOI for tokenizer type {}".format(args.model_name_or_path))
                tokenizer.ids_to_tokens[1] = '[NOI]'
                logger.info("Adding [NOI] to the vocabulary 1")
        except:
            continue
        break
    

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.from_scratch:
        model = BertForMaskedLM()
    else:
        model = BertForMaskedLM.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Prepare fine-tuned text embedding model
    my_tokenizer = DistilBertTokenizer.from_pretrained('./multimodal_model/trained_bert_classifier', local_files_only = True)
    my_bert = DistilBertForSequenceClassification.from_pretrained('./multimodal_model/trained_bert_classifier', local_files_only = True)
    #my_tokenizer.to(device)
    my_bert.to(device)

    # Prepare multimodal classification model
    mlp = MultiLayerPerceptron()
    mlp.load_state_dict(torch.load('./multimodal_model/multimodal_classifier.pth'))
    mlp.eval()
    #mlp.to(device)
    #classification_loss_function = nn.CrossEntropyLoss()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in range(args.epochs):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory, args=args)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            print(step)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, example_ids = batch
            outputs = model(input_ids, segment_ids, input_mask, lm_label_ids,)
            loss = outputs[0]

            ## updated mis-classification loss computation begins here
            logits = outputs[1]
            predictions = logits.argmax(dim = 2)
            for row in range(prediction.size(dim=0)):
                example_id = example_ids[row].detach().cpu().numpy()

                ## get the emmbedding for text using the trained bert classifier (fine-tuned BERT)
                ## note that the tokenizers of the text-only BERT classifier and the POINTER model being fine-tuned have to be the same
                ## based on the encoders that you use for the BERT classifier, you may have to update some of the code below
                predictions = predictions.unsqueeze(0).to(device)
                bert_output = my_bert(predictions)
                bert_embedding = torch.mean(bert_output.hidden_states[-1], 1, True)

                ## get the image embedding and concatenate the text and image embeddings
                image_embedding = img_embeddings[all_image_ids.index(example_id)]
                concatenated_embedding = torch.cat((image_embedding, bert_embedding), dim = 1)
                curr_class = torch.tensor(class_labels[all_image_ids.index(example_id)], dtype=torch.long)

                ## pass the concatenation to mlp and evaluate the misclassification loss
                ## different formulations of the adversarial loss are possible
                ## it's possible to sum the probability scores of all incorrect classes, 
                ## or consider the probability score of the secong largest probabily score
                ##  the following formulation is the former one
                curr_pred = mlp(concatenated_embedding)
                # sorted_logits, _ = torch.sort(curr_pred, dim = -1, descending = True) # this step is useful for the latter formulation
                predicted_class = np.argmax(curr_pred)
                if curr_class == predicted_class:
                    classification_loss = F.cross_entropy(torch.Tensor([1.0 - curr_pred[predicted_class], curr_pred[predicted_class]]).view(1, -1), torch.tensor([0], dtype=torch.long).view(1,))
                elif curr_class != predicted_class:
                    classification_loss = torch.tensor(0.0)
                loss = loss + 0.01*classification_loss

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                #scheduler.step()  # Update learning rate schedule
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.output_step == 0 and args.local_rank in [-1, 0]:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

        if args.local_rank in [-1, 0]:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
        logger.info("PROGRESS: {}%".format(round(100 * (epoch + 1) / args.epochs, 4)))
        logger.info("EVALERR: {}%".format(tr_loss))


    # Save a trained model
    if  args.local_rank == -1 or torch.distributed.get_rank() == 0 :
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
