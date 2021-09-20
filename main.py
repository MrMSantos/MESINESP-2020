import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification, AdamW

class Dataset(torch.utils.data.Dataset):

	def __init__(self, encodings, labels):

		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):

		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx], dtype = torch.float32)
		return item

	def __len__(self):
		
		return len(self.labels)

#BERT Model built with attention over label embeddings and a linear layer on top
class CustomBERTModel(nn.Module):

	def __init__(self, num_labels):

		super().__init__()
		self.bert = BertModel.from_pretrained('./beto')
		self.num_labels = num_labels
		#New layers
		self.dropout = nn.Dropout(0.1)
		self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

	def forward(self, input_ids, attention_mask, label_embeddings, labels = None):

		outputs = self.bert(input_ids, attention_mask = attention_mask)

		pooled_output = outputs[1]
		pooled_output = self.dropout(pooled_output)

		#Attention over labels
		value_matmul = torch.matmul(pooled_output, torch.t(label_embeddings))
		value_weights = nn.functional.softmax(value_matmul, dim = 1)
		attention_weights = torch.matmul(value_weights, label_embeddings)

		pooled_output = attention_weights + pooled_output
		logits = self.classifier(pooled_output)

		loss = None
		if labels != None:
			loss_function = nn.BCEWithLogitsLoss()
			loss = loss_function(logits, labels)

		return loss, logits

#Load dataset into pandas dataframe
train_df = pd.read_json('./MESINESP_PREPROCESSED_TRAINING.json')

#Read the first entries
print(train_df.head())

#Checking for Missing Values
print('Title missing values:', pd.isna(train_df.title).sum())
print('Journal missing values:', pd.isna(train_df.journal).sum())
print('Abstract missing values:', pd.isna(train_df.abstractText).sum())

#Get labels from labels file
labels_df = pd.read_csv('./DeCS.2019.both.v5.tsv', sep = '\t')
labels_list = labels_df.Term_Spanish.tolist()
unique_labels = set([label for label_list in labels_list for label in label_list])
num_labels = len(unique_labels)

x_train = train_df.abstractText.tolist()

#Coding labels into one-hot vectors
y_train = train_df.decsCodes
mlb = MultiLabelBinarizer(sparse_output = False)
mlb.fit(unique_labels)
train_labels = mlb.transform(y_train)

#Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('./beto')

#Spanish BETO model was used to get correct embeddings for the language
#model = BertForSequenceClassification.from_pretrained('./beto', num_labels = num_labels)
model = CustomBERTModel(num_labels = num_labels)

#Load model and tensors to GPU 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_encodings = tokenizer(x_train, truncation = True, padding = True, max_length = 128)

train_dataset = Dataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

#Load label embeddings for BERT with attention over labels
with open('list_label_embeddings.sav', 'rb') as handle:
	label_embeddings = torch.as_tensor(pickle.load(handle)).to(device)

model.to(device)
model.train()

optim = AdamW(model.parameters(), lr = 2e-5)
num_epochs = 2

for epoch in range(num_epochs):
	with tqdm(train_loader, unit = 'batch') as tepoch:
		for batch in tepoch:
			optim.zero_grad()
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)
			outputs = model(input_ids, attention_mask = attention_mask, label_embeddings = label_embeddings, labels = labels)
			loss = outputs[0]
			loss.backward()
			optim.step()

model.eval()
