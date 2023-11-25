#general purpose packages
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os

#preprocessing
from sklearn import preprocessing

#transformers
from transformers import AutoModel, AutoConfig, XLMRobertaModel, AutoTokenizer, get_linear_schedule_with_warmup

#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

#torch
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

#set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

print('LOADING DATA...')

#load data
train_path = 'data/data-raw/train_raw.csv'
dev_path = 'data/data-raw/dev_raw.csv'
test_path = 'data/data-raw/test_raw.csv'
col_names = ['text', 'label']

train_data = pd.read_csv(train_path, names=col_names, header=None)
dev_data = pd.read_csv(dev_path, names=col_names, header=None)
test_data = pd.read_csv(test_path, names=col_names, header=None)

class_names = ['very neg', 'neg', 'neu', 'pos', 'very pos']

def to_sentiment(label):
    label = str(label)
    if label == 'very pos':
        return 4
    elif label == 'pos':
        return 3
    elif label == 'neu':
        return 2
    elif label == 'neg':
        return 1
    elif label == 'very neg':
        return 0

train_data['label'] = train_data.label.apply(to_sentiment)
dev_data['label'] = dev_data.label.apply(to_sentiment)
test_data['label'] = test_data.label.apply(to_sentiment)

print('FINISHED LOADING DATA')

######################################
### XLM-RoBERTa Sentiment Analysis ###
######################################
DATA_VERSON = 'Raw' #Raw, Pre, Aug
PRE_TRAINED_MODEL_NAME = 'xlm-roberta-base'
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 15
LR = 5e-6

paren_folder = 'data/models/saved_models' 
PATH = os.path.join(paren_folder, PRE_TRAINED_MODEL_NAME)
try: 
    os.mkdir(PATH) 
except OSError as error: 
    print(error)

MODEL_NAME = PRE_TRAINED_MODEL_NAME + '_ml' + str(MAX_LEN) + '_bs' + str(BATCH_SIZE) + '_ep' + str(EPOCHS) + '_lr' + str(LR) + '_data' + DATA_VERSON 

#tokenizer
config = AutoConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#dataloader
class SST_Dataset(Dataset):

  def __init__(self, texts, labels, tokenizer, max_len):
    self.text = texts
    self.targets = labels
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.text)
  
  def __getitem__(self, item):
    text = str(self.text[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      truncation = True,
      return_tensors='pt',
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
  
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = SST_Dataset(
    texts=df.text.to_numpy(),
    labels=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(dev_data, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

#modeling
class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = XLMRobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config = config)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)     

model = SentimentClassifier(len(class_names))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

#training
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

#evaluating
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

#START TRAIN
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(train_data)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(dev_data)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc.item())
  history['train_loss'].append(train_loss.item())
  history['val_acc'].append(val_acc.item())
  history['val_loss'].append(val_loss.item())

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), PATH + '/' + MODEL_NAME + '.bin')
    best_accuracy = val_acc

#testing
def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  val_data_loader
)    

class_names = ['very neg', 'neg', 'neu', 'pos', 'very pos']

clf_rp = classification_report(y_test, y_pred, target_names=class_names)

print(clf_rp)

clf_rp_path = 'data/models/classification_reports/' + MODEL_NAME + '_VAL_classification_report.csv'
with open(clf_rp_path, 'w') as f:
    f.write(clf_rp)

def show_confusion_matrix(confusion_matrix, path):
    plt.figure()
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment')
    plt.savefig(path)

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
clf_rp_path = 'data/models/confusion_matrices/' + MODEL_NAME + '_VAL_confusion_matrice.png'
show_confusion_matrix(df_cm, clf_rp_path)

def show_how_it_learns_acc(history, path):
    plt.figure()
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);
    plt.savefig(path)

def show_how_it_learns_loss(history, path):
    plt.figure()
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')

    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(path)

show_learning_path = 'data/models/history/' + MODEL_NAME + '_acc.png'
show_how_it_learns_acc(history, show_learning_path)
show_learning_path = 'data/models/history/' + MODEL_NAME + '_loss.png'
show_how_it_learns_loss(history, show_learning_path)



y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)    

class_names = ['very neg', 'neg', 'neu', 'pos', 'very pos']

clf_rp = classification_report(y_test, y_pred, target_names=class_names, digits=4)

print(clf_rp)

clf_rp_path = 'data/models/classification_reports/' + MODEL_NAME + '_TEST_classification_report.csv'
with open(clf_rp_path, 'w') as f:
    f.write(clf_rp)

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
clf_rp_path = 'data/models/confusion_matrices/' + MODEL_NAME + '_TEST_confusion_matrice.png'
show_confusion_matrix(df_cm, clf_rp_path)