import pandas as pd
import json

def json_dumper(path, data):
    with open(path, 'w') as fp:
        fp.write(json.dumps(data, indent=4, default=str))

class Data_stats:
    
    def __init__(self, dataset_path):
        self.raw_data = pd.read_csv(dataset_path, header=None)
        self.raw_data.columns = ['text', 'label']

    def count_sents(self, data):
        # code for return number of sentences
        return len(data)

    def count_vocab_tokens(self, data):
        # code for returning vocab and counting vocabs and token counter

        sents = data.text.to_list()

        tokens_counter = 0

        vocab_set = set()
        
        for sent in sents:
            words = sent.split(' ')
            tokens_counter += len(words)
            vocab_set.update(set(words))
        
        return vocab_set, len(vocab_set), tokens_counter

    def count_labels(self, data):
        # code for counting #sents by labels

        labels_count = data.label.value_counts()

        labels_count_dict = {}

        for label in labels_count.index:
            labels_count_dict[label] = labels_count[label]
        
        return labels_count_dict

    def avg_len(self, data):
        # code for calculating average sentence length

        data['sent_len'] = [len(sent.split(' '))for sent in data['text']]

        avg_sent_len = data['sent_len'].mean()

        return avg_sent_len
    
    def stats_pipeline(self):
        """
        Stats pipeline: 
        """

        data_stats = {}

        # 1. count number of sentences
        data_stats['#sents'] = self.count_sents(data = self.raw_data)

        # 2. count number of vocabs, tokens
        _, vocabs, tokens = self.count_vocab_tokens(data = self.raw_data)
        data_stats['#vocabs'] = vocabs
        data_stats['#tokens'] = tokens

        # 3. average sentences length
        data_stats['avg_sent_len'] = self.avg_len(data = self.raw_data)

        # 4. count number of sentences by labels
        data_stats['labels_count_dict'] = self.count_labels(data = self.raw_data)

        return data_stats

train_stats = Data_stats(dataset_path='data/data-raw/train_raw.csv')
dev_stats = Data_stats(dataset_path='data/data-raw/dev_raw.csv')

train_stats_dict = train_stats.stats_pipeline()
dev_stats_dict = dev_stats.stats_pipeline()

json_dumper('data/statistics/data-stats/train-stats.json', train_stats_dict)
json_dumper('data/statistics/data-stats/dev-stats.json', dev_stats_dict)

print('TRAIN STATS:\n')
print(train_stats_dict)
print('\n')
print('DEV STATS:\n')
print(dev_stats_dict)

