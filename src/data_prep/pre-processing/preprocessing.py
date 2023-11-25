import pandas as pd
import nltk, string, json

with open('data/nltk_data/corpora/stopwords/english') as f:
    STOPWORDS = f.read().splitlines() 

def json_dumper(path, data):
    with open(path, 'w') as fp:
        fp.write(json.dumps(data, indent=4, default=str))

class Preprocessor:
    
    def __init__(self, dataset_path):
        self.raw_data = pd.read_csv(dataset_path, header=None)
        self.raw_data.columns = ['text', 'label']
        self.lem_counter = 0
        self.sw_counter = 0

    def lower_case(self, data):
        # code for preprocessing
        df = data.copy()
        df['text'] = df['text'].str.lower()
        return df
    
    def lemmatizer(self, data):
        #todo do pos for sentence before do lemmatizer
    
        # code for lemmatizing
        lemmatizer = nltk.stem.WordNetLemmatizer()

        def lemmatize_words(text):
            sent = []
            for word in str(text).split():
                n_word = lemmatizer.lemmatize(word)
                if n_word != word:
                    self.lem_counter += 1
                sent.append(n_word)
            return " ".join(sent)
        
        df = data.copy()
        df['text'] = df['text'].apply(lambda text: lemmatize_words(text))

        return df, self.lem_counter
    
    def stopwords_remover(self, data):
        """code for removing stopwords and counting them in the data"""

        def remove_stopwords(text):
            sent = []
            for word in str(text).split():
                if word not in STOPWORDS:
                    sent.append(word)
                else:
                    self.sw_counter += 1
            return " ".join(sent)
        
        df = data.copy()
        df['text'] = df['text'].apply(lambda text: remove_stopwords(text))

        return df, self.sw_counter
    
    # def punc_remover(self, data):
    #     PUNCT_TO_REMOVE = string.punctuation

    #     def remove_punctuation(text):
    #         """custom function to remove the punctuation"""
    #         return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    #     data["text"] = data["text"].apply(lambda text: remove_punctuation(text))


    def preprocess_pipeline_full(self, data_path, stats_path=None, do_stat = True):
        """lowercase -> Lemmatizer -> rm stopwords -> rm punctuation"""

        # 1. lowercase
        data = self.lower_case(data=self.raw_data)

        # 2. lemmatizer
        data, lem_count = self.lemmatizer(data=data)

        # 3. remove stopwords
        data, sw_count = self.stopwords_remover(data=data)

        # # 4. remove punctuation
        # data = self.punc_remover(data=data)
        # print(data)

        if do_stat:
            stats = {
                'Lemmatized words': lem_count,
                'Stopwords removed': sw_count
            }

            json_dumper(path = stats_path, data=stats)

        data.to_csv(data_path, index = False, header= False)
    
    
dev_processor = Preprocessor(dataset_path='data/data-raw/dev_raw.csv')
train_processor = Preprocessor(dataset_path='data/data-raw/train_raw.csv')
test_processor = Preprocessor(dataset_path='data/data-raw/test_raw.csv')

train_processor.preprocess_pipeline_full(data_path='data/data-processed/train-preprocessed.csv',
                                       stats_path='data/statistics/preprocessing-stats/train-preproces-stats.json')

dev_processor.preprocess_pipeline_full(data_path='data/data-processed/dev-preprocessed.csv',
                                       stats_path='data/statistics/preprocessing-stats/dev-preproces-stats.json')

test_processor.preprocess_pipeline_full(data_path='data/data-processed/test-preprocessed.csv',
                                       do_stat=False)
