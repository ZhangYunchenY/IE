import json
import pickle
from tqdm import tqdm
from transformers import BertForQuestionAnswering

DATA_PATH = './_data/labed_data.json'
DATA_SAVE_PATH = './_data/processed_data.json'
UNLABELED_DATA_PATH = './_data/unlabeled_data_dir/content.json'
UNLABELED_DATA_SAVE_PATH = './_data/unlabeled_content.json'


class Example:
    def __init__(self,
                 document: str,
                 qas: dict,
                 ):
        self.document = document
        self.qas = qas


class Feature:
    def __init__(self,
                 input_ids,
                 attention_masks,
                 token_type_ids,
                 labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


def data_reader_json(data_path, save_path):
    documents = {"documents": []}
    with open(data_path, mode='r') as reader:
        read = reader.readlines()
    for dic in tqdm(read, desc='Reading data'):
        dic = json.loads(dic)
        documents["documents"].append(dic)
    with open(save_path, mode='w') as writer:
        print('===== Writing data =====')
        writer.write(json.dumps(documents, indent=2, ensure_ascii=False))


def pkl_writer(path, features):
    with open(path, mode='wb') as writer:
        pickle.dump(features, writer)


def pkl_reader(path):
    with open(path, mode='rb') as reader:
        feature = pickle.load(reader)
        return feature
