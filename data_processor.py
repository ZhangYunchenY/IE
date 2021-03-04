import json
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

BIO_DICT = {"B": 0, "I": 1, "O": 2}
DATA_PATH = './_data/labed_data.json'
DATA_SAVE_PATH = './_data/processed_data.json'
UNLABELED_DATA_PATH = './_data/unlabeled_data_dir/content.json'
UNLABELED_DATA_SAVE_PATH = './_data/unlabeled_content.json'


class Example:
    def __init__(self,
                 key: str,
                 question: str,
                 content: str,
                 label: list
                 ):
        self.key = key
        self.question = question
        self.content = content
        self.label = label


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


def format_json(data_path, save_path):
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


def conver_answer_to_label(content, answer):
    assert answer in content
    label = [BIO_DICT["O"] for index in range(len(content))]
    start_index = content.find(answer)
    end_index = start_index + len(answer) - 1
    # 如果要从content中获取字串 则需要 content[start_index: end_index+1]
    label[start_index: end_index + 1] = BIO_DICT["I"]
    label[start_index] = BIO_DICT["B"]
    return label


def read_examples(examples_path):
    examples = []
    with open(examples_path, mode='r') as reader:
        read = reader.readline()
        examples_dict = json.loads(read)
    for item in tqdm(examples_dict["documents"], desc='Processing examples'):
        content = ""
        for sentence in item["document"]:
            content += sentence["text"]
        for question_answers_dict in item["qas"][0]:
            for answer_dict in question_answers_dict["answers"]:
                answer = answer_dict["text"]
                question = question_answers_dict["question"]
                key = item["key"]
                label = conver_answer_to_label(content, answer)
                example = Example(key=key, question=question, content=content, label=label)
                examples.append(example)
    return examples


def split_examples(examples):
    train_examples, dev_examples = train_test_split(examples, shuffle=True, random_state=626, test_size=0.1)
    return train_examples, dev_examples


def convert_examples_to_features(examples: Example, model_name):
    questions, content, labels = [], [], []
    for example in tqdm(examples, desc='Reading examples'):
        questions.append(example.question)
        content.append(example.content)
        labels.append(example.label)
    print("===== Encoding... =====")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    encoded = tokenizer(questions, content, truncation=True, padding=True)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    features = Feature(input_ids=input_ids, attention_masks=attention_mask,
                       token_type_ids=token_type_ids, labels=labels)
    return features


def create_dataloader(features, batch_size):
    input_ids = torch.tensor(features.input_ids)
    attention_masks = torch.tensor(features.attention_masks)
    token_type_ids = torch.tensor(features.token_type_ids)
    labels = torch.tensor(features.labels)
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
