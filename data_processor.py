import json
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
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
                 answer: str,
                 label=None
                 ):
        self.key = key
        self.question = question
        self.content = content
        self.answer = answer
        self.label = label


class Feature:
    def __init__(self,
                 input_ids,
                 attention_masks,
                 token_type_ids,
                 labels,
                 head_mask=None
                 ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.head_mask = head_mask
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


def pkl_writer(features, path):
    with open(path, mode='wb') as writer:
        pickle.dump(features, writer)


def pkl_reader(path):
    with open(path, mode='rb') as reader:
        feature = pickle.load(reader)
        return feature


def max_sentence_len(examples):
    max_len = 0
    normal_sentence_num = 0
    abnormal_sentence_num = 0
    for example in tqdm(examples, desc='Calculating max sentence length'):
        length = len(example.question) + len(example.content)
        if length <= 256:
            normal_sentence_num += 1
        else:
            abnormal_sentence_num += 1
        if length > max_len:
            max_len = length
        else:
            ...
    return max_len, normal_sentence_num, abnormal_sentence_num


def analysis_data_distribution(examples):
    num_sentence_length = [i for i in range(1000)]
    _0_list = [0 for i in range(1000)]
    num_dic = dict(zip(num_sentence_length, _0_list))
    for example in tqdm(examples, desc='Analysis data distribution:'):
        length = len(example.question) + len(example.content)
        num_dic[length] += 1
    x_label = num_dic.keys()
    y_label = num_dic.values()
    plt.bar(x_label, y_label)
    plt.xlabel('sentence length')
    plt.ylabel('num of sentence')
    plt.show()


def conver_answer_to_label(content, answer):
    assert answer in content
    label = [BIO_DICT["O"] for index in range(len(content))]
    start_index = content.find(answer)
    end_index = start_index + len(answer) - 1
    # 如果要从content中获取字串 则需要 content[start_index: end_index+1]
    label[start_index: end_index + 1] = [BIO_DICT["I"] for index in range(len(answer))]
    label[start_index] = BIO_DICT["B"]
    return label


def read_examples(examples_path):
    examples = []
    with open(examples_path, mode='r') as reader:
        read = reader.read()
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
                example = Example(key=key, question=question, answer=answer,
                                  content=content, label=label)
                examples.append(example)
    return examples


def split_examples(examples):
    train_examples, dev_examples = train_test_split(examples, shuffle=True, random_state=626, test_size=0.1)
    return train_examples, dev_examples


def padding(head_masks, labels, max_len):
    padded_head_masks, padded_labels = [], []
    for head_mask, label in tqdm(zip(head_masks, labels), total=len(head_masks), desc='Padding'):
        pad_4_head_mask = [0 for i in range(max_len - len(head_mask))]
        pad_4_label = [0 for i in range(max_len - len(label))]
        padded_head_mask = head_mask + pad_4_head_mask
        padded_label = label + pad_4_label
        padded_head_masks.append(padded_head_mask)
        padded_labels.append(padded_label)
    return padded_head_masks, padded_labels


def create_head_mask(example):
    question = example.question
    content = example.content
    # [CLS] question [SEP] content [SEP] [PAD] [PAD]
    mask_4_question = [0 for i in range(len(question))]
    mask_4_content = [1 for i in range(len(content))]
    heed_mask = [0] + mask_4_question + [0] + mask_4_content + [0]
    return heed_mask


def example_filter(examples, max_length):
    new_examples = []
    for example in tqdm(examples, desc='Filter examples'):
        length = len(example.question) + len(example.content)
        if length <= max_length:
            new_examples.append(example)
        else:
            ...
    return new_examples


def convert_examples_to_features(examples: Example, model_name, max_length):
    questions, contents, labels, head_masks = [], [], [], []
    for example in tqdm(examples, desc='Reading examples'):
        questions.append(example.question)
        contents.append(example.content)
        labels.append(example.label)
        head_mask = create_head_mask(example)
        head_masks.append(head_mask)
    print("===== Encoding... =====")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    encoded = tokenizer(questions, contents, truncation=True, padding=True, max_length=max_length)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    length = len(input_ids[0])
    assert len(input_ids) == len(attention_mask) == len(token_type_ids)

    assert len(input_ids) == len(attention_mask) == len(token_type_ids) == \
           len(head_masks) == len(labels)
    assert len(input_ids[0]) == len(attention_mask[0]) == len(token_type_ids[0]) == \
           len(head_masks[0]) == len(labels[0])
    features = Feature(input_ids=input_ids, attention_masks=attention_mask,
                       token_type_ids=token_type_ids, head_mask=head_masks, labels=labels)
    return features


def create_dataloader(features, batch_size):
    input_ids = torch.tensor(features.input_ids)
    attention_masks = torch.tensor(features.attention_masks)
    token_type_ids = torch.tensor(features.token_type_ids)
    labels = torch.tensor(features.labels)
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
