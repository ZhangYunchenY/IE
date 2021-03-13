import json
import copy
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
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
                 answer: list,
                 answer_ids: list
                 ):
        self.key = key
        self.question = question
        self.content = content
        self.answer = answer
        self.answer_ids = answer_ids


class Feature:
    def __init__(self,
                 input_ids,
                 attention_masks,
                 token_type_ids,
                 labels,
                 offsets,
                 head_mask=None
                 ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.head_mask = head_mask
        self.labels = labels
        self.offsets = offsets


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


# def conver_answer_to_label(content, answer):
#     assert answer in content
#     label = [BIO_DICT["O"] for index in range(len(content))]
#     start_index = content.find(answer)
#     end_index = start_index + len(answer) - 1
#     # 如果要从content中获取字串 则需要 content[start_index: end_index+1]
#     label[start_index: end_index + 1] = [BIO_DICT["I"] for index in range(len(answer))]
#     label[start_index] = BIO_DICT["B"]
#     return label


def read_examples(examples_path, model_name):
    examples = []
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    with open(examples_path, mode='r') as reader:
        read = reader.read()
        examples_dict = json.loads(read)
    for item in tqdm(examples_dict["documents"], desc='Processing examples'):
        content = ""
        for sentence in item["document"]:
            content += sentence["text"]
        for question_answers_dict in item["qas"][0]:
            question = question_answers_dict["question"]
            key = item["key"]
            answer = []
            answer_ids = []
            for answer_dict in question_answers_dict["answers"]:
                answer.append(answer_dict["text"])
                answer_ids.append(tokenizer(answer)['input_ids'][0])
            example = Example(key=key, question=question, answer=answer,
                              content=content, answer_ids=answer_ids)
            examples.append(example)
    return examples


def split_examples(examples):
    train_examples, dev_examples = train_test_split(examples, shuffle=True, random_state=626, test_size=0.1)
    return train_examples, dev_examples


def padding(labels, max_length):
    # 标注
    new_labels = []
    for label in tqdm(labels, desc='Padding labels'):
        empty_label = [BIO_DICT['O'] for i in range(max_length)]
        for index in label:
            start_index = index[0]
            end_index = index[-1]
            empty_label[start_index: end_index + 1] = [BIO_DICT['I'] for i in range(end_index - start_index + 1)]
            empty_label[start_index] = BIO_DICT['B']
        new_labels.append(empty_label)
    return new_labels


def create_head_mask(token_type_ids):
    # [CLS] question [SEP] content [SEP] [PAD] [PAD]
    head_masks = []
    for token_type_id in tqdm(token_type_ids, desc='Creating head masks'):
        _1_index = [i for i, x in enumerate(token_type_id) if x == 1]
        head_mask = copy.deepcopy(token_type_id)
        head_mask[_1_index[-1]] = 0
        head_masks.append(head_mask)
    return head_masks


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
    questions, contents, answer_ids = [], [], []
    labels, offsets = [], []
    for example in tqdm(examples, desc='Reading examples'):
        questions.append(example.question)
        contents.append(example.content)
        answer_ids.append(example.answer_ids)
    print("===== Encoding... =====")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    encoded = tokenizer(questions, contents, truncation=True, padding=True, max_length=max_length)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    for encoding in tqdm(encoded.encodings, desc='Building offset'):
        offset = encoding.offsets
        offsets.append(offset)
    for input_id, answer_id in tqdm(zip(input_ids, answer_ids), desc='Create labels', total=len(input_ids)):
        # 将问题的答案全部标出
        index_list = []
        for aid in answer_id:
            aid = aid[1: -1]
            appearance_index = list(filter(lambda x: input_id[x] == aid[0], list(range(len(input_id)))))
            # 为了避免产生歧义，将句子中出现和答案相同的字符串全部标出
            for index in appearance_index:
                if input_id[index: index + len(aid)] == aid:
                    temp_list = list(range(index, index + len(aid)))
                    index_list.append(temp_list)
                else:
                    ...
        labels.append(index_list)
    length = len(input_ids[0])
    labels = padding(labels, length)
    assert len(input_ids) == len(attention_mask) == len(token_type_ids)
    head_masks = create_head_mask(token_type_ids)
    assert len(input_ids) == len(attention_mask) == len(token_type_ids) == \
           len(head_masks) == len(labels)
    features = Feature(input_ids=input_ids, attention_masks=attention_mask,
                       token_type_ids=token_type_ids, head_mask=head_masks,
                       labels=labels, offsets=offsets)
    return features


def create_dataloader(features, batch_size):
    input_ids = torch.tensor(features.input_ids)
    attention_masks = torch.tensor(features.attention_masks)
    token_type_ids = torch.tensor(features.token_type_ids)
    head_mask = torch.tensor(features.head_mask)
    labels = torch.tensor(features.labels)
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids,
                            head_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def bio_inference(prediction):
    pre_spans = []
    FLAG = -1
    for index, value in enumerate(prediction):
        if value == BIO_DICT['B']:
            start_index = index
            FLAG = BIO_DICT['B']
        elif value == BIO_DICT['B'] and FLAG == BIO_DICT['B']:
            end_index = index
            pre_spans.append([start_index, end_index])
            FLAG = BIO_DICT['B']
        elif value == BIO_DICT['O'] and FLAG == BIO_DICT['B']:
            end_index = index
            pre_spans.append([start_index, end_index])
            FLAG = BIO_DICT['O']
        elif value == BIO_DICT['I'] and FLAG == BIO_DICT['B']:
            FLAG = BIO_DICT['I']
        elif value == BIO_DICT['I'] and FLAG == BIO_DICT['I']:
            FLAG = BIO_DICT['I']
        elif value == BIO_DICT['O'] and FLAG == BIO_DICT['I']:
            end_index = index
            pre_spans.append([start_index, end_index])
            FLAG = BIO_DICT['O']
    return pre_spans
