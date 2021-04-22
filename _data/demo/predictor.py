# -*- coding:utf-8 -*-
import json
import copy
import torch
from tqdm import tqdm
from model.BertModelForInformationExtract import BertMRC4BIO
from transformers import BertTokenizerFast, BertConfig

# 模型下载链接：https://drive.google.com/file/d/1DhASp98LY_0JI7NOVxxgnQ8RQkUQz9eB/view?usp=sharing
# 下载完后保存在./trained_model 中
MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = './trained_model/ner.pt'
qas_content = {"question": None, "answers": []}
answer_dict = {"start_block": "0",
               "start": None,
               "end_block": "0",
               "end": None,
               "text": None,
               "sub_answer": None
               }
BIO_DICT = {"B": 0, "I": 1, "O": 2}


def find_all(finded_str, sub):
    start = 0
    while True:
        start = finded_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


class Predictor:

    def __init__(self, model_name, model_path):
        """
            初始化模型、配置
        """
        self.model_name = model_name
        self.model_path = model_path
        static_dict = torch.load(self.model_path)
        config = BertConfig.from_pretrained(self.model_name)
        model = BertMRC4BIO(
            config=config,
            model_name=model_name,
            bio_num=3
        )
        self.model = model

    def answer_postion(self, answers, text):
        """
        :param answer: 答案
        :param text: 句子
        :return: 答案在句子中的所有位置的开头和结尾，分别为一个列表
        """
        start_postion_list = []
        for answer in answers:
            start_postion_list += list(find_all(text, answer))
        end_postion_list = list(range(len(start_postion_list)))
        for i in range(len(start_postion_list)):
            end_postion_list[i] = start_postion_list[i] + len(answer) - 1
        return start_postion_list, end_postion_list

    def inference(self, ques, text):
        """
        模拟模型
        :param ques: 问题(str)
        :param text: 句子(str)
        :return: 答案(str)
        """
        tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        encoded = tokenizer(ques, text, truncation=True, padding=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        attention_mask = encoded['attention_mask']
        head_mask = self.create_head_mask([token_type_ids])[0]
        self.model.cuda()
        self.model.eval()
        input_ids = torch.tensor([input_ids]).cuda()
        token_type_ids = torch.tensor([token_type_ids]).cuda()
        attention_mask = torch.tensor([attention_mask]).cuda()
        head_mask = torch.tensor([head_mask]).cuda()
        offset = encoded.encodings[0].offsets
        with torch.no_grad():
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
                head_mask=head_mask
            )
            logits = output
            predictions = torch.argmax(logits, dim=2)
            head_mask_index = head_mask == 1
            predictions = predictions[head_mask_index].detach().cpu().numpy().tolist()
            offset = torch.tensor([offset])[head_mask_index].detach().cpu().numpy().tolist()
            pre_spans = self.bio_inference(predictions)
            pre_spans_4_content = []
            for span in pre_spans:
                start, end = span
                start_index_4_content = offset[start][0]
                end_index_4_content = offset[end - 1][-1]
                pre_spans_4_content.append([start_index_4_content, end_index_4_content])
            answers = []
            for span in pre_spans_4_content:
                start, end = span
                answer = text[start: end]
                answers.append(answer)
            return answers


    def make_content_list(self, start_postion_list, end_postion_list, answer_text):
        answer_model = copy.deepcopy(answer_dict)
        answer_model['text'] = answer_text
        one_answer_list = []
        for start_postion, end_postion in zip(start_postion_list, end_postion_list):
            one_answer = copy.deepcopy(answer_model)
            one_answer['start'] = start_postion
            one_answer['end'] = end_postion
            one_answer_list.append(one_answer)
        return one_answer_list


    def create_head_mask(self, token_type_ids):
        # [CLS] question [SEP] content [SEP] [PAD] [PAD]
        head_masks = []
        for token_type_id in tqdm(token_type_ids, desc='Creating head masks'):
            _1_index = [i for i, x in enumerate(token_type_id) if x == 1]
            head_mask = copy.deepcopy(token_type_id)
            head_mask[_1_index[-1]] = 0
            head_masks.append(head_mask)
        return head_masks


    def bio_inference(self, prediction):
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


    def predict(self, content: dict) -> dict:
        """
        输入标注格式，已转为dict
        输出同标注格式，dict格式
        :param content: 标注格式，见样例:
        :return str:
        """
        zhongxin_answer = content['qas'][0][0]
        output = content
        output['qas'][0] = []
        ques_text = content['document'][0]['text']
        for ques in ["原因中的核心名词", "原因中的谓语或状态", "中心词", "结果中的核心名词", "结果中的谓语或状态"]:
            if ques == '中心词':
                output['qas'][0].append(zhongxin_answer)
            else:
                answer_text_list = self.inference(ques, ques_text, self.model)
                answer_content = copy.deepcopy(qas_content)
                answer_content['question'] = ques
                for answer_text in answer_text_list:
                    start_postion_list, end_postion_list = self.answer_postion(answer_text, ques_text)
                    one_answer_list = self.make_content_list(start_postion_list, end_postion_list, answer_text)
                    for one_answer in one_answer_list:
                        answer_content['answers'].append(one_answer)
                output['qas'][0].append(answer_content)
        return output


if __name__ == "__main__":
    example_input = '{"document": [{"block_id": "0", "text": ' \
                    '"08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。"}], ' \
                    '"key": "79c29068d30a686", "qas": [[{"question": "中心词", "answers": [{"start_block": "0", ' \
                    '"start": 57, "end_block": "0", "end": 58, "text": "导致", "sub_answer": null}]}]]} '
    example_output = '{"document": [{"block_id": "0", "text": ' \
                     '"08年4月，郑煤集团拟以非公开发行的方式进行煤炭业务整体上市，解决与郑州煤电同业竞争问题，但之后由于股市的大幅下跌导致股价跌破发行价而被迫取消整体上市。"}], ' \
                     '"key": "79c29068d30a686", "qas": [[{"question": "原因中的核心名词", "answers": [{"start_block": "0", ' \
                     '"start": 50, "end_block": "0", "end": 51, "text": "股市", "sub_answer": null}]}, {"question": ' \
                     '"原因中的谓语或状态", "answers": [{"start_block": "0", "start": 53, "end_block": "0", "end": 56, ' \
                     '"text": "大幅下跌", "sub_answer": null}]}, {"question": "中心词", "answers": [{"start_block": "0", ' \
                     '"start": 57, "end_block": "0", "end": 58, "text": "导致", "sub_answer": null}]}, {"question": ' \
                     '"结果中的核心名词", "answers": [{"start_block": "0", "start": 59, "end_block": "0", "end": 60, ' \
                     '"text": "股价", "sub_answer": null}]}, {"question": "结果中的谓语或状态", "answers": [{"start_block": "0", ' \
                     '"start": 61, "end_block": "0", "end": 65, "text": "跌破发行价", "sub_answer": null}]}]]} '
    obj = Predictor(MODEL_NAME, MODEL_PATH)
    output = obj.predict(json.loads(example_input))
    print(output)
    print(json.loads(example_output))
    print(output == json.loads(example_output))
