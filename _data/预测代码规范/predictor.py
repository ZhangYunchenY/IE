# -*- coding:utf-8 -*-
import json


class Predictor:

    def __init__(self):
        """
            初始化模型、配置
        """
        pass

    def predict(self, content: dict) -> dict:
        """
        输入标注格式，已转为dict
        输出同标注格式，dict格式
        :param content 标注格式，见样例:
        :return str:
        """
        pass
        return content


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
    obj = Predictor()
    output = obj.predict(json.loads(example_input))
    print(output != json.loads(example_output))
