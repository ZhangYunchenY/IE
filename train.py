from IE import data_processor as P
from IE.model import BertModelForInformationExtract as BIO
from transformers import BertConfig

TRAIN_FEATURE_PATH = './data_pkl/train_features.pkl'
DEV_FEATURE_PATH = './data_pkl/dev_features.pkl'
MODEL_NAME = 'bert-base-chinese'


def loading_model(model_name):
    print('===== Loading model... =====')
    config = BertConfig.from_pretrained(model_name)
    model = BIO.BertMRC4BIO(config, model_name)

if __name__ == '__main__':
    pass