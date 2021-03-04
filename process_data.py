from IE.data_processor import *

MODEL_NAME = 'bert-base-chinese'
DATA_PATH = './_data/labed_data.json'
DATA_SAVE_PATH = './_data/processed_data.json'
UNLABELED_DATA_PATH = './_data/unlabeled_data_dir/content.json'
UNLABELED_DATA_SAVE_PATH = './_data/unlabeled_content.json'
TRAIN_FEATURES_PATH = './data_pkl/train_features.pkl'
DEV_FEATURES_PATH = './data_pkl/dev_features.pkl'


if __name__ == '__main__':
    # format_json(DATA_PATH, DATA_SAVE_PATH)
    # format_json(UNLABELED_DATA_PATH, UNLABELED_DATA_SAVE_PATH)
    examples = read_examples(DATA_SAVE_PATH)
    train_examples, dev_examples = split_examples(examples)
    train_features = convert_examples_to_features(train_examples, MODEL_NAME)
    dev_features = convert_examples_to_features(dev_examples, MODEL_NAME)
    pkl_writer(train_features, TRAIN_FEATURES_PATH)
    pkl_writer(dev_features, DEV_FEATURES_PATH)
