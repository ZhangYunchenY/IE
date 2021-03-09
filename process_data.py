from IE.data_processor import *

MAX_LENGTH = 256
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
    examples = read_examples(DATA_SAVE_PATH, MODEL_NAME)
    train_examples, dev_examples = split_examples(examples)
    train_sequence_len, train_sentence_num, train_ab_sentence_num = max_sentence_len(train_examples)
    dev_sequence_len, dev_sentence_num, dev_ab_sentence_num = max_sentence_len(dev_examples)
    print(f'Train sentence max length: {train_sequence_len}, Dev sentence max length: {dev_sequence_len}')
    print(f'Train normal sentence num: {train_sentence_num}, abnormal sentence num: {train_ab_sentence_num}')
    print(f'Dev normal sentence num: {dev_sentence_num}, abnormal sentence num: {dev_ab_sentence_num}')
    # analysis_data_distribution(train_examples)
    # analysis_data_distribution(dev_examples)
    train_examples = example_filter(train_examples, MAX_LENGTH)
    dev_examples = example_filter(dev_examples, MAX_LENGTH)
    train_features = convert_examples_to_features(train_examples, MODEL_NAME, MAX_LENGTH)
    dev_features = convert_examples_to_features(dev_examples, MODEL_NAME, MAX_LENGTH)
    pkl_writer(train_features, TRAIN_FEATURES_PATH)
    pkl_writer(dev_features, DEV_FEATURES_PATH)
