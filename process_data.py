from IE.data_processor import *


DATA_PATH = './_data/labed_data.json'
DATA_SAVE_PATH = './_data/processed_data.json'
UNLABELED_DATA_PATH = './_data/unlabeled_data_dir/content.json'
UNLABELED_DATA_SAVE_PATH = './_data/unlabeled_content.json'


if __name__ == '__main__':
    data_reader_json(DATA_PATH, DATA_SAVE_PATH)
    data_reader_json(UNLABELED_DATA_PATH, UNLABELED_DATA_SAVE_PATH)