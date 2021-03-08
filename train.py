import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
from IE import data_processor as P
from torch.utils.tensorboard import SummaryWriter
from IE.model import BertModelForInformationExtract as BIO
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

EPOCH = 5
BATCH_SIZE = 36
LOG_PATH = './log'
DATA_SAVE_PATH = './_data/processed_data.json'
TRAIN_FEATURE_PATH = './data_pkl/train_features.pkl'
DEV_FEATURE_PATH = './data_pkl/dev_features.pkl'
MODEL_SAVE_PATH = './trained_model/MRC4BIO_epoch_'
MODEL_SUFFIX = '.pt'
MODEL_NAME = 'bert-base-chinese'


def loading_model(model_name):
    print('===== Loading model... =====')
    config = BertConfig.from_pretrained(model_name)
    model = BIO.BertMRC4BIO(config, model_name)
    return model


def train(model, train_dataloader):
    model.cuda()
    # Set optimizer and scheduler
    total_step = EPOCH * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                num_training_steps=total_step)
    # tensor board
    tensor_board_writer = SummaryWriter(LOG_PATH)
    # training
    epoch_loss = 0  # 对loss做平滑处理
    for step, batch in tqdm(enumerate(train_dataloader), desc='Training', total=len(train_dataloader)):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        batch = tuple(t.cuda() for t in batch)
        b_input_ids, b_attention_masks, b_token_type_ids, b_head_masks, b_labels = batch
        output = model(b_input_ids, attention_mask=b_attention_masks, token_type_ids=b_token_type_ids,
                       labels=b_labels, head_mask=b_head_masks)
        loss = output
        epoch_loss += loss.item()
        tensor_board_writer.add_scalar('train_loss', epoch_loss / (step + 1),
                                       step + i * len(train_dataloader))
        tensor_board_writer.flush()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model


def validation(model, dev_dataloader, examples_length):
    model.cuda()
    model.eval()
    dev_epoch_loss = 0

    for step, batch in tqdm(enumerate(dev_dataloader), desc='Validation', total=len(dev_dataloader)):
        batch = tuple(t.cuda() for t in batch)
        d_input_ids, d_attention_masks, d_token_type_ids, d_head_masks, d_labels = batch
        with torch.no_grad():
            output = model(d_input_ids, attention_mask=d_attention_masks, token_type_ids=d_token_type_ids,
                       labels=None, head_mask=d_head_masks)
            # loss
            d_head_mask_index = d_head_masks.view(-1) == 1
            token_logits = output.view(-1, len(P.BIO_DICT))[d_head_mask_index]
            d_labels = d_labels.long().view(-1)[d_head_mask_index]
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(token_logits, d_labels)
            dev_epoch_loss += loss.item()
            # prediction
            logits = output  # batch_size * seq_len * num_label
            predictions = torch.argmax(logits, dim=2)  # batch_size * seq_len
            # predictions = predictions.detach().cpu().numpy().tolist()
            # detached_labels = d_labels.detach().cpu().numpy().tolist()
            # detached_head_masks = d_head_masks.detach().cpu().numpy().tolist()
            for prediction, detached_label, detached_head_mask in zip(predictions, d_labels, d_head_masks):
                head_mask_index = detached_head_mask == 1
                prediction = prediction[head_mask_index]
                detached_label = detached_label[head_mask_index]
                pre_B_indexs = list(filter(lambda x: prediction[x] == P.BIO_DICT['B'], list(range(len(prediction)))))
                pre_I_indexs = list(filter(lambda x: prediction[x] == P.BIO_DICT['I'], list(range(len(prediction)))))
                pre_indexs = (pre_B_indexs + pre_I_indexs).sort()
                true_B_indexs = list(filter(lambda x: detached_label[x] == P.BIO_DICT['B'], list(range(len(detached_label)))))
                true_I_indexs = list(filter(lambda x: detached_label[x] == P.BIO_DICT['I'], list(range(len(detached_label)))))
                true_indexs = (true_B_indexs + true_I_indexs).sort()

    return model


if __name__ == '__main__':
    examples = P.read_examples(DATA_SAVE_PATH)
    train_examples, dev_examples = P.split_examples(examples)
    train_features = P.pkl_reader(TRAIN_FEATURE_PATH)
    dev_features = P.pkl_reader(DEV_FEATURE_PATH)
    train_dataloader = P.create_dataloader(train_features, BATCH_SIZE)
    dev_dataloader = P.create_dataloader(dev_features, BATCH_SIZE)
    model = loading_model(MODEL_NAME)
    for i in range(0, EPOCH):
        # model = train(model, train_dataloader)
        model = validation(model, dev_dataloader, dev_examples)
        torch.save(model.state_dict(), MODEL_SAVE_PATH + str(i) + MODEL_SUFFIX)
