import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
from IE import data_processor as P
from torch.utils.tensorboard import SummaryWriter
from IE.model import BertModelForInformationExtract as BIO
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

EPOCH = 50
BATCH_SIZE = 80
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
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
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


def validation(model, dev_dataloader):
    model.cuda()
    model.eval()
    # tensor board
    tensor_board_writer = SummaryWriter(LOG_PATH)
    dev_epoch_loss = 0
    all_pre_spans_in_content = []
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
            batch_pre_spans_in_content = []
            for prediction, d_head_mask, offset in zip(predictions, d_head_masks, dev_features.offsets):
                head_mask_index = d_head_mask == 1
                prediction = prediction[head_mask_index].detach().cpu().numpy().tolist()
                offset = torch.tensor(offset)[head_mask_index].detach().cpu().numpy().tolist()
                pre_spans = P.bio_inference(prediction)  # span start_index, end_index + 1 刚好能够索引出所有字符串
                # O O B I I I O O O
                #     S       E
                pre_spans_4_content = []
                for span in pre_spans:
                    start, end = span
                    start_index_4_content = offset[start][0]
                    end_index_4_content = offset[end-1][-1]
                    pre_spans_4_content.append([start_index_4_content, end_index_4_content])
                batch_pre_spans_in_content.append(pre_spans_4_content)
        all_pre_spans_in_content += batch_pre_spans_in_content
    dev_epoch_loss /= len(dev_dataloader)
    # Calculate PRF
    denominator_precision, denominator_recall, TP = 0, 0, 0
    for example, span in zip(dev_examples, all_pre_spans_in_content):
        true_answer = example.answer
        denominator_recall += len(true_answer)
        content = example.content
        pre_answer = []
        for spn in span:
            start_index, end_index = spn
            pre_answer.append(content[start_index: end_index])
        for answer in pre_answer:
            if answer in true_answer:
                TP += 1
            else:
                ...
        denominator_precision += len(pre_answer)
    tensor_board_writer.add_scalar('dev_epoch_loss', dev_epoch_loss, i)
    if denominator_precision != 0:
        precision = TP / denominator_precision
        recall = TP / denominator_recall
        f1 = (2 * recall * precision) / (precision + recall)
        tensor_board_writer.add_scalar('dev_precision', precision, i)
        tensor_board_writer.add_scalar('dev_recall', recall, i)
        tensor_board_writer.add_scalar('dev-f1', f1, i)
    tensor_board_writer.flush()
    return model


if __name__ == '__main__':
    examples = P.read_examples(DATA_SAVE_PATH)
    train_examples, dev_examples = P.split_examples(examples)
    print(f'Train dataset size: {len(train_examples)}, dev dataset size: {len(dev_examples)}')
    train_features = P.pkl_reader(TRAIN_FEATURE_PATH)
    dev_features = P.pkl_reader(DEV_FEATURE_PATH)
    train_dataloader = P.create_dataloader(train_features, BATCH_SIZE)
    dev_dataloader = P.create_dataloader(dev_features, BATCH_SIZE)
    model = loading_model(MODEL_NAME)
    for i in range(0, EPOCH):
        model = train(model, train_dataloader)
        model = validation(model, dev_dataloader)
        torch.save(model.state_dict(), MODEL_SAVE_PATH + str(i) + MODEL_SUFFIX)
