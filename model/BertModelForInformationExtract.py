import torch
import torch.nn as nn
from transformers import BertModel


class BertMRC4BIO(nn.Module):
    def __init__(self, config, model_name, bio_num=3):
        super(BertMRC4BIO, self).__init__()
        self.bio_num = bio_num
        # self.head_mask = head_mask
        self.model_name = model_name
        # self.head_word_mask = head_word_mask  # 最后调整关于中心词的mask
        self.bert = BertModel.from_pretrained(self.model_name, config=config, add_pooling_layer=False)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bio_classifier = nn.Linear(self.hidden_size, self.bio_num)  # 用于BIO标注

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=None
        )
        sequence_output = outputs[0]  # batch_size * seq_len * embedding_size
        last_bert_layer = self.dropout(sequence_output)
        logits = self.bio_classifier(last_bert_layer)  # batch_size * seq_len * num_label
        if labels is not None:
            head_mask_index = head_mask.view(-1) == 1  # [batch_size * seq_len] ->  batch_size * seq_len
            sequence_logits = logits.view(-1, self.bio_num)[head_mask_index]
            # batch_size * seq_len * embedding_size -> [batch_size * seq_len] x 3
            labels = labels.long()
            labels = labels.view(-1)[head_mask_index]
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(sequence_logits, labels)
            return loss
        else:
            return logits  # batch_size * seq_len * num_label
