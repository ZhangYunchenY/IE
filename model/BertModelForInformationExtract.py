import torch
import torch.nn as nn
from transformers import BertModel


class BertMRCBIO(nn.Module):
    def __init__(self, config, head_mask, head_word_mask=None, bio_num=3):
        super(BertMRCBIO, self).__init__()
        self.bio_num = bio_num
        self.head_mask = head_mask
        self.head_word_mask = head_word_mask  # 最后调整关于中心词的mask
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config, add_pooling_layer=False)
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
        sequence_output = outputs[0]
        sequence_without_head = torch.matmul(sequence_output, head_mask)
        token_logits = self.bio_classifier(sequence_without_head)
        if labels is not None:
            labels = labels.float()
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(token_logits, labels)
            return loss
        else:
            softmax_function = nn.Softmax(dim=1)
            logits = softmax_function(token_logits)
            return logits
