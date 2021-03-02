import torch
import torch.nn as nn
from transformers import BertModel


class BertMRCBIO(nn.Module):
    def __init__(self, config, token_mask, bio_num=3):
        super(BertMRCBIO, self).__init__()
        self.bio_num = bio_num
        self.token_mask = token_mask
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=config)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(self.hidden_size, 2)
        self.bio_classifier = nn.Linear(self.hidden_size, self.bio_num)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=None
        )
        pass
