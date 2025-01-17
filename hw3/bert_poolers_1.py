import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertModel,
    BertEmbeddings, BertEncoder, BertForSequenceClassification, BertPooler,
)


class MeanMaxTokensBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wmmt = nn.Linear(2*config.hidden_size, config.hidden_size) # W_MMT
        self.activation = nn.Tanh()

        self.hidden_size = config.hidden_size
        # raise NotImplementedError

    def forward(self, hidden_states, *args, **kwargs):
        norm = hidden_states.norm(dim=2) # [N, T]
        max_id = torch.argmax(norm, dim=1) # [N]
        max_id = torch.reshape(max_id, (-1, 1, 1)) # [N, 1, 1]
        indices = torch.broadcast_to(max_id, (-1, 1, self.hidden_size)) # [N, 1, H]
        maxT = torch.gather(hidden_states, 1, indices) # [N, 1, H]

        # option 1: sum(T_i) || max(T_i)
        sumT = torch.mean(hidden_states, dim=1) # [N, H]
        c_mmt = torch.cat((sumT, maxT.squeeze()), dim=1) # [N, 2H]

        pooled_output = self.wmmt(c_mmt) # [N, H]
        pooled_output = self.activation(pooled_output) # [N, H]

        return pooled_output
        # raise NotImplementedError


class MyBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # MyBertPooler

        # Attentive Pooling
        self.query = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax()
        self.w_2h = nn.Linear(2*config.hidden_size, config.hidden_size) # w_2h

        self.hidden_size = config.hidden_size
        self.activation = nn.Tanh()

    def forward(self, hidden_states, *args, **kwargs):
        # hidden_states: [N, T, H]

        score = self.query(hidden_states[:, 1:, :]) # hidden_states[:, 1:, :] = [N, T-1, H] - everything except [CLS]
        score /= torch.sqrt(torch.tensor(self.hidden_size))
        score = self.softmax(score)
        pooled_output = torch.sum(score * hidden_states[:, 1:, :], dim=1) # [N, H]
        pooled_output = torch.cat((hidden_states[:, 0], pooled_output), dim=1) # [N, 2H]
        pooled_output = self.w_2h(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class MyBertConfig(BertConfig):
    def __init__(self, pooling_layer_type="CLS", **kwargs):
        super().__init__(**kwargs)
        self.pooling_layer_type = pooling_layer_type


class MyBertModel(BertModel):

    def __init__(self, config: MyBertConfig):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        if config.pooling_layer_type == "CLS":
            # See src/transformers/models/bert/modeling_bert.py#L869
            self.pooler = BertPooler(config)
        elif config.pooling_layer_type == "MEAN_MAX":
            self.pooler = MeanMaxTokensBertPooler(config)
        elif config.pooling_layer_type == "MINE":
            self.pooler = MyBertPooler(config)
        else:
            raise ValueError(f"Wrong pooling_layer_type: {config.pooling_layer_type}")

        self.init_weights()

    @property
    def pooling_layer_type(self):
        return self.config.pooling_layer_type


class MyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.bert.pooling_layer_type in ["CLS", "MEAN_MAX", "MINE"]:
            return super().forward(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict
            )
        else:
            raise NotImplementedError
