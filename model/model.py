import torch
from transformers import BertModel, BertForTokenClassification, BertForSequenceClassification
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from model.model_utils import label2span, span2merged_sent
from evaluate.utils import ngram_similarity


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

class RelRelationClassifierBaseline(nn.Module):
    def __init__(self, type_emb_size=512, hidden_size=1024, label_num=3, tokenizer=None, model_type='base'):
        super(RelRelationClassifierBaseline, self).__init__()
        self.start_index = tokenizer.convert_tokens_to_ids(['[concept_start]'])[0]
        self.end_index = tokenizer.convert_tokens_to_ids(['[concept_end]'])[0]
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-{}-uncased'.format(model_type), add_pooling_layer=False)

    def forward(self, sent, length, span_start1, type1,
                span_start2, type2, relation=None):
        ret = []
        for s, t1, t2, start1, start2 in zip(sent, type1, type2, span_start1, span_start2):
            if t1 == t2:
                tmp1 = []
                #print(s)
                #print(self.start_index, self.end_index)
                #print(start1, start2)
                assert s[start1] == self.start_index
                assert s[start2] == self.start_index
                for x in s[start1 + 1:]:
                    if x == self.end_index:
                        break
                    tmp1.append(x.item())

                tmp2 = []
                for x in s[start2 + 1:]:
                    if x == self.end_index:
                        break
                    tmp2.append(x.item())
                '''    
                flag = False
                for x in tmp1:
                    if x in tmp2:
                        flag = True
                for x in tmp2:
                    if x in tmp1:
                        flag = True
                '''
                span1 = ' '.join(self.tokenizer.convert_ids_to_tokens(tmp1))
                span2 = ' '.join(self.tokenizer.convert_ids_to_tokens(tmp2))
                score = ngram_similarity(span1, span2)
                flag = score > 0.6
                if flag:
                    ret.append(torch.tensor([0., 1.]))
                else:
                    ret.append(torch.tensor([1., 0.]))
            else:
                ret.append(torch.tensor([1., 0.]))
        return torch.stack(ret)


class RelationClassifier(nn.Module):
    # 0: No RelationShip, 1: Same, 2: Have Relationship
    def __init__(self, type_emb_size=512, hidden_size=1024, label_num=3, tokenizer=None, model_type='base'):
        super(RelationClassifier, self).__init__()
        self.label_num = label_num
        
        self.encoder = BertModel.from_pretrained('bert-{}-uncased'.format(model_type),
                                                 add_pooling_layer=False)
        '''
        self.encoder = BertForSequenceClassification.from_pretrained('bert-{}-uncased'.format(model_type))
        '''
        if tokenizer is not None:
            self.encoder.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.type_embedding = nn.Embedding(num_embeddings=2,
                                            embedding_dim=type_emb_size)
        
        self.transformer = nn.Sequential(
            #nn.LayerNorm((type_emb_size + self.encoder.config.hidden_size)),
            nn.Linear(2 * (type_emb_size + self.encoder.config.hidden_size), hidden_size),
            nn.ReLU()
        )
        self.trans2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_size, label_num)
        
    def forward(self, sent, length, span_start1, type1,
                span_start2, type2, relation=None):
        attention_mask = length_to_mask(length)
        output = self.encoder(sent, attention_mask=attention_mask)[0]
        output = self.dropout(output)
        emb = []
        for start1, start2, o in zip(span_start1, span_start2, output):
            emb.append(torch.cat([o[start1], o[start2]]))
        emb = torch.stack(emb)
        merged_emb = torch.cat([emb, self.type_embedding(type1), self.type_embedding(type2)], dim=-1)
        feature = self.transformer(merged_emb)
        #feature = self.trans2(feature)
        logits = self.classifier(feature)
        if relation is not None:
            #r = self.encoder(sent, attention_mask=attention_mask, labels=relation)
            #return ret.logits, ret.loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.label_num),
                            relation.view(-1))
            return logits, loss
        else:
            return logits

    def get_span_emb(self, sent, span_start, ty, length):
        attention_mask = length_to_mask(length)
        emb = self.encoder(sent, attention_mask=attention_mask)[0] # bs * length * emb
        ret = []
        for start, e in zip(span_start, emb):
            ret.append(e[start])
        ret = torch.stack(ret) # bs * emb
        return torch.cat([ret, self.type_embedding(ty)], dim=-1)

    def predict(self, sent, span_start1, type1,
                span_start2, type2, return_logit=False):
            sent = sent.unsqueeze(0)
            span_start1 = torch.tensor([span_start1], dtype=torch.int64, device=sent.device)
            span_start2 = torch.tensor([span_start2], dtype=torch.int64, device=sent.device)
            type1 = torch.tensor([type1], dtype=torch.int64, device=sent.device)
            type2 = torch.tensor([type2], dtype=torch.int64, device=sent.device)
            length = torch.tensor([sent.size(1)], dtype=torch.int64, device=sent.device)
            logits = self.forward(sent, length, span_start1, type1,
                                  span_start2, type2)

            if return_logit:
                return torch.softmax(logits[0], dim=-1)
            else:
                return torch.argmax(logits[0], dim=-1)


class ExtractConcept(nn.Module):
    def __init__(self, label_nums=2, dropout=0.1, model_type='base'):
        super(ExtractConcept, self).__init__()
        self.label_nums = label_nums
        self.encoder = BertModel.from_pretrained(
            'bert-{}-uncased'.format(model_type), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.encoder.config.hidden_size
        self.table_classifier = nn.Linear(hidden_size, label_nums)
        self.column_classifier = nn.Linear(hidden_size, label_nums)

    def forward(self, sent, table_label, column_label, length):
        attention_mask = length_to_mask(length)
        outputs = self.encoder(sent, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        #print(sequence_output.size(), table_label.size())
        table_logits = self.table_classifier(sequence_output)
        column_logits = self.column_classifier(sequence_output)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(table_logits.view(-1, self.label_nums),
                        table_label.view(-1)) + \
                        loss_fct(column_logits.view(-1, self.label_nums),
                        column_label.view(-1))
        return TokenClassifierOutput(
            loss=loss,
            logits=torch.stack([table_logits, column_logits]).transpose(0, 1),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict(self, sent, length, threshold=0.5):
        attention_mask = length_to_mask(length)
        #print(sent.size(), length.size(), attention_mask.size())
        outputs = self.encoder(sent, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        #print(sequence_output.size(), table_label.size())
        table_logits = self.table_classifier(sequence_output)
        column_logits = self.column_classifier(sequence_output)
        #print(table_logits.size(), column_logits.size())
        table_logits = table_logits.softmax(dim=-1)
        column_logits = column_logits.softmax(dim=-1)
        table_logits[:,:,0] += (2 * threshold) - 1
        column_logits[:, :, 0] += (2 * threshold) - 1
        table_pred = table_logits.argmax(dim=-1)
        column_pred = column_logits.argmax(dim=-1)
    
        return outputs[0], table_pred, column_pred


class PipeLine(nn.Module):
    def __init__(self, tokenizer, concept_start_index, concept_end_index,
                 threshold=0.4, alpha=0.5):
        super(PipeLine, self).__init__()
        self.extractor = ExtractConcept()
        self.classifier = RelationClassifier(tokenizer=tokenizer)
        self.concept_start_index = concept_start_index
        self.concept_end_index = concept_end_index
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, sents, lengths):
        concept_list = []
        _, table_preds, column_preds = self.extractor.predict(sents, lengths)
        for i, (t_pred, c_pred, leng) in enumerate(zip(table_preds, column_preds, lengths)):
            #print(t_pred.size(), c_pred.size())
            t_pred = t_pred.tolist()[:leng]
            c_pred = c_pred.tolist()[:leng]
            table_spans = label2span(t_pred, i, 0)
            column_spans = label2span(c_pred, i, 1)
            concept_list += table_spans
            concept_list += column_spans

        for concept in concept_list:
            concept.name = ' '.join(sents[concept.sent_id][concept.start_index:
                                                           concept.end_index])
        concept_num = len(concept_list)
        relation_map = {c1:{c2: 0 for c2 in concept_list}
                     for c1 in concept_list}
        #same_map = {c1:{c2: ngram_similarity(c1.name, c2.name) for c2 in concept_list}
        #            for c1 in concept_list}
        #start_token = torch.tensor([self.concept_start_index], dtype=torch.int64, device=sents.device)
        #end_token = torch.tensor([self.concept_end_index], dtype=torch.int64, device=sents.device)
        for i in range(concept_num):
            for j in range(i + 1, concept_num):
                span1 = concept_list[i]
                span2 = concept_list[j]
                sent1 = sents[span1.sent_id]
                sent2 = sents[span2.sent_id]
                if sent1 == sent2:
                    sent, span1, span2 = span2merged_sent(span1, span2, sent1,
                                                          self.concept_start_index,
                                                          self.concept_end_index)
                pred = self.classifier.predict(
                        sent1, span1.start_index, span1.type,
                        sent2, span2.start_index, span2.type,
                        return_logit=True
                        ).tolist()
                relation_map[span1][span2] = pred[1]
                relation_map[span2][span1] = pred[1]
        import pickle
        with open('./inter_pred.pkl', 'wb') as file:
            pickle.dump((concept_list, relation_map), file)
        return concept_list, relation_map

    def aggregate(self, concept_list, same_map, relation_map):

        '''
        span_weight = {span:1 for span in concept_list}
        while len(concept_list) >= 2:
            max_similarity = 0
            max_span1 = None
            max_span2 = None
            concept_num = len(concept_list)
            for i in range(concept_num):
                for j in range(i + 1, concept_num):
                    span1 = concept_list[i]
                    span2 = concept_list[j]
                    if torch.argmax(label_map[span1][span2]) == 1:
                        similarity = (1 - self.alpha) * (torch.stack([
                            (label_map[span1][x] * label_map[span2][x]).sum()
                            if x != span1 and x != span2 else torch.tensor(0.)
                            for x in concept_list]).mean()) + \
                                     self.alpha * label_map[span1][span2][1]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            max_span1 = span1
                            max_span2 = span2
            if max_similarity >= self.threshold:
                self.merge(concept_list, label_map, span_weight, max_span1, max_span2)
            else:
                break
        '''

    def merge(self, concept_list, label_map, span_weight, span1, span2):
        concept_list.remove(span2)
        label_map.pop(span2)
        weight1 = span_weight[span1]
        weight2 = span_weight[span2]
        span_weight.pop(span2)
        span_weight[span1] += weight2
        for span in concept_list:
            if span != span1:
                type1 = label_map[span][span1]
                type2 = label_map[span][span2]
                new_type = (type1 * weight1 + type2 * weight2) / (weight1 + weight2)
                label_map[span1][span] = new_type
                label_map[span][span1] = new_type
            label_map[span].pop(span2)


    def load_state_dict(self, state_dict, strict=True):
        self.extractor.load_state_dict(state_dict['extractor'])
        self.classifier.load_state_dict(state_dict['classifier'])







    
