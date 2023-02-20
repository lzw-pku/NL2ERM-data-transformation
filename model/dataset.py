import sys
sys.path.append('..')
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import pickle
import os
from utils.database_util import Database, Table, Column
from collections import defaultdict
from model.model_utils import label2span
import random
from model.model_utils import span2merged_sent


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'

def link(bert_tok, tok):
    index = 0
    i = 0
    tmp_size = ''
    leng = len(bert_tok)
    position = []
    while i < leng:
        if i == 0 or position[-1] != index:
            assert not bert_tok[i].startswith('##')
            if bert_tok[i] == tok[index]:
                position.append(index)
                index += 1
            else:
                assert tok[index].startswith(bert_tok[i])
                position.append(index)
                tmp_size = len(bert_tok[i])
        else:
            if bert_tok[i].startswith('##'):
                bert_tok[i] = bert_tok[i][2:]
            assert tok[index][tmp_size:].startswith(bert_tok[i])
            position.append(index)
            tmp_size += len(bert_tok[i])
            if tmp_size == len(tok[index]):
                index += 1
                tmp_size = 0
        i += 1
    assert index == len(tok)
    assert ''.join(bert_tok) == ''.join(tok)
    return position


class ConceptData(Dataset):
    #TODO: 0: None,  1: Begin-Concept  2: In-Concept
    def __init__(self, path, split, tokenizer, sep=False):
        with open(os.path.join(path, '{}.pkl'.format(split)), 'rb') as file:
            query = pickle.load(file)
        self.query = query
        self.sep = sep
        self.tokenizer = tokenizer
        self.sent, self.table_label, self.column_label, \
        self.table_indexs, self.column_indexs = self.process_data()
        assert len(self.sent) == len(self.column_label) == len(self.table_label)
        print(len(self.sent))
        assert all([len(s) == len(c) == len(t) for s, c, t in
                    zip(self.sent, self.column_label, self.table_label)])

    def process_data(self):
        sents = []
        table_labels = []
        column_labels = []
        table_labels_index = []
        column_labels_index = []
        for q in self.query:
            nl_tok = q['processed_question_toks']
            table_label, column_label = q['schema_linking']
            nl = ' '.join(nl_tok)
            bert_nl_tok = self.tokenizer.tokenize(nl)
            if self.sep:
                sents.append(self.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + bert_nl_tok + [SEP_TOKEN]))
            else:
                sents.append(self.tokenizer.convert_tokens_to_ids(bert_nl_tok))
            position = link(bert_nl_tok, nl_tok)
            if self.sep:
                new_table_label = [-100] + [1 if table_label[i] > 0 else 0 for i in position] + [-100]
                new_column_label = [-100] + [1 if column_label[i] > 0 else 0 for i in position] + [-100]
            else:
                new_table_label = [1 if table_label[i] > 0 else 0 for i in position]
                new_column_label = [1 if column_label[i] > 0 else 0 for i in position]
            #TODO:
            '''
            new_table_label = []
            for i, x in enumerate(position):
                if table_label[x] > 0:
                    if i == 0 or :
                        new_table_label.append(2)
                    else:
                        new_table_label.append(1)
                else:
                    new_table_label.append(0)
            '''
            table_labels.append(torch.as_tensor(new_table_label, dtype=torch.int64))
            column_labels.append(torch.as_tensor(new_column_label, dtype=torch.int64))
            if self.sep:
                table_labels_index.append([0] + [table_label[i] for i in position] + [0])
                column_labels_index.append([0] + [column_label[i] for i in position] + [0])
            else:
                table_labels_index.append([table_label[i] for i in position])
                column_labels_index.append([column_label[i] for i in position])
        return sents, table_labels, column_labels, table_labels_index, column_labels_index

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, item):
        return self.sent[item], self.table_label[item], self.column_label[item]


def collate(data):
    sent = [x[0] for x in data]
    table_label = [x[1] for x in data]
    column_label = [x[2] for x in data]
    max_len = max([len(x) for x in sent])
    length = [len(x) for x in sent]
    new_sent = []
    new_table_label = []
    new_column_label = []
    for s, t, c in zip(sent, table_label, column_label):
        new_sent.append(s + [0] * (max_len - len(s)))
        new_table_label.append(list(t) + [-100] * (max_len - len(t)))
        new_column_label.append(list(c) + [-100] * (max_len - len(c)))
    return torch.tensor(new_sent), \
           torch.LongTensor(new_table_label), \
           torch.LongTensor(new_column_label), torch.tensor(length)


class RelationData(ConceptData):
    # 0: No RelationShip, 1: Same, 2: Have Relationship
    def __init__(self, path, split, tokenizer, sep=False):
        super(RelationData, self).__init__(path, split, tokenizer, sep=sep)
        self.concept_start_index = self.tokenizer.convert_tokens_to_ids(
            ['[concept_start]'])[0]
        self.concept_end_index = self.tokenizer.convert_tokens_to_ids(
            ['[concept_end]'])[0]
        with open(os.path.join(path, 'database.pkl'), 'rb') as file:
            databases = pickle.load(file)
        self.databases = databases
        db2span = defaultdict(list)
        for i, (s, q, t_ind, c_ind) in enumerate(
                zip(self.sent, self.query, self.table_indexs, self.column_indexs)):
            db2span[q['db_id']] += label2span(t_ind, index=i, ty=0) + \
                                   label2span(c_ind, index=i, ty=1)
        self.span_pairs = []
        relationConcept = []
        noRelationConcept = []
        self.db2data = defaultdict(dict)
        def safe(s1, s2):
            if s1.start_index > s2.start_index:
                s1, s2 = s2, s1            
            return s1.end_index <= s2.start_index
        for k, v in db2span.items():
            tmpConcept = defaultdict(list)
            for span in v:
                concept = self.span2concept(self.databases[k], span)
                span.concept = concept
                tmpConcept[concept].append(span)
            self.db2data[k]['noRelation'] = []
            for concept, spans in tmpConcept.items():
                for s1 in spans:
                    for s2 in spans:
                        if s1 != s2 and s1.sent_id == s2.sent_id:
                            self.db2data[k]['noRelation'].append((s1, s2, 0))
            noRelationConcept += self.db2data[k]['noRelation']
            ''' 
            def safe(s1, s2):
                if s1.start_index < s2.start_index:
                    s1, s2 = s2, s1
                return s1.end_index <= s2.start_index
            '''
            self.db2data[k]['relation'] = []
            for concept1, spans1 in tmpConcept.items():
                for concept2, spans2 in tmpConcept.items():
                    if concept1 == concept2: continue
                    relation = self.get_relation(self.databases[k], concept1, concept2)
                    if relation == 2:
                        for s1 in spans1:
                            for s2 in spans2:
                                if s1.sent_id == s2.sent_id and safe(s1, s2):
                                    self.db2data[k]['relation'].append((s1, s2, 1))
                    else:
                        assert relation == 0
                        for s1 in spans1:
                            for s2 in spans2:
                                if s1.sent_id == s2.sent_id and safe(s1, s2):
                                    self.db2data[k]['noRelation'].append((s1, s2, 0))
            noRelationConcept += self.db2data[k]['noRelation']
            relationConcept += self.db2data[k]['relation']
        random.shuffle(relationConcept)
        random.shuffle(noRelationConcept)
        self.span_pairs = relationConcept + noRelationConcept
        print(len(relationConcept), len(noRelationConcept))
        random.shuffle(self.span_pairs)

    def get_relation(self, db, concept1, concept2):
        if concept1 == concept2:
            return 1
        if isinstance(concept2, Table):
            concept1, concept2 = concept2, concept1
        if isinstance(concept1, Table) and isinstance(concept2, Table):
            assert db.erm.is_entity(concept1) and db.erm.is_entity(concept2)
            if db.erm.get_table_relation(concept1, concept2):
                return 2
            else:
                return 0
        elif isinstance(concept1, Table) and isinstance(concept2, Column):
            assert db.erm.is_entity(concept1) and db.erm.is_attr(concept2)
            if db.erm.get_table_column_relation(concept1, concept2):
                return 2
            else:
                return 0
        else:
            assert isinstance(concept1, Column) and isinstance(concept2, Column)
            return 0

    def span2concept(self, db, span):
        ty = span.type
        concept_num = span.concept_index - 1
        concept = db.get_table(concept_num) \
            if ty == 0 else db.get_column(concept_num)
        return concept

    def __len__(self):
        return len(self.span_pairs)
    
    def __getitem__(self, item):
        span1, span2, relation = self.span_pairs[item]
        assert self.sent[span1.sent_id] == self.sent[span2.sent_id]
        sent = self.sent[span1.sent_id]
        if span1.start_index > span2.start_index: span1, span2 = span2, span1
        sent  = span2merged_sent(span1, span2, sent,self.concept_start_index, self.concept_end_index)
        return sent, span1.start_index, span1.type, span2.start_index + 2, span2.type, relation

    def getDBdata(self):
        for k, v in self.db2data.items():
            same = v['same']
            noRelation = v['noRelation']
            relation = v['relation']
            data = same + noRelation + relation
            ret = []
            for span1, span2, r in data:
                sent1, sent2 = self.sent[span1.sent_id], self.sent[span2.sent_id]
                sent = span2merged_sent(span1, span2, sent1, sent2,
                                        self.concept_start_index, self.concept_end_index)
                ret.append((sent, span1.start_index, span1.type,
                            span2.start_index + len(sent1) - 1, span2.type, r, span1, span2))
            yield ret

    def f(self, item):
        span1, span2, relation = self.span_pairs[item]

        sent1, sent2 = self.sent[span1.sent_id], self.sent[span2.sent_id]
        
        sent1 = sent1[:span1.start_index] + \
                [self.concept_start_index] + \
                sent1[span1.start_index: span1.end_index] + \
                [self.concept_end_index] + \
                sent1[span1.end_index:]

        sent2 = sent2[:span2.start_index] + \
                [self.concept_start_index] + \
                sent2[span2.start_index: span2.end_index] + \
                [self.concept_end_index] + \
                sent2[span2.end_index:]
        '''
        sent1 = [sent1[0]] + sent1[span1.start_index: span1.end_index] + [sent1[-1]]
        sent2 = [sent2[0]] + sent2[span2.start_index: span2.end_index] + [sent2[-1]]
        assert sent1[0] == sent2[0] and sent1[-1] == sent2[-1]
        def getRelation(s1, s2, t1, t2):
            if t1 == t2:
                if s1 == s2:
                    return 1
                else:
                    return 0
                tmp1 = s1[1:-1]
                tmp2 = s2[1:-1]
                flag = False
                for x in tmp1:
                    if x in tmp2:
                        flag = True
                for x in tmp2:
                    if x in tmp1:
                        flag = True
                if flag: 
                    return 1
                else:
                    return 0
            else:
                return 0
        '''
        return sent1, span1.start_index, span1.type, \
               sent2, span2.start_index, span2.type, relation
        '''
        #relation = getRelation(sent1, sent2, span1.type, span2.type)
        return sent1, 0, span1.type, sent2, 0, span2.type, relation
        '''


def relation_collate(data):
    sent = [x[0] for x in data]
    span_start1 = [x[1] for x in data]
    type1 = [x[2] for x in data]
    span_start2 = [x[3] for x in data]
    type2 = [x[4] for x in data]
    relation = [x[5] for x in data]
    max_len = max([len(x) for x in sent])
    length = [len(x) for x in sent]
    new_sent = []
    for s in sent:
        new_sent.append(s + [0] * (max_len - len(s)))
    return torch.tensor(new_sent), torch.tensor(length), \
           torch.tensor(span_start1), \
           torch.tensor(type1), \
           torch.tensor(span_start2), \
           torch.tensor(type2), \
           torch.tensor(relation)

def old_relation_collate(data):
    sent1 = [x[0] for x in data]
    span_start1 = [x[1] for x in data]
    type1 = [x[2] for x in data]
    sent2 = [x[3] for x in data]
    span_start2 = [x[4] for x in data]
    type2 = [x[5] for x in data]
    relation = [x[6] for x in data]
    max_len1 = max([len(x) for x in sent1])
    length1 = [len(x) for x in sent1]
    max_len2 = max([len(x) for x in sent2])
    length2 = [len(x) for x in sent2]
    new_sent1 = []
    new_sent2 = []
    for s in sent1:
        new_sent1.append(s + [0] * (max_len1 - len(s)))
    for s in sent2:
        new_sent2.append(s + [0] * (max_len2 - len(s)))
    return torch.tensor(new_sent1), torch.tensor(span_start1), \
           torch.tensor(type1), torch.tensor(length1), \
           torch.tensor(new_sent2), torch.tensor(span_start2), \
           torch.tensor(type2), torch.tensor(length2), \
           torch.tensor(relation)


class EvaluateData:
    def __init__(self, path, split, tokenizer):
        with open(os.path.join(path, '{}.pkl'.format(split)), 'rb') as file:
            query = pickle.load(file)
        self.query = query
        self.tokenizer = tokenizer
        for q in self.query:
            nl_tok = q['processed_question_toks']
            nl = ' '.join(nl_tok)
            bert_nl_tok = self.tokenizer.tokenize(nl)
            q['bert_idx'] = self.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + bert_nl_tok + [SEP_TOKEN])
        with open(os.path.join(path, 'database.pkl'), 'rb') as file:
            databases = pickle.load(file)
        db2query = defaultdict(list)
        for q in self.query:
            db2query[q['db_id']].append(q)
        self.data = []
        total_token = 0
        total_sent = 0
        for db_id, queries in db2query.items():
            length = [len(q['bert_idx']) for q in queries]
            max_len = max(length)
            length = torch.tensor(length)
            sents = []
            for q in queries:
                s = q['bert_idx']
                sents.append(s + [0] * (max_len - len(s)))
            sents = torch.tensor(sents)
            total_token += length.sum()
            total_sent += len(length)
            self.data.append((sents, length, databases[db_id]))
        print(len(self.data), total_token / len(self.data), total_sent)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sents, length, db in self.data:
            yield sents, length, db
