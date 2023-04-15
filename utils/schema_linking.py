import pickle
import numpy as np
from itertools import product, combinations
from nltk.corpus import stopwords
import stanza
import json
import tqdm


nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
stopwords = stopwords.words("english")

def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\"" ]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question


def preprocess_database(db: dict):
    """ Tokenize, lemmatize, lowercase table and column names for each database """
    table_toks, table_names = [], []
    for tab in db['table_names']:
        doc = nlp(tab)
        tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
        table_toks.append(tab)
        table_names.append(" ".join(tab))

    db['processed_table_toks'], db['processed_table_names'] = table_toks, table_names
    column_toks, column_names = [], []
    for _, c in db['column_names']:
        doc = nlp(c)
        c = [w.lemma.lower() for s in doc.sentences for w in s.words]
        column_toks.append(c)
        column_names.append(" ".join(c))
    db['processed_column_toks'], db['processed_column_names'] = column_toks, column_names
    return db


def preprocess_question(entry: dict):
    """ Tokenize, lemmatize, lowercase question"""
    # stanza tokenize, lemmatize and POS tag
    question = ' '.join(quote_normalization(entry['question_toks']))
    doc = nlp(question)
    raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
    toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
    pos_tags = [w.xpos for s in doc.sentences for w in s.words]

    entry['raw_question_toks'] = raw_toks
    entry['processed_question_toks'] = toks
    entry['pos_tags'] = pos_tags
    return entry



def schema_linking(entry, tables, columns):
    """ Perform schema linking: both question and database need to be preprocessed """
    table_names, table_tokens, table_index = [t[0].processed_name for t in tables],\
                                             [t[0].processed_token for t in tables], \
                                             [t[1] for t in tables]
    column_names, column_tokens, column_index = [c[0].processed_name for c in columns],\
                                                [c[0].processed_token for c in columns],\
                                                [c[1] for c in columns]

    question_toks = entry['processed_question_toks']
    q_num = len(question_toks)
    table_tag = np.zeros(q_num)
    max_len = max([len(t) for t in table_tokens]) if table_tokens else 0
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len,
                              combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        if phrase in stopwords: continue
        for index, name in zip(table_index, table_names):
            if (phrase == name) or \
                    ((j - i == 1 and phrase in name.split()) or
                     (j - i > 1 and phrase in name)):
                '''
                if not all([table_tag[x] == 0 for x in range(i, j)]):
                    print(question_toks)
                    print(table_names)
                '''
                table_tag[range(i, j)] = index + 1

    column_tag = np.zeros(q_num)
    max_len = max([len(c) for c in column_tokens]) if column_tokens else 0
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len,
                              combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        if phrase in stopwords: continue
        for index, name in zip(column_index, column_names):
            if (phrase == name) or \
                    ((j - i == 1 and phrase in name.split()) or
                     (j - i > 1 and phrase in name)):
                column_tag[range(i, j)] = index + 1
    entry['schema_linking'] = (table_tag, column_tag)
    return entry

'''
def schema_linking(entry: dict, db: Database):
    """ Perform schema linking: both question and database need to be preprocessed """
    raw_question_toks, question_toks = entry['raw_question_toks'], entry['processed_question_toks']
    table_toks, column_toks = [table.processed_token for table in db.tables], []
    table_names, column_names = db['processed_table_names'], db['processed_column_names']
    q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'

    # relations between questions and tables, q_num*t_num and t_num*q_num
    q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
    max_len = max([len(t) for t in table_toks])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        if phrase in stopwords: continue
        for idx, name in enumerate(table_names):
            if phrase == name: # fully match will overwrite partial match due to sort
                q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
            elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'

    # relations between questions and columns
    q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
    max_len = max([len(c) for c in column_toks])
    index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
    index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
    for i, j in index_pairs:
        phrase = ' '.join(question_toks[i: j])
        if phrase in stopwords: continue
        for idx, name in enumerate(column_names):
            if phrase == name: # fully match will overwrite partial match due to sort
                q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
            elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
    # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
    q_col_mat[:, 0] = 'question-*-generic'
    q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
    entry['schema_linking'] = q_schema.tolist()

    return entry
'''


if __name__ == '__main__':
    with open('data/tables.json') as file:
        databases = json.load(file)
        processed_dbs = {}
        print('Start Processing Tables')
        for db in tqdm.tqdm(databases):
            db = preprocess_database(db)
            processed_dbs[db['db_id']] = db

    with open('data/tables.pkl', 'wb') as file:
        pickle.dump(processed_dbs, file)

    with open('data/train_spider.json') as file:
        train_data = json.load(file)
        processed_data = []
        print('Start Processing Examples')
        for example in tqdm.tqdm(train_data):
            example = preprocess_question(example)
            processed_data.append(example)

    with open('data/train.pkl', 'wb') as file:
        pickle.dump(processed_data, file)

    with open('data/dev.json') as file:
        dev_data = json.load(file)
        dev_processed_data = []
        print('Start Processing Examples')
        for example in tqdm.tqdm(dev_data):
            example = preprocess_question(example)
            dev_processed_data.append(example)

    with open('data/dev.pkl', 'wb') as file:
        pickle.dump(dev_processed_data, file)