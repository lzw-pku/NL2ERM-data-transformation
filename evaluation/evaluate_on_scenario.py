from model.dataset import EvaluateData
from transformers import BertTokenizer
from model.model import PipeLine
import torch
from nltk.corpus import stopwords
import stanza
import nltk


nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')
stanza_tokenizer = stanza.Pipeline('en', processors='tokenize')
stopwords = stopwords.words("english")
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'

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


def preprocess_question(nl):
    """ Tokenize, lemmatize, lowercase question"""
    # stanza tokenize, lemmatize and POS tag
    stanza_toks = stanza_tokenizer(nl)
    question = ' '.join(quote_normalization(stanza_toks))
    doc = nlp(question)
    #raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
    toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
    #pos_tags = [w.xpos for s in doc.sentences for w in s.words]
    return toks


def getData(nlset, tokenizer):
    lengths = []
    sents = []
    for nl in nlset:
        toks = preprocess_question(nl)
        nl = ' '.join(toks)
        bert_nl_tok = tokenizer.tokenize(nl)
        bert_index = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + bert_nl_tok + [SEP_TOKEN])
        length = len(bert_index)
        lengths.append(length)
        sents.append(bert_index)
    max_len = max(lengths)
    batch_sents = []
    for s in sents:
        batch_sents.append(s + [0] * (max_len - len(s)))
    lengths = torch.tensor(length)
    batch_sents = torch.tensor(batch_sents)
    return batch_sents, lengths


def test_relation():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.convert_ids_to_tokens()
    tokenizer.add_tokens('[Concept_Start]')
    tokenizer.add_tokens('[Concept_End]')
    start_index = tokenizer.convert_tokens_to_ids(['concept_start'])[0]
    end_index = tokenizer.convert_tokens_to_ids(['concept_end'])[0]
    model = PipeLine(tokenizer=tokenizer,
                     concept_start_index=start_index,
                     concept_end_index=end_index
                     )
    model.cuda()
    extractor_dict = torch.load('saved/extractor.pt')
    classifier_dict = torch.load('saved/classifier.pt')
    model.load_state_dict(state_dict={'extractor': extractor_dict['model'],
                                      'classifier': classifier_dict['model']})
    model.eval()
    with open('testset.txt', 'r') as file:
        nl = file.readlines()
    batch_sents, lengths = getData(nl, tokenizer)
    batch_sents = batch_sents.cuda()
    lengths = lengths.cuda()
    ret = model(batch_sents, lengths)
    print(ret)