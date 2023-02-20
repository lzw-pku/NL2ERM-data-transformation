from model.dataset import EvaluateData, ConceptData, collate
from model.model import PipeLine, ExtractConcept
import torch
import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from model.model_utils import label2span
from evaluate.utils import ngram_similarity, match


def evaluate_extractor(model_type='base', sep=False):
    tokenizer = BertTokenizer.from_pretrained('bert-{}-uncased'.format(model_type))
    tokenizer.add_tokens('[Concept_Start]')
    tokenizer.add_tokens('[Concept_End]')
    batch_size = 16
    dev_data = DataLoader(ConceptData('data', 'generated_dev_data', tokenizer, sep=sep), batch_size=batch_size,
                          shuffle=False, collate_fn=collate)
    model = ExtractConcept(model_type=model_type)
    model.cuda()
    extractor_dict = torch.load('saved/extractor.pt')
    model.load_state_dict(extractor_dict['model'])
    model.eval()
    tp1, tn1, fp1, fn1 = 0, 0, 0, 0
    tp2, tn2, fp2, fn2 = 0, 0, 0, 0

    def f(pred, gt):
        score_map = {t1: {t2: 0 for t2 in gt} for t1 in pred}
        for i, span1 in enumerate(pred):
            for j, span2 in enumerate(gt):
                ngram1 = ' '.join(tokenizer.convert_ids_to_tokens(
                    sent[i][span1.start_index: span1.end_index]))
                ngram2 = ' '.join(tokenizer.convert_ids_to_tokens(
                    sent[i][span2.start_index: span2.end_index]))
                score_map[span1][span2] = ngram_similarity(ngram1, ngram2)
        score, m = match(score_map)
        #print(score, m)
        tmp = 0
        for k, v in m.items():
            if score_map[k][v] > 0.33:
                tmp += 1
        return tmp, len(pred) - tmp, len(gt) - tmp

    for sent, table_label, column_label, length in dev_data:
        sent = sent.cuda()
        table_label = table_label.cuda()
        column_label = column_label.cuda()
        length = length.cuda()
        _, table_preds, column_preds = model.predict(sent, length)
        #tp, tn, fp, fn = 0, 0, 0, 0
        for i, (t_pred, c_pred, leng, t_gt, c_gt) in enumerate(zip(table_preds, column_preds, length,
                                                                   table_label, column_label)):
            t_pred = t_pred.tolist()[:leng]
            c_pred = c_pred.tolist()[:leng]
            table_spans = label2span(t_pred, i, 0)
            column_spans = label2span(c_pred, i, 1)
            gt_table_spans = label2span(t_gt, i, 0)
            gt_column_spans = label2span(c_gt, i, 1)
            #print(table_spans)
            #print(gt_table_spans)
            #exit(0)
            '''
            score_map = {t1: {t2: 0 for t2 in gt_table_spans} for t1 in table_spans}
            for i, span1 in enumerate(table_spans):
                for j, span2 in enumerate(gt_table_spans):
                    ngram1 = ' '.join(tokenizer.convert_ids_to_tokens(
                        sent[i][span1.start_index: span1.end_index]))
                    ngram2 = ' '.join(tokenizer.convert_ids_to_tokens(
                        sent[i][span2.start_index: span2.end_index]))
                    score_map[span1][span2] = ngram_similarity(ngram1, ngram2)
            score, m = match(score_map)
            #print(score, m)
            tmp = 0
            for k, v in m.items():
                if score_map[k][v] > 0.33:
                    tmp += 1
            '''
            s1, s2, s3 = f(table_spans, gt_table_spans)
            tp1 += s1
            fp1 += s2
            fn1 += s3

            s1, s2, s3 = f(column_spans, gt_column_spans)
            tp2 += s1
            fp2 += s2
            fn2 += s3
            #print(tp, fp, fn)
            #print('*'*80)
    precision = tp1 / (tp1 + fp1)
    recall = tp1 / (tp1 + fn1)
    f = 2 * precision * recall / (precision + recall)
    print(tp1 / (tp1 + fp1), tp1 / (tp1 + fn1), f)
    print(tp2 / (tp2 + fp2), tp2 / (tp2 + fn2))


def evaluate_pipeline():
    threshold = 0.5
    alpha = 0.5
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens('[Concept_Start]')
    tokenizer.add_tokens('[Concept_End]')
    train_data = EvaluateData('data', 'generated_train_data', tokenizer)
    dev_data = EvaluateData('data', 'generated_dev_data', tokenizer)
    start_index = tokenizer.convert_tokens_to_ids(['concept_start'])[0]
    end_index = tokenizer.convert_tokens_to_ids(['concept_end'])[0]
    model = PipeLine(tokenizer=tokenizer,
                     concept_start_index=start_index,
                     concept_end_index=end_index,
                     threshold=threshold,
                     alpha=alpha)
    model.cuda()
    extractor_dict = torch.load('saved/extractor.pt')
    classifier_dict = torch.load('saved/large_data_classifier.pt')
    model.load_state_dict(state_dict={'extractor': extractor_dict['model'],
                                      'classifier': classifier_dict['model']})
    model.eval()
    for sent, length, db in dev_data:
        sent = sent.cuda()
        length = length.cuda()
        ret = model(sent, length)
        #import pickle
        #with open('../pred.pkl', 'wb') as file:
        #    pickle.dump((ret, db), file)
        #exit(0)

def evaluate_baseline(model_type='base', sep=False):
    from evaluate.baseline import nl2erm
    from model.model_utils import Span
    tokenizer = BertTokenizer.from_pretrained('bert-{}-uncased'.format(model_type))
    tokenizer.add_tokens('[Concept_Start]')
    tokenizer.add_tokens('[Concept_End]')
    batch_size = 16
    dev_data = DataLoader(ConceptData('data', 'generated_dev_data', tokenizer, sep=sep), batch_size=batch_size,
                          shuffle=False, collate_fn=collate)

    tp1, tn1, fp1, fn1 = 0, 0, 0, 0
    tp2, tn2, fp2, fn2 = 0, 0, 0, 0

    def f(pred, gt):
        score_map = {t1: {t2: 0 for t2 in gt} for t1 in pred}
        for i, span1 in enumerate(pred):
            for j, span2 in enumerate(gt):
                ngram1 = span1
                ngram2 = span2
                score_map[span1][span2] = ngram_similarity(ngram1, ngram2)
                print(ngram1, ngram2, score_map[span1][span2])
        score, m = match(score_map, threshold=0.3)
        tmp = 0
        for k, v in m.items():
            if score_map[k][v] > 0.32:
                tmp += 1
        #print('!!!', tmp)
        return tmp, len(pred) - tmp, len(gt) - tmp

    for sent, table_label, column_label, length in dev_data:
        #tp, tn, fp, fn = 0, 0, 0, 0
        for i, (s, l, t_gt, c_gt) in enumerate(zip(sent, length, table_label, column_label)):
            #print(s)
            E, A, R = nl2erm([tokenizer.convert_ids_to_tokens(s[:l])])
            gt_table_spans = label2span(t_gt, i, 0)

            gt_table_spans = [' '.join(tokenizer.convert_ids_to_tokens(
                s[span.start_index:span.end_index])) for span in gt_table_spans]
            gt_column_spans = label2span(c_gt, i, 1)
            gt_column_spans = [' '.join(tokenizer.convert_ids_to_tokens(
                s[span.start_index:span.end_index])) for span in gt_column_spans]
            #print(gt_table_spans)
            table_spans = E
            column_spans = A
            #print(table_spans)
            #exit(0)
            #print(table_spans)
            #print(tokenizer.convert_ids_to_tokens(s[:l]))
            #print(gt_table_spans)

            s1, s2, s3 = f(table_spans, gt_table_spans)
            tp1 += s1
            fp1 += s2
            fn1 += s3

            s1, s2, s3 = f(column_spans, gt_column_spans)
            tp2 += s1
            fp2 += s2
            fn2 += s3
            #print(tp1, fp1, fn1, tp2, fp2, fn2)
            #if i == 3:exit(0)
    precision = tp1 / (tp1 + fp1)
    recall = tp1 / (tp1 + fn1)
    f = 2 * precision * recall / (precision + recall)
    print(tp1 / (tp1 + fp1), tp1 / (tp1 + fn1), f)
    print(tp2 / (tp2 + fp2), tp2 / (tp2 + fn2))

if __name__ == '__main__':
    evaluate_baseline()
