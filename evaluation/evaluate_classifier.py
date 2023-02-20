from model.dataset import RelationData, relation_collate
from model.model import RelationClassifier
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from collections import defaultdict


if __name__ == '__main__':
    sep = True
    label_num = 3
    batch_size = 32
    model_type = 'large'
    tokenizer = BertTokenizer.from_pretrained('bert-{}-uncased'.format(model_type))
    tokenizer.add_tokens('[Concept_Start]')
    tokenizer.add_tokens('[Concept_End]')
    test_data = RelationData('../data', 'generated_dev_data', tokenizer, sep=sep)
    start_index = tokenizer.convert_tokens_to_ids(['concept_start'])[0]
    end_index = tokenizer.convert_tokens_to_ids(['concept_end'])[0]
    model = RelationClassifier(tokenizer=tokenizer, label_num=label_num, model_type=model_type)
    model.cuda()
    classifier_dict = torch.load('saved/classifier.pt')
    model.load_state_dict(classifier_dict['model'])
    model.eval()
    tot_loss = 0
    tot_data_num = 0
    tp1 = tn1 = fp1 = fn1 = 0
    tp2 = tn2 = fp2 = fn2 = 0

    dev_data = DataLoader(RelationData('../data', 'generated_dev_data', tokenizer, sep=sep), batch_size=batch_size,
                          shuffle=False, collate_fn=relation_collate)

    for sent, length, span_start1, type1, \
        span_start2, type2, relation in dev_data:
        sent = sent.cuda()
        span_start1 = span_start1.cuda()
        type1 = type1.cuda()
        length = length.cuda()
        span_start2 = span_start2.cuda()
        type2 = type2.cuda()
        relation = relation.cuda()
        logit, loss = model(sent, length, span_start1, type1,
                            span_start2, type2,
                            relation)
        tot_loss += loss.item()
        #for l, r in zip(logit, relation):
        pred = torch.argmax(logit, dim=-1)
        #pred = torch.tensor([1 if x[1] > 0.65 else 0 for x in logit], device=logit.device)
        #print(pred)
        pred_same = torch.tensor(pred == 1, dtype=torch.int, device=relation.device)
        pred_relation = torch.tensor(pred == 2, dtype=torch.int, device=relation.device)
        r_same = torch.tensor(relation == 1, dtype=torch.int, device=relation.device)
        r_relation = torch.tensor(relation == 2, dtype=torch.int, device=relation.device)

        tp1 += (pred_same * r_same).sum().item()
        tn1 += ((1 - pred_same) * (1 - r_same)).sum().item()
        fp1 += (pred_same * (1 - r_same)).sum().item()
        fn1 += ((1 - pred_same) * r_same).sum().item()


        tp2 += (pred_relation * r_relation).sum().item()
        tn2 += ((1 - pred_relation) * (1 - r_relation)).sum().item()
        fp2 += (pred_relation * (1 - r_relation)).sum().item()
        fn2 += ((1 - pred_relation) * r_relation).sum().item()


    precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) != 0 else 0
    recall1 = tp1 / (tp1 + fn1) if (tp1 + fn1) != 0 else 0
    f1_score1 = (2 * recall1 * precision1) / (precision1 + recall1) \
        if (precision1 + recall1) != 0 else 0
    precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) != 0 else 0
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) != 0 else 0
    f1_score2 = (2 * recall2 * precision2) / (precision2 + recall2) \
        if (precision2 + recall2) != 0 else 0
    print('Dev Loss:', tot_loss / len(dev_data))
    print('Same: F1:{} Precision:{} Recall:{}'.format(f1_score1, precision1, recall1), tp1, tn1, fp1, fn1)
    print('Relation: F1:{} Precision:{} Recall:{}'.format(f1_score2, precision2, recall2), tp2, tn2, fp2, fn2)
    exit(0)




    db2edge = []
    db2gt = []
    for db_data in test_data.getDBdata():
        tot = 0
        import random
        random.shuffle(db_data)
        print(len(db_data))
        edge = defaultdict(dict)
        gt = defaultdict(dict)
        tot_data_num += len(db_data)
        for i in range(0, len(db_data), batch_size):
            batch_data = db_data[i: i + batch_size]
            sent = [x[0] for x in batch_data]
            span_start1 = [x[1] for x in batch_data]
            type1 = [x[2] for x in batch_data]
            span_start2 = [x[3] for x in batch_data]
            type2 = [x[4] for x in batch_data]
            relation = [x[5] for x in batch_data]
            span1 = [x[6] for x in batch_data]
            span2 = [x[7] for x in batch_data]
            max_len = max([len(x) for x in sent])
            length = [len(x) for x in sent]
            new_sent = []
            for s in sent:
                new_sent.append(s + [0] * (max_len - len(s)))
            sent = torch.tensor(new_sent).cuda()
            length = torch.tensor(length).cuda()
            start1 = torch.tensor(span_start1).cuda()
            type1 = torch.tensor(type1).cuda()
            start2 = torch.tensor(span_start2).cuda()
            type2 = torch.tensor(type2).cuda()
            relation = torch.tensor(relation).cuda()

            logit, loss = model(sent, length, span_start1, type1,
                                span_start2, type2,
                                relation)
            tot_loss += loss.item()
            pred = torch.argmax(logit, dim=-1)


            pred_same = torch.tensor(pred == 1, dtype=torch.int, device=relation.device)
            pred_relation = torch.tensor(pred == 2, dtype=torch.int, device=relation.device)
            r_same = torch.tensor(relation == 1, dtype=torch.int, device=relation.device)
            r_relation = torch.tensor(relation == 2, dtype=torch.int, device=relation.device)

            tp1 += (pred_same * r_same).sum().item()
            tn1 += ((1 - pred_same) * (1 - r_same)).sum().item()
            fp1 += (pred_same * (1 - r_same)).sum().item()
            fn1 += ((1 - pred_same) * r_same).sum().item()

            tp2 += (pred_relation * r_relation).sum().item()
            tn2 += ((1 - pred_relation) * (1 - r_relation)).sum().item()
            fp2 += (pred_relation * (1 - r_relation)).sum().item()
            fn2 += ((1 - pred_relation) * r_relation).sum().item()

            relation = relation.cpu()
            for s1, s2, p, r in zip(span1, span2, pred, relation):
                edge[s1][s2] = p
                gt[s1][s2] = r
            tot+=1
            if tot == 20:break
        db2edge.append(edge)
        db2gt.append(gt)

    precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) != 0 else 0
    recall1 = tp1 / (tp1 + fn1) if (tp1 + fn1) != 0 else 0
    f1_score1 = (2 * recall1 * precision1) / (precision1 + recall1) \
        if (precision1 + recall1) != 0 else 0
    precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) != 0 else 0
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) != 0 else 0
    f1_score2 = (2 * recall2 * precision2) / (precision2 + recall2) \
        if (precision2 + recall2) != 0 else 0
    print('Dev Loss:', tot_loss / tot_data_num)
    print('Same: F1:{} Precision:{} Recall:{}'.format(f1_score1, precision1, recall1), tp1, tn1, fp1, fn1)
    print('Relation: F1:{} Precision:{} Recall:{}'.format(f1_score2, precision2, recall2), tp2, tn2, fp2, fn2)

    with open('saved/pred.pkl', 'wb') as file:
        import pickle
        pickle.dump((db2edge, db2gt), file)
