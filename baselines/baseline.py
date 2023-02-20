import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import stanza
from collections import defaultdict
nlp = stanza.Pipeline('en')

lemmatizer = WordNetLemmatizer()
NOUN = ['NN', 'NNP', 'NNS']
VERB = ['VBP', 'VB', 'VBN', 'VBZ']

def convertNoun(n):
    return lemmatizer.lemmatize(n.lower())

def nl2erm_baseline1(nl): # ER-Gen
    E = set()
    A = set()
    R = set()
    for sent in nl:
        if isinstance(sent, str):
            sent = word_tokenize(sent)
            #print(sent)
        pos = nltk.pos_tag(sent)
        entitySet = set()
        attributeSet = set()
        #print(pos)
        tmp = []
        i = 0
        while i < len(pos):
            word, p = pos[i]
            if p not in NOUN:
                tmp.append((word, p))
                i += 1
                continue
            tmp_n = word
            i += 1
            while i < len(pos):
                if pos[i][1] in NOUN:
                    tmp_n += ' ' + pos[i][0]
                    i += 1
                else:
                    break
            tmp.append((tmp_n, p))
        pos = tmp
        #print(pos)
        for i, (word, p) in enumerate(pos):
            if p in NOUN:
                if i + 2 < len(pos):
                    if pos[i + 1][0] in [',', 'and'] and pos[i + 2][1] in NOUN:
                        attributeSet.add(convertNoun(word))
                        continue
                if i + 3 < len(pos):
                        if pos[i + 1][0] == ',' and pos[i + 2][0] == 'and' \
                                and pos[i + 3][1] in NOUN:
                            attributeSet.add(convertNoun(word))
                            continue
                if i - 2 >= 0:
                    if pos[i - 1][0] in [',', 'and'] and pos[i - 2][1] in NOUN:
                        attributeSet.add(convertNoun(word))
                        continue
                if i - 3 >= 0:
                    if pos[i - 1][0] == 'and' and pos[i - 2][0] == ',' \
                        and pos[i - 3][1] in NOUN:
                        attributeSet.add(convertNoun(word))
                        continue
                #noun_list.append((lemmatizer.lemmatize(word), i))
                entitySet.add(convertNoun(word))
        relationSet = set()
        for e1 in entitySet:
            for e2 in entitySet:
                if e1 != e2 and (e2, e1) not in relationSet and (e1, e2) not in relationSet:
                    relationSet.add((e1, e2))
        for e in entitySet:
            for a in attributeSet:
                relationSet.add((e, a))
        E.update(entitySet)
        A.update(attributeSet)
        R.update(relationSet)
        #print(entitySet)
        #print(attributeSet)
        #print(relationSet)
        #print('*' * 80)
    print(E)
    print(A)
    print(R)
    return E, A, R



def nl2erm_baseline2(nl): #Scenario baseline
    E = set()
    A = set()
    R = set()
    for sent in nl:
        entitySet = set()
        attributeSet = set()
        if isinstance(sent, list): sent = ' '.join(sent)
        doc = nlp(sent)
        toks = [w.text.lower() for s in doc.sentences for w in s.words]
        pos_tags = []
        for s in doc.sentences:
            for word in s.words:
                flag = False
                for ent in s.ents:
                    if word.start_char >= ent.start_char and word.end_char <= ent.end_char:
                        flag = True
                        break
                if not flag:
                    pos_tags.append(word.xpos)
                else:
                    pos_tags.append('ProperNoun')
        #print(toks)
        #print(pos_tags)
        pos = [(w, p) for w, p in zip(toks, pos_tags)]
        tmp = []
        i = 0
        while i < len(pos):
            #print(i, pos[i])
            word, p = pos[i]
            if p[0] not in NOUN:
                tmp.append((word, p))
                i += 1
                continue
            tmp_n = word
            i += 1
            while i < len(pos):
                if pos[i][1][0] in NOUN:
                    tmp_n += ' ' + pos[i][0]
                    i += 1
                else:
                    break
            tmp.append((tmp_n, p))
        pos = tmp
        label = []
        i = 0
        relation_map = defaultdict(dict)
        def left_noun(j):
            j -= 1
            while j >= 0:
                if pos[j][1] in NOUN:
                    return j
                j -= 1
            return j
        def right_noun(j):
            j += 1
            while j < len(pos):
                if pos[j][1] in NOUN:
                    return j
                j += 1
            return j
        while i < len(pos):
            word = pos[i][0]
            p = pos[i][1]
            if p in NOUN:
                if i > 0 and pos[i - 1][0] in ['each', 'Each']:
                    entitySet.add(convertNoun(word))
                    label.append('entity')
                    i += 1
                    continue

                if i + 2 < len(pos):
                    if pos[i + 1][0] in [',', 'and'] and pos[i + 2][1] in NOUN:
                        label.append('attr')
                        attributeSet.add(convertNoun(word))
                        i += 1
                        continue
                if i + 3 < len(pos):
                    if pos[i + 1][0] == ',' and pos[i + 2][0] == 'and' \
                            and pos[i + 3][1] in NOUN:
                        label.append('attr')
                        attributeSet.add(convertNoun(word))
                        i += 1
                        continue
                if i - 2 >= 0:
                    if pos[i - 1][0] in [',', 'and'] and pos[i - 2][1] in NOUN:
                        label.append('attr')
                        attributeSet.add(convertNoun(word))
                        i += 1
                        continue
                if i - 3 >= 0:
                    if pos[i - 1][0] == 'and' and pos[i - 2][0] == ',' \
                            and pos[i - 3][1] in NOUN:
                        label.append('attr')
                        attributeSet.add(convertNoun(word))
                        i += 1
                        continue
                #noun_list.append((lemmatizer.lemmatize(word), i))
                if convertNoun(word) not in ['record', 'system', 'information', 'organization']:
                    label.append('entity')
                    entitySet.add(convertNoun(word))
                    i += 1
                    continue

            elif word in ['has', 'have']:
                label.append('false')
                start_p = left_noun(i)
                i += 1
                while i < len(pos):
                    if pos[i][1] in NOUN:
                        label.append('attr')
                        attributeSet.add(convertNoun(pos[i][0]))
                        relation_map[start_p][i] = 1
                    else:
                        label.append('false')
                    i += 1
                continue
            elif word == 'of':
                if i + 2 < len(pos):
                    if pos[i + 1][0] == 'the' and pos[i + 2][1] in NOUN:
                        ent = right_noun(i)
                        label += ['false', 'false', 'entity']
                        entitySet.add(convertNoun(pos[i + 2][0]))
                        j = i - 1
                        while j >= 0:
                            if label[j] == 'attr':
                                relation_map[ent][j] = 1
                            if pos[j][1] in VERB:break
                            j -= 1
                        i += 3
                        continue
            elif p in VERB:
                relation_map[left_noun(i)][right_noun(i)] = 1
            label.append('false')
            i += 1
        assert len(label) == len(pos)
        relationSet = set()
        for k1, v1 in relation_map.items():
            for k2, v2 in v1.items():
                if k1 >= 0 and k1 < len(pos) and k2 >= 0 and k2 < len(pos):
                    if label[k1] in ['entity', 'attr'] and label[k2] in ['entity', 'attr']:
                        if not(label[k1] == 'attr' and label[k2] == 'attr'):
                            relationSet.add((convertNoun(pos[k1][0]), convertNoun(pos[k2][0])))
        E.update(entitySet)
        A.update(attributeSet)
        R.update(relationSet)
    print(E)
    print(A)
    print(R)
    return E, A, R

if __name__ == '__main__':
    sentList = [
        'A library contains libraries, books, authors and patrons.',
        'Libraries are described by library name and location.',
        'Books are described by title and pages.',
        'Authors are described by author name.',
        'Patrons are described by patron name and patron weight.',
        'A library can hold many books.',
        'The book can appear in many libraries.',
    ]
    #5/8 5/7
    #
    #for sent in sentList:
    #    nl2erm_baseline2([sent])
    #    print('*' * 80)
    nl2erm_baseline2(sentList)