class Span:
    def __init__(self, sent_id, start_index, end_index, ty, concept_index):
        self.sent_id = sent_id
        self.start_index = start_index
        self.end_index = end_index
        self.type = ty
        self.concept_index = concept_index

    def __str__(self):
        return '(Span  sent_id:{}, start:{}, end:{}, type:{}, concept_index:{})'.format(
            self.sent_id, self.start_index, self.end_index, self.type, self.concept_index)

    def __repr__(self):
        return str(self)

def label2span(concept_label_index, index, ty):
    span_list = []
    flag = False
    start_index = 0
    concept_index = 0
    for i, x in enumerate(concept_label_index):
        if x > 0 and not flag:
            start_index = i
            concept_index = x
            flag = True
        elif x == 0 and flag:
            span_list.append(Span(index, start_index, i, ty, int(concept_index)))
            flag = False
        elif x > 0 and flag and x != concept_index:
            span_list.append(Span(index, start_index, i, ty, int(concept_index)))
            start_index = i
            concept_index = x
    if flag:
        i += 1
        span_list.append(Span(index, start_index, i, ty, int(concept_index)))
    return span_list


def span2merged_sent(span1, span2, sent,
                     concept_start_index, concept_end_index):
    assert span1.start_index < span2.start_index 

    tmp1 = sent[:span1.start_index] + \
            [concept_start_index] + \
            sent[span1.start_index: span1.end_index] + \
            [concept_end_index]

    tmp2 = sent[span1.end_index:span2.start_index] + \
            [concept_start_index] + \
            sent[span2.start_index: span2.end_index] + \
            [concept_end_index] + \
            sent[span2.end_index:]
    sent = tmp1 + tmp2
    #span2.start_index = span2.start_index + 2
    #print(sent)
    #print(span1.start_index, span2.start_index)
    #print(span1)
    #print(span2)
    assert sent[span1.start_index] == concept_start_index
    assert sent[span2.start_index + 2] == concept_start_index
    return sent
