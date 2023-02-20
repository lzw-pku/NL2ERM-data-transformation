from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
import pickle


def coverage(db_data, databases, translator):
    all_entity0, all_attr0, all_key_relation0, covered_entity0, covered_attr0, covered_have_relation0, \
    covered_key_relation0, linked_entity0, linked_attr0, cross_sent_have_relation0, cross_sent_key_relation0, \
    linked_have_relation0, linked_key_relation0 = [],[],[],[],[],[],[],[],[],[],[],[],[]
    for db_id, db in db_data.items():
        all_entity, all_attr, all_key_relation, covered_entity, covered_attr, covered_have_relation, \
        covered_key_relation, linked_entity, linked_attr, cross_sent_have_relation, cross_sent_key_relation, \
        linked_have_relation, linked_key_relation = databases[db_id].compute_coverage(translator)

        all_entity0 += all_entity
        all_attr0 += all_attr
        all_key_relation0 += all_key_relation
        covered_entity0 += covered_entity
        covered_attr0 += covered_attr
        covered_have_relation0 += covered_have_relation
        covered_key_relation0 += covered_key_relation
        linked_entity0 += linked_entity
        linked_attr0 += linked_attr
        cross_sent_have_relation0 += cross_sent_have_relation
        cross_sent_key_relation0 += cross_sent_key_relation
        linked_have_relation0 += linked_have_relation
        linked_key_relation0 += linked_key_relation

    covered_key_relation0 = covered_key_relation0[::2]
    cross_sent_key_relation0 = cross_sent_key_relation0[::2]
    linked_key_relation0 = linked_key_relation0[::2]

    print(len(all_entity0), len(all_attr0), len(all_attr0), len(all_key_relation0))
    print(len(covered_entity0), len(covered_attr0), len(covered_have_relation0), len(covered_key_relation0))
    print(len(linked_entity0), len(linked_attr0), len(cross_sent_have_relation0), len(cross_sent_key_relation0))
    print(len(linked_have_relation0), len(linked_key_relation0))

    print('*' * 80)
    print(len(covered_entity0)/len(all_entity0), len(covered_attr0)/len(all_attr0), len(covered_have_relation0)/len(all_attr0), len(covered_key_relation0)/len(all_key_relation0))
    print(len(linked_entity0)/len(covered_entity0), len(linked_attr0)/len(covered_attr0), len(cross_sent_have_relation0)/len(covered_have_relation0), len(cross_sent_key_relation0)/len(covered_key_relation0))
    print(len(linked_have_relation0)/len(cross_sent_have_relation0), len(linked_key_relation0)/len(cross_sent_key_relation0))

if __name__ == '__main__':
    grammar = ASDLGrammar.from_filepath(file_path='../asdl/sql/grammar/sql_asdl_v2.txt')
    translator = TransitionSystem.get_class_by_lang('sql')(grammar)
    with open('../data/database.pkl', 'rb') as file:
        databases = pickle.load(file)

    with open('../data/tables.pkl', 'rb') as file:
        db_data = pickle.load(file)

    coverage(db_data, databases, translator)