from collections import defaultdict
from asdl.asdl import ASDLGrammar
from asdl.transition_system import TransitionSystem
from utils.database_util import Database
import pickle

with open('data/tables.pkl', 'rb') as file:
    db_data = pickle.load(file)

with open('data/train.pkl', 'rb') as file:
    querys = pickle.load(file)

with open('data/dev.pkl', 'rb') as file:
    dev_querys = pickle.load(file)

db2query = defaultdict(list)
for q in querys:
    db2query[q['db_id']].append(q['query'])

grammar = ASDLGrammar.from_filepath(file_path='./asdl/sql/grammar/sql_asdl_v2.txt')
translator = TransitionSystem.get_class_by_lang('sql')(grammar)

databases = {}

for db_id, db in db_data.items():
    databases[db_id] = Database.generate_from_dict(db)
    databases[db_id].database2er()

grammar = ASDLGrammar.from_filepath(file_path='./asdl/sql/grammar/sql_asdl_v2.txt')
translator = TransitionSystem.get_class_by_lang('sql')(grammar)

s1 = set([q['db_id'] for q in querys])
s2 = set([q['db_id'] for q in dev_querys])
assert all([x not in s2 for x in s1]) and all([x not in s1 for x in s2])
for q in querys + dev_querys:
    databases[q['db_id']].add_query(q, translator)
print('Schema linking finished')


with open('data/database.pkl', 'wb') as file:
    pickle.dump(databases, file)

with open('data/generated_train_data.pkl', 'wb') as file:
    query = []
    for db_id in s1:
        query += databases[db_id].querys
    pickle.dump(query, file)

with open('data/generated_dev_data.pkl', 'wb') as file:
    query = []
    for db_id in s2:
        query += databases[db_id].querys
    pickle.dump(query, file)