import pickle
import json
from utils.schema_linking import preprocess_question, preprocess_database

with open('data/spider/train_spider.json') as file:
    train_data = json.load(file)
for entry in train_data:
    preprocess_question(entry)
with open('data/train.pkl', 'wb') as file:
    pickle.dump(train_data, file)

with open('data/spider/dev.json') as file:
    dev_data = json.load(file)
for entry in dev_data:
    preprocess_question(entry)
with open('data/dev.pkl', 'wb') as file:
    pickle.dump(dev_data, file)

with open('data/spider/tables.json') as file:
    db = json.load(file)
db = {entry['db_id']: entry for entry in db}
for k, v in db.items():
    preprocess_database(v)
with open('data/tables.pkl', 'wb') as file:
    pickle.dump(db, file)