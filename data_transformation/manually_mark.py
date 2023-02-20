import math
def size(x):
    if x == 0: return 1
    return math.floor(math.log10(x)) + 1

class ManuallyMark:
    def __init__(self):
        self.pkl_file = open('manually_mark.pkl', 'wb')
        self.write_file = open('manually_mark.txt', 'w')
        self.obj = []

    def write_manually(self, query, tables, columns):
        self.obj.append((query, tables, columns))
        tables, table_index = [t[0] for t in tables], [t[1] for t in tables]
        table_names = [t.processed_name for t in tables]
        columns, column_index = [c[0] for c in columns], [c[1] for c in columns]
        column_names = [c.processed_name for c in columns]

        self.write_file.write(query['query'] + '\n')
        self.write_file.write(str([(name, index + 1) for name, index in zip(table_names, table_index)]) + '\n')
        self.write_file.write(str([(name, index + 1) for name, index in zip(column_names, column_index)]) + '\n')
        self.write_file.write(str(query['processed_question_toks']) + '\n')
        self.write_file.write('  ')
        for x, y in zip(query['processed_question_toks'], query['schema_linking'][0]):
            self.write_file.write(str(int(y)))
            self.write_file.write(' ' *  ((len(x) + 4 - size(y))))
        self.write_file.write('\n')
        self.write_file.write('  ')
        for x, y in zip(query['processed_question_toks'], query['schema_linking'][1]):
            self.write_file.write(str(int(y)))
            self.write_file.write(' ' * ((len(x) + 4 - size(y))))
        self.write_file.write('\n\n\n')

    def dump(self):
        import pickle
        pickle.dump(self.obj, self.pkl_file)

    def close(self):
        self.pkl_file.close()
        self.write_file.close()