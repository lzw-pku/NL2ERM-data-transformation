from collections import defaultdict
from utils.erd_util import EntityType, RelationShip, ERModel
from utils.schema_linking import schema_linking
def convert_table_to_entity(table):
    entity = EntityType(table.name)
    for column in table.columns:
        entity.add_attribute(column.name)
    return entity


class Column:
    def __init__(self, name, origin_name, processed_column_name,
                 processed_column_token, type):
        assert type in ['text', 'number', 'time', 'boolean', 'others']
        self.name = name
        self.processed_name = processed_column_name
        self.processed_token = processed_column_token
        self.origin_name = origin_name
        self.type = type
        self.parent_table = None

    def set_parent_table(self, table):
        self.parent_table = table


class Table:
    def __init__(self, name, origin_name, processed_table_name,
                 processed_table_token, columns):
        self.name = name
        self.origin_name = origin_name
        self.processed_name = processed_table_name
        self.processed_token = processed_table_token
        self.columns = columns
        map(lambda x: x.set_parent_table(self), self.columns)
        self.primary_key = -1

    @property
    def size(self):
        return len(self.columns)
    #def set_foreign_key(self, foreign_keys):
    #    self.foreign_keys = foreign_keys


from asdl.asdl_ast import AbstractSyntaxTree
class Database:
    def __init__(self, db_id, tables, foreign_keys):
        self.db_id = db_id
        self.tables = tables
        self.name2table = {table.name: table for table in self.tables}
        self.foreign_keys = foreign_keys
        self.table_relations = defaultdict(list)
        for fk_set in self.foreign_keys:
            from_ks = [pair[0] for pair in fk_set]
            to_ks = [pair[1] for pair in fk_set]
            from_table = self.get_table_from_column_id(from_ks[0])
            to_table = self.get_table_from_column_id(to_ks[0])
            assert all([self.get_table_from_column_id(x) == from_table
                        for x in from_ks])
            assert all([self.get_table_from_column_id(x) == to_table
                        for x in to_ks])
            self.table_relations[from_table].append((to_table, from_ks, to_ks))
        self.querys = []
        #self.set_global_id()

    @staticmethod
    def generate_from_dict(db):
        column_list = [[] for _ in db['table_names']]

        assert len(db['column_types']) == len(db['column_names']) and \
               len(db['column_names']) == len(db['column_names_original']) and \
               len(db['column_names']) == len(db['processed_column_names']) and \
               len(db['column_names']) == len(db['processed_column_toks'])

        for name, origin, processed_column_name, processed_column_token, ctype \
                in zip(db['column_names'][1:], db['column_names_original'][1:],
                       db['processed_column_names'][1:], db['processed_column_toks'][1:],
                       db['column_types'][1:]):
            column_list[name[0]].append(Column(name[1], origin[1],
                                               processed_column_name,
                                               processed_column_token, ctype))

        assert len(column_list) == len(db['table_names']) and \
               len(column_list) == len(db['table_names_original']) and \
               len(column_list) == len(db['processed_table_names']) and \
               len(column_list) == len(db['processed_table_toks'])

        table_list = []
        for table_name, origin, processed_table_name, \
            processed_table_token, columns in zip(
                db['table_names'], db['table_names_original'],
                db['processed_table_names'], db['processed_table_toks'], column_list):
            table_list.append(Table(table_name, origin, processed_table_name,
                                    processed_table_token, columns))
        assert len(table_list) == len(db['table_names'])
        for primary_key in db['primary_keys']:
            index = primary_key[0]
            assert index > 0
            index -= 1
            for table in table_list:
                if index < table.size:
                    assert table.primary_key == -1
                    assert all([0 <= x - (primary_key[0] - index) < table.size
                                for x in primary_key])
                    table.primary_key = primary_key
                    break
                else:
                    index -= table.size
        return Database(db['db_id'], table_list, db['foreign_keys'])

    """    
    def set_global_id(self):
        total = 1
        for i, table in enumerate(self.tables):
            table.global_id = i
            for j, column in enumerate(table.columns):
                column.global_id = total + j
            total += len(table.columns)
    """

    def get_table(self, name):
        if isinstance(name, str):
            return self.name2table[name]
        else:
            assert isinstance(name, int) and name >= 0 and name < len(self.tables)
            return self.tables[name]

    def get_column(self, id):
        assert isinstance(id, int) and id > 0
        id -= 1
        for table in self.tables:
            if id < table.size:
                return table.columns[id]
            else:
                id -= table.size
        return None

    def get_table_from_column_id(self, id):
        assert isinstance(id, int) and id > 0
        id -= 1
        for i, table in enumerate(self.tables):
            if id < table.size:
                return table
            else:
                id -= table.size

    def table_to_index(self, table):
        return self.tables.index(table)

    def add_query(self, query, translator):
        filtered_tables_ids, filtered_columns_ids = \
            self.filter_table_and_column(query, translator)
        filtered_tables = [(self.get_table(index), index) for index in filtered_tables_ids]
        filtered_columns = [(self.get_column(index), index) for index in filtered_columns_ids]
        '''
        if not all([self.erm.is_entity(x[0]) for x in filtered_tables]):
            print('!!!', [(t[0].processed_name, t[1]) for t in filtered_tables])
        if not all([self.erm.is_attr(x[0]) for x in filtered_columns]):
            print('!!!', [(t[0].processed_name, t[1]) for t in filtered_columns])
        '''
        filtered_tables = list(filter(lambda x: self.erm.is_entity(x[0]), filtered_tables))
        filtered_columns = list(filter(lambda x: self.erm.is_attr(x[0]), filtered_columns))
        '''
        print(query['query'], [(t[0].processed_name, t[1]) for t in filtered_tables],
              [(c[0].name, c[1]) for c in filtered_columns])
        '''
        query = schema_linking(query, filtered_tables, filtered_columns)
        '''
        print([(tok, tid, cid) for tok, tid, cid in zip(query['processed_question_toks'],
                                                        query['schema_linking'][0],
                                                        query['schema_linking'][1])])
        '''
        self.querys.append(query)

    def filter_table_and_column(self, query, translator):
        def get_table_and_column(node):
            ret_table, ret_column = [], []
            for field in node.fields:
                for child in field.as_value_list:
                    if isinstance(child, AbstractSyntaxTree):
                        child_table, child_column = get_table_and_column(child)
                        ret_table += child_table
                        ret_column += child_column
                    else:
                        assert isinstance(child, int)
                        assert field.name in ['tab_id', 'col_id']
                        if field.name == 'tab_id':
                            ret_table.append(child)
                        else:
                            if child != 0:
                                ret_column.append(child)
            return ret_table, ret_column

        sql = query['sql']
        ast = translator.surface_code_to_ast(sql)
        table_ids, column_ids = get_table_and_column(ast)
        table_ids, column_ids = set(table_ids), set(column_ids)
        assert len(table_ids) <= len(self.tables)
        assert len(column_ids) <= sum([t.size for t in self.tables])
        return list(table_ids), list(column_ids)

    def database2er(self):
        flag = {table: False for table in self.tables}
        table2entity = {}
        all_relationships = []

        for relations in self.table_relations.values():
            for to_table, _, _ in relations:
                flag[to_table] = True
        for table in self.tables:
            if flag[table]:
                table2entity[table] = convert_table_to_entity(table)

        for table in self.tables:
            table_relations = self.table_relations[table]
            tmp = []
            for _, f_c, _ in table_relations:
                tmp += f_c
            tmp = set(tmp)
            convert = False
            if not flag[table] and len(table_relations) == 2 and table.size - len(tmp) <= 1:
                split_name = table.name.split('_')
                if len(split_name) != 2:
                    #print('please manually check:', database.db_id, table.name, '\n', [c.name for c in table.columns], '\n',
                    #      [t.name for t, _, _ in table_relations])
                    #convert = input() == '1'
                    convert = True
                else:
                    name1, name2 = split_name
                    if set([name1, name2]) != set([t.name for t, _, _ in table_relations]):
                        #print('please manually check:', database.db_id, table.name, '\n', [c.name for c in table.columns], '\n',
                        #      [t.name for t, _, _ in table_relations])
                        #convert = input() == '1'
                        convert = True
                    else: convert = True
                if convert:
                    print('Convert table to Relationship:', table.name)
                    t1, t2 = [t for t, _, _ in table_relations]
                    tmp = [self.get_column(id) for id in tmp]
                    new_relation = RelationShip(table.name,
                                                table2entity[t1],
                                                table2entity[t2],
                                                card=[-1, -1],
                                                attr=[c.name for c in
                                                      filter(lambda x: x not in tmp,
                                                             table.columns)])
                    all_relationships.append(new_relation)

            if not convert:
                if not flag[table]:
                    table2entity[table] = convert_table_to_entity(table)
                from_entity = table2entity[table]
                for to_table, from_k, to_k in table_relations:
                    for k in from_k:
                        c = self.get_column(k)
                        if c.name in from_entity.attribute:
                            from_entity.delete_attribute(c.name)

                    to_entity = table2entity[to_table]
                    new_relation = RelationShip(from_entity.name + '_' + to_entity.name,
                                                from_entity, to_entity,
                                                card=[-1, 1])
                    all_relationships.append(new_relation)

        #all_entities = list(table2entity.values())
        #print('Finish transforming:', self.db_id)
        self.erm = ERModel(table2entity, all_relationships)


    def compute_coverage(self, translator):
        all_entity = self.erm.entities
        all_attr = self.erm.valid_columns
        all_key_relation = self.erm.relations

        covered_entity = set()
        covered_attr = set()
        covered_have_relation = set()
        covered_key_relation = set()

        linked_entity = set()
        linked_attr = set()
        linked_have_relation = set()
        linked_key_relation = set()
        def get_relation(tables, columns):
            have_relation, key_relation = [], []
            for t1, _ in tables:
                for t2, _ in tables:
                    if t1 == t2: continue
                    if self.erm.get_table_relation(t1, t2):
                        key_relation.append((t1, t2))

            for t, _ in tables:
                for c, _ in columns:
                    if self.erm.get_table_column_relation(t, c):
                        have_relation.append((t, c))
            return have_relation, key_relation

        for query in self.querys:
            filtered_tables_ids, filtered_columns_ids = \
                self.filter_table_and_column(query, translator)
            filtered_tables = [(self.get_table(index), index) for index in filtered_tables_ids]
            filtered_columns = [(self.get_column(index), index) for index in filtered_columns_ids]
            filtered_tables = list(filter(lambda x: self.erm.is_entity(x[0]), filtered_tables))
            filtered_columns = list(filter(lambda x: self.erm.is_attr(x[0]), filtered_columns))
            covered_entity.update(filtered_tables)
            covered_attr.update(filtered_columns)
            '''
            have_relation, key_relation = get_relation(filtered_tables, filtered_columns)
            covered_have_relation.update(have_relation)
            covered_key_relation.update(key_relation)
            '''

            table_tag, column_tag = query['schema_linking']
            tmp_table = set()
            for index in table_tag:
                if index > 0:
                    tmp_table.add((self.get_table(int(index - 1)), int(index - 1)))

            tmp_column = set()
            for index in column_tag:
                if index > 0:
                    tmp_column.add((self.get_column(int(index - 1)), int(index - 1)))
            linked_entity.update(tmp_table)
            linked_attr.update(tmp_column)
            tmp_have_relation, tmp_key_relation = get_relation(tmp_table, tmp_column)
            linked_have_relation.update(tmp_have_relation)
            linked_key_relation.update(tmp_key_relation)
        covered_have_relation, covered_key_relation = get_relation(covered_entity, covered_attr)
        cross_sent_have_relation, cross_sent_key_relation = get_relation(linked_entity, linked_attr)
        #print(len(all_entity), len(all_attr), len(all_attr), len(all_key_relation))
        #print(len(covered_entity), len(covered_attr), len(covered_have_relation), len(covered_key_relation))
        #print(len(linked_entity), len(linked_attr), len(cross_sent_have_relation), len(cross_sent_key_relation))
        #print(len(linked_have_relation), len(linked_key_relation))
        return all_entity, all_attr, all_key_relation, covered_entity, covered_attr, covered_have_relation, \
               covered_key_relation, linked_entity, linked_attr, cross_sent_have_relation, cross_sent_key_relation, \
               linked_have_relation, linked_key_relation
