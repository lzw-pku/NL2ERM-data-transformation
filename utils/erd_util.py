from collections import defaultdict


class EntityType:
    def __init__(self, name):
        self.name = name
        self.attribute = []

    def add_attribute(self, attr):
        assert attr not in self.attribute
        self.attribute.append(attr)

    def delete_attribute(self, attr):
        assert attr in self.attribute
        self.attribute.remove(attr)

    @property
    def size(self):
        return len(self.attribute)

    def __repr__(self):
        s = self.name + '\n'
        for attr in self.attribute:
            s += '----' + attr + '\n'
        return s

class RelationShip:
    def __init__(self, name, entity1, entity2, card=None, attr=None):
        self.name = name
        self.entity1 = entity1
        self.entity2 = entity2
        self.card = card if card else -1
        self.attribute = attr if attr else []

    def __repr__(self):
        return '{}({}, {}){}'.format(self.name, self.entity1.name,
                                     self.entity2.name, self.attribute)

class ERModel:
    def __init__(self, table2entity, relationships):
        self.table2entity = table2entity
        self.entities = list(table2entity.values())
        self.relations = relationships
        self.entity2relation = {entity1:{entity2:False for entity2 in self.entities}
                                for entity1 in self.entities}
        for relation in self.relations:
            self.entity2relation[relation.entity1][relation.entity2] = True
            self.entity2relation[relation.entity2][relation.entity1] = True

        self.valid_columns = set()
        for table, entity in self.table2entity.items():
            for column in table.columns:
                if column.name in entity.attribute:
                    self.valid_columns.add(column)

    def table_to_entity(self, table):
        return self.table2entity[table]

    def is_entity(self, table):
        return table in self.table2entity

    def is_attr(self, column):
        return column in self.valid_columns

    def get_table_relation(self, table1, table2):
        entity1 = self.table2entity[table1]
        entity2 = self.table2entity[table2]
        return self.entity2relation[entity1][entity2]

    def get_table_column_relation(self, table, column):
        return column in table.columns and self.is_attr(column)

    def __repr__(self):
        s = ''
        for entity in self.entities:
            s += str(entity)
        for relation in self.relationships:
            s += str(relation) + '\n'
        return s
