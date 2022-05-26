from typing import Iterable
from .. import db


class BaseModel(db.Model):
    # __abstract__ means that SQLAlchemy will not create a table for this model
    __abstract__ = True

    def to_json(self, populate: Iterable = (), exclude_columns: Iterable = (), include_foreign_keys=False):
        """
        Convert Models to json

        Args:
            populate (Iterable): List of field names to populate with their objects
                                 This works recursively, as long as there is a field in `populate` that can be expanded
            exclude_columns (Iterable): List of column names to exclude from the json output
                                        In the format of `[<table_name>.<column_name>]` (e.g. `Person.id`)
            include_foreign_keys (bool): Whether to include foreign key columns in the json output

        Returns:
            json (dict): json representation of the model
        """

        json = {}

        for c in self.__table__.columns:
            if str(c) not in exclude_columns and (not c.foreign_keys or (c.foreign_keys and include_foreign_keys)):
                json[c.name] = getattr(self, c.name)

        for field in populate:
            if not hasattr(self, field) or field in json:
                continue

            value = getattr(self, field)
            if hasattr(value, '__iter__'):
                json[field] = [v.to_json(populate, exclude_columns, include_foreign_keys) for v in value]
            else:
                json[field] = value.to_json(populate, exclude_columns, include_foreign_keys)
        return json
