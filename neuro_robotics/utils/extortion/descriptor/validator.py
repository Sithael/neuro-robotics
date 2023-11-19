class NonBlank:
    def __set__(self, instance, entity):
        if len(entity) == 0:
            raise ValueError(f"{self.storage_name} must not be empty")
        instance.__dict__[self.storage_name] = entity


class DictValuesAreNotEmpty:
    def __set__(self, instance, entity: dict):
        for ev in entity.values():
            if ev == "" or ev is None:
                raise ValueError(f"{self.storage_name} dict values has to be defined")
        instance.__dict__[self.storage_name] = entity


class ModelMeta(type):

    defined_descriptors = (
        NonBlank,
        DictValuesAreNotEmpty,
    )

    def __init__(cls, name, bases, dic):
        super().__init__(name, bases, dic)

        for name, attr in dic.items():
            if isinstance(attr, cls.defined_descriptors):
                attr.storage_name = name


class Model(metaclass=ModelMeta):
    """inherit from this class to obtain descriptor"""

    pass
