from abc import ABC, abstractmethod

class BaseANNIndex(ABC):
    def __init__(self, dim: int, metric: str):
        self.dim = dim
        self.metric = metric

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.build_strategies = {}
        cls.query_strategies = {}

    @classmethod
    def register_build(cls, attribute_type):
        def decorator(func):
            cls.build_strategies[attribute_type] = func
            return func
        return decorator
    
    @classmethod
    def register_query(cls, attribute_type, query_type):
        def decorator(func):
            cls.query_strategies[(attribute_type, query_type)] = func
            return func
        return decorator

    def build(self, vectors, attributes, parameters, config):
        self.attribute_type = config.attribute_type
        if self.attribute_type not in self.build_strategies:
            raise ValueError(f"Unsupported attribute type: {self.attribute_type}")
        
        fn = self.build_strategies[self.attribute_type]
        return fn(self, vectors, attributes, parameters)

    def query(self, vectors, filters, k, parameters, config):
        if config.attribute_type != self.attribute_type:
            raise ValueError("Mismatch between build and query attribute type")

        key = (config.attribute_type, config.query_type)
        if key not in self.query_strategies:
            raise ValueError(f"Unsupported query type: {key}")

        fn = self.query_strategies[key]
        return fn(self, vectors, filters, k, parameters)

    @abstractmethod
    def name(self) -> str:
        pass