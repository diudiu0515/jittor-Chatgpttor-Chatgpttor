from .spec import DummySystem, DummyWriter
from .vm import VMSystem, VMWriter

def get_system(**kwargs) -> DummySystem:
    MAP = {
        'dummy': DummySystem,
        'vm': VMSystem,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)

def get_writer(**kwargs) -> DummyWriter:
    MAP = {
        'dummy': DummyWriter,
        'vm': VMWriter,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)