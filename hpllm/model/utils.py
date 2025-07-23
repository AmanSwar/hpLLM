import jax
from jax import tree_util

import dataclasses



def pytree_struct(cls, meta_fields: tuple = ()):
    """
    register_dataclass wrapper that automatically infers the data_fields

    Args:
        meta_fields (tuple, optional): meta_fields about the class , field which are supposed to be meta information. Defaults to ().

    """    
    assert not dataclasses.is_dataclass(cls)

    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)

    return tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )


