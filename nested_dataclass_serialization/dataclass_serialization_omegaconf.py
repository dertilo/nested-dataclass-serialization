from omegaconf import OmegaConf

from nested_dataclass_serialization.dataclass_serialization import (
    deserialize_dataclass,
    serialize_dataclass,
)
from nested_dataclass_serialization.dataclass_serialization_utils import Dataclass


def deserialize_from_yaml(
    yaml_file: str,
) -> Dataclass:
    """
    TODO: do I really have to serialize the DictConfig before deserializing?
    """
    cfg = OmegaConf.load(yaml_file)
    return deserialize_dataclass(serialize_dataclass(cfg))


def dataclass_to_yaml(
    o: Dataclass,
    skip_undefined: bool = False,
    skip_keys: list[str] | None = None,
) -> str:
    sd = serialize_dataclass(o, skip_undefined=skip_undefined, skip_keys=skip_keys)
    cfg = OmegaConf.create(sd)
    return OmegaConf.to_yaml(cfg)
    # print(yaml)
    # deser_obj = deserialize_dataclass(sd)
    # deserialize_dataclass can be different due to FILLED_AT_RUNTIME values that are filtered out
    # assert str(deser_obj) == str(o),deser_obj
