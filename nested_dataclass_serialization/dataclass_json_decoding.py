import logging
from dataclasses import dataclass, field
from typing import Any

from nested_dataclass_serialization.dataclass_serialization_utils import (
    CLASS_REF_KEY,
    IDKEY,
    NODE_ID_KEY,
    instantiate_via_importlib,
)

logger = logging.getLogger(
    __name__,
)


@dataclass(slots=True)
class DataclassDecoderObjectHook:
    object_registry: dict[str, Any] = field(default_factory=dict)
    class_ref_key: str = CLASS_REF_KEY
    nodeid_2_objid: dict[str, str] | None = None

    def __call__(self, dct: dict) -> Any:
        # logger.debug(f"decoding: {dct}")
        if self.class_ref_key in dct.keys() and IDKEY in dct.keys():
            eid = dct.pop(IDKEY)

            if eid in self.object_registry.keys():
                o = self.object_registry[eid]
            else:
                fullpath = dct.pop(self.class_ref_key)
                # TODO: here some try except with lookup in TARGET_CLASS_MAPPING
                try:
                    o = instantiate_via_importlib(dct, fullpath)
                finally:
                    logger.error(f"could not instantiate: {fullpath}")
                self.object_registry[eid] = o
        elif NODE_ID_KEY in dct.keys():
            o = self.object_registry[self.nodeid_2_objid[dct[NODE_ID_KEY]]]
        else:
            o = dct
        return o
