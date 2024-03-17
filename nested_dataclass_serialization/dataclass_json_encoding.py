import copy
import dataclasses
import inspect
import json
import logging
import re
import uuid
from collections.abc import Callable
from hashlib import sha1
from typing import Any

from nested_dataclass_serialization.dataclass_serialization_utils import (
    CLASS_REF_KEY,
    IDKEY,
    NODE_ID_KEY,
    NODES_KEY,
    Dataclass,
    OmegaConfDict,
    OmegaConfList,
    PythonBuiltinData,
)
from nested_dataclass_serialization.strange_hack import (
    fix_module_if_class_in_same_file_as_main,
)

logger = logging.getLogger(
    __name__,
)  # "The name is potentially a period-separated hierarchical", see: https://docs.python.org/3.10/library/logging.html

salt = (
    uuid.uuid1()
)  # to prevent clashes when "merging" graphs built in different python processes, cause then objects-ids could clash!

KEY, VALUE = str, Any
KeyValues = list[tuple[KEY, VALUE]]
DictFactory = Callable[[KeyValues], dict]


class CannotEncodeError(Exception):
    pass


class DataclassEncoder(json.JSONEncoder):
    """
    # see: https://stackoverflow.com/questions/64777931/what-is-the-recommended-way-to-include-properties-in-dataclasses-in-asdict-or-se
    """

    class_reference_key = CLASS_REF_KEY
    skip_undefined = True
    encode_for_hash = False  # excludes fields who's names are listed in "__exclude_from_hash__" serialization and thereby from hash, but if encode_for_hash=False they still get serialized!
    sparse: bool = True  # TODO: rename to "flat"!?
    is_special = re.compile(
        r"^__[^\d\W]\w*__\Z",
        re.UNICODE,
    )  # Dunder name. -> field from stackoverflow
    skip_keys: list[str] | None = None

    def _flat_dag_dict(self, key_value: list[tuple[str, Any]]) -> dict:
        dct = dict(key_value)
        if IDKEY in dct.keys():
            obj_id = dct[IDKEY]
            if obj_id in self._object2node_id:
                node_id = self._object2node_id[obj_id]
            else:
                node_id = f"node-{len(self._id2node_.keys())}"
                self._object2node_id[obj_id] = node_id
                self._id2node_[node_id] = dct
            node_dct = {NODE_ID_KEY: node_id}
        else:
            node_dct = dct
        return node_dct

    def default(self, o: Any) -> PythonBuiltinData:
        """
        this overwrites json.JSONEncoders default method
        """
        if self.sparse:
            # msg = "if you want it, fix it!"
            # raise NotImplementedError(msg)
            self._object2node_id = {}
            self._id2node_ = {}
        dct = self._obj2dict(
            o,
            dict_factory=self._flat_dag_dict if self.sparse else dict,
        )
        if self.sparse:
            dct[NODES_KEY] = self._id2node_
        return dct

    def _obj2dict(  # noqa: C901, PLR0912, WPS231
        self,
        obj: Any,
        dict_factory: DictFactory,
    ) -> PythonBuiltinData:
        if dataclasses.is_dataclass(obj):
            out = self._encode_dataclass(obj, dict_factory)
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # TODO: this could return any class that implements _fields method! WTF! not what I want!
            raise CannotEncodeError
            # out = type(obj)(*(self._obj2dict(v, dict_factory) for v in obj))
        elif isinstance(obj, list | tuple | OmegaConfList):
            if isinstance(obj, OmegaConfList):
                obj = list(obj)
            out = type(obj)(self._obj2dict(v, dict_factory) for v in obj)
        elif isinstance(obj, dict | OmegaConfDict):
            if isinstance(obj, OmegaConfDict):
                obj = dict(obj)

            out = type(obj)(
                (
                    self._obj2dict(k, dict_factory),
                    self._obj2dict(v, dict_factory),
                )
                for k, v in obj.items()
                if self.skip_keys is None or k not in self.skip_keys
            )
        elif callable(obj):
            logger.warning("you are encoding a callable! this cannot be decoded!")
            out = inspect.getsource(
                obj,
            )  # TODO: this is hacky! and not working for deserialization!
        else:
            # try:
            obj = copy.deepcopy(obj)
            # except: # TODO: when is it not possible to deepcopy?
            #     obj= f"{UNSERIALIZABLE}{id(obj)=}{UNSERIALIZABLE}"
            obj = (
                obj._to_dict(self.skip_keys)  # noqa: SLF001
                if hasattr(obj, "_to_dict")
                else obj
            )
            out = obj
        return out

    def _encode_dataclass(self, obj: Dataclass, dict_factory: DictFactory) -> dict:
        result: list[tuple[str, Any]] = []
        module = obj.__class__.__module__
        if module == "__main__":
            module = fix_module_if_class_in_same_file_as_main(obj)
        clazz_name = obj.__class__.__name__
        _target_ = f"{module}.{clazz_name}"
        self.maybe_append(result, self.class_reference_key, _target_)
        clazz_name_hash = sha1(clazz_name.encode("utf-8")).hexdigest()  # noqa: S324
        hash_ = f"{salt}-{id(obj)}-{clazz_name_hash}"
        self.maybe_append(result, IDKEY, f"{hash_}")
        result.extend(self._fields_to_serialize(dict_factory, obj))
        result.extend(self._values_of_non_special_properties(obj))
        result.extend(self._serialize_anyhow(obj))
        return dict_factory(result)

    def maybe_append(self, r: list, k: str, v: Any) -> None:
        skip_this_one = self.skip_keys and k in self.skip_keys
        if not skip_this_one:
            r.append((k, v))

    def _fields_to_serialize(self, dict_factory: DictFactory, obj: Any) -> KeyValues:
        def exclude_for_hash(o: Dataclass, f_name: str) -> bool:
            if self.encode_for_hash and hasattr(o, "__exclude_from_hash__"):
                out = f_name in o.__exclude_from_hash__
            else:
                out = False
            return out

        feelds = (
            f
            for f in dataclasses.fields(obj)
            if f.repr
            and hasattr(obj, f.name)
            and not f.name.startswith("_")
            and not exclude_for_hash(obj, f.name)
            and not (self.skip_keys and f.name in self.skip_keys)
        )
        name_values = ((f.name, getattr(obj, f.name)) for f in feelds)
        return [
            (name, self._obj2dict(value, dict_factory))
            for name, value in name_values
            if value.__class__.__name__ != "_UNDEFINED"
            or not self.skip_undefined  # TODO(tilo): hardcoded this UNDEFINED here! think about it! and fix it!
        ]

    def _values_of_non_special_properties(self, obj: Any) -> KeyValues:
        # Add values of non-special attributes which are properties.
        # idea comes from https://stackoverflow.com/questions/64777931/what-is-the-recommended-way-to-include-properties-in-dataclasses-in-asdict-or-se
        # but why did I want this?

        is_special = self.is_special.match  # Local var to speed access.
        properties_to_be_serialized = (
            obj.__serializable_properties__
            if hasattr(obj, "__serializable_properties__")
            else []
        )
        return [
            (name, attr.__get__(obj))
            for name, attr in vars(type(obj)).items()
            if (
                not is_special(name)
                and isinstance(attr, property)
                and name in properties_to_be_serialized
            )
        ]

    def _serialize_anyhow(self, obj: Any) -> KeyValues:
        if hasattr(obj, "__serialize_anyhow__"):
            serialize_those = [
                (name, getattr(obj, name))
                for name in obj.__serialize_anyhow__
                if hasattr(obj, name)
            ]
        else:
            serialize_those = []
        return serialize_those
