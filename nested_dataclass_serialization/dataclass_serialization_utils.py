import dataclasses
import importlib
from typing import Annotated, Any, ClassVar, Protocol, TypeVar, runtime_checkable

from beartype.vale import Is

# ------------TODO: move the following upstream to some utils repo?-----------------
NeStr = Annotated[str, Is[lambda s: len(s) > 0]]
Dataclass = Annotated[object, Is[dataclasses.is_dataclass]]
JsonLoadsOutput = (
    dict[str, Any] | list[Any] | str | int | float | bool | None
)  # forgot anything? set  cannot be handled by json
PythonBuiltinData = JsonLoadsOutput | tuple[Any, ...] | set[Any]


# -------------------------------------------------------------------------------------


@runtime_checkable
class DataclassP(Protocol):
    # see: https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[dict[str, Any]]


IDKEY = "_id_"
ID_KEY = IDKEY  # TODO: rename
CLASS_REF_KEY = "_target_"
SPECIAL_KEYS = [IDKEY, CLASS_REF_KEY, "_cls_", "_was_built"]
CLASS_REF_NO_INSTANTIATE = "_not_to_instantiate_python_dataclass_"  # use this to prevent instantiate_via_importlib, if one wants class-reference for documentation purposes only
UNSERIALIZABLE = "<UNSERIALIZABLE>"
NODE_ID_KEY = "__node_id__"
NODES_KEY = "__nodes__"


def is_dunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


def instantiate_via_importlib(
    d: dict[str, Any],
    fullpath: str,
) -> Any:
    *module_path, class_name = fullpath.split(".")
    module_reference = ".".join(module_path)
    clazz = getattr(importlib.import_module(module_reference), class_name)
    if hasattr(clazz, "create"):
        out = clazz.create(**d)
    elif hasattr(
        clazz,
        dataclasses._FIELDS,  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
    ):
        out = shallow_dataclass_from_dict(clazz, d)
    else:
        out = clazz(**d)
    return out


T = TypeVar("T")


def shallow_dataclass_from_dict(clazz: type[T], dct: dict) -> T:
    """
    NO decoding of nested dicts to nested dataclasses here!!
    is used as a "factory" in instantiate_via_importlib
    dict can contain dataclasses or whatever objects!
    """
    assert dataclasses.is_dataclass(clazz)  # just for type narrowing
    kwargs = {
        f.name: dct[f.name]
        for f in dataclasses.fields(clazz)
        if (f.init and f.name in dct.keys())
    }
    obj = clazz(**kwargs)
    set_noninit_fields(clazz, dct, obj)
    return obj


def set_noninit_fields(cls: Dataclass, dct: dict, obj: Any) -> None:
    assert dataclasses.is_dataclass(
        cls,
    )  # just for type narrowing -> even though beartype checks this, pyright still wants an assert
    state_fields = (
        f for f in dataclasses.fields(cls) if (not f.init and f.name in dct)
    )
    for f in state_fields:
        setattr(obj, f.name, dct[f.name])


class ImpossibleType:
    def __new__(cls):
        raise NotImplementedError


try:  # noqa: WPS229
    import omegaconf

    OmegaConfList = omegaconf.listconfig.ListConfig
    OmegaConfDict = omegaconf.dictconfig.DictConfig
except:  # noqa: E722
    OmegaConfList = ImpossibleType
    OmegaConfDict = ImpossibleType
