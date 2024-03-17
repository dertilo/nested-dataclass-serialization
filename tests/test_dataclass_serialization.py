# flake8: noqa
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pprint import pprint
from typing import ClassVar

import pytest
from beartype.roar import BeartypeCallException

from nested_dataclass_serialization.dataclass_serialization import (
    encode_dataclass,
    deserialize_dataclass,
    serialize_dataclass,
    decode_dataclass,
)
from nested_dataclass_serialization.dataclass_serialization_utils import IDKEY


class TestCasing(str, Enum):
    # TODO: copy pasted this, do I really needed for testing? or move tests to text_processing!
    lower = auto()
    upper = auto()
    original = auto()

    def _to_dict(self, skip_keys: list[str] | None = None) -> dict:
        obj = self
        module = obj.__class__.__module__
        _target_ = f"{module}.{obj.__class__.__name__}"
        # TODO: WTF? why _target_ and _id_ stuff here?
        d = {"_target_": _target_, "value": self.value, "_id_": str(id(self))}
        skip_keys = skip_keys if skip_keys is not None else []
        return {k: v for k, v in d.items() if k not in skip_keys}

    def apply(self, s: str) -> str:
        if self is TestCasing.upper:
            o = s.upper()
        elif self is TestCasing.lower:
            o = s.lower()
        elif self is TestCasing.original:
            o = s
        else:
            raise AssertionError
        return o

    @staticmethod
    def create(value: str | int) -> "TestCasing":
        return TestCasing(str(value))


@dataclass
class AnotherDataClass:
    foo: str
    bar: int = 1
    _another_private_field: str = field(
        default="not serialized",
        init=False,
        repr=False,
    )


class Singleton(type):
    """
    see: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    _instances = {}  # noqa: RUF012

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass(frozen=True, slots=True)
class TypeWithSemanticMeaning(metaclass=Singleton):
    pass


TYPE_WITH_SEMANTIC_MEANING = TypeWithSemanticMeaning()

i_got_serialized = "i got serialized"
i_dont_get_serialized = "i dont get serialized"


@dataclass
class TestDataClass:
    # TODO: inner-class serialization not yet working!!
    # @dataclass
    # class InnerTypeWithSemanticMeaning(metaclass=Singleton):
    #     pass
    #
    # INNTER_TYPE_WITH_SEMANTIC_MEANING = InnerTypeWithSemanticMeaning()

    foo: str
    bla: dict[str, list[AnotherDataClass]]
    three_times_the_same: list[AnotherDataClass]
    bar: int | TypeWithSemanticMeaning = TYPE_WITH_SEMANTIC_MEANING
    class_var: ClassVar[str] = "class-var is not serialized"
    # inner_bar: Union[int, InnerTypeWithSemanticMeaning] = INNTER_TYPE_WITH_SEMANTIC_MEANING
    _private_field: str = field(
        default="not serialized and thereby NOT deserialized",
        init=False,
        repr=False,
    )
    state_field: str = field(
        default="is serialized and deserialized",
        init=False,
        repr=True,
    )
    # TODO: how to handle fields that cannot be pickled? serialized? -> currently it raises at copy.deepcopy
    # iterator_field:Optional[Iterator]=field(default=None,init=True,repr=True)
    __serializable_properties__: ClassVar[list[str]] = ["property_to_be_serialized"]

    @property
    def please_serialize_me(self) -> str:
        return i_dont_get_serialized

    @property
    def property_to_be_serialized(self) -> str:
        return i_got_serialized


@pytest.fixture()
def test_object() -> TestDataClass:
    obj = AnotherDataClass("samesame")
    return TestDataClass(
        foo="foo",
        bla={"foo": [AnotherDataClass(f"foo-{k}") for k in range(3)], "bar": None},
        three_times_the_same=[obj for _ in range(3)],
    )


def test_serialize_deserialize_dataclass(
    test_object: TestDataClass,
) -> None:
    test_object.state_field = "state changed"
    serialized_object = serialize_dataclass(test_object)
    assert "TYPE_WITH_SEMANTIC_MEANING" not in serialized_object
    assert "TypeWithSemanticMeaning" in serialized_object
    assert "class-var is not serialized" not in serialized_object
    assert i_dont_get_serialized not in serialized_object
    assert i_got_serialized in serialized_object

    deser_obj = deserialize_dataclass(serialized_object)
    assert str(deser_obj) == str(test_object), deser_obj


def test_sparse_serialize_deserialize_dataclass(
    test_object: TestDataClass,
) -> None:
    test_object.state_field = "state changed"
    serialized_object = serialize_dataclass(test_object, sparse=True)
    assert "TYPE_WITH_SEMANTIC_MEANING" not in serialized_object
    assert "TypeWithSemanticMeaning" in serialized_object
    assert "class-var is not serialized" not in serialized_object
    assert i_dont_get_serialized not in serialized_object
    assert i_got_serialized in serialized_object
    # pprint(test_object)
    # pprint(json.loads(serialized_object))

    deser_obj = deserialize_dataclass(serialized_object, is_sparse=True)
    assert str(deser_obj) == str(test_object), deser_obj


def test_private_field(test_object: TestDataClass) -> None:
    test_object._private_field = "changed value"  # noqa: SLF001
    test_object.bla["foo"][0]._another_private_field = "changed value"  # noqa: SLF001
    sd = serialize_dataclass(test_object)
    deser_obj = deserialize_dataclass(sd)
    print(f"{deser_obj=}")
    assert (
        deser_obj._private_field != test_object._private_field  # noqa: SLF001
    ), deser_obj

    deser_obj._private_field = "changed value"  # noqa: SLF001
    deser_obj.bla["foo"][0]._another_private_field = "changed value"  # noqa: SLF001
    assert str(deser_obj) == str(test_object), deser_obj


#
#
@dataclass
class Bar:
    x: str
    casing: TestCasing | None = None
    another_cache_dir: str = "bla"


@dataclass
class Foo:
    bars: list[Bar]




def test_object_registry() -> None:
    bar = Bar(x="test")
    foo = Foo(bars=[bar, bar])
    s = serialize_dataclass(foo)
    des_foo = deserialize_dataclass(s)

    assert id(des_foo.bars[0]) == id(des_foo.bars[1])


def test_skip_keys() -> None:
    bar = Bar(x="test", casing=TestCasing.lower)
    foo = Foo(bars=[bar, bar])
    skip_keys = [IDKEY, "cache_base", "cache_dir"]
    s = serialize_dataclass(foo, skip_keys=skip_keys)
    assert not any(f'"{k}"' in s for k in skip_keys), f"{s}"


def test_deserialization_not_bothered_by_unknown_keys() -> None:
    bar = Bar(x="test", casing=TestCasing.lower)
    d = encode_dataclass(bar)
    d["unknown_key"] = "extra-data"
    s = json.dumps(d)
    o = deserialize_dataclass(s)
    assert o == bar, f"{o}"


def test_encode_nonencodable() -> None:
    inp = [1, 2]
    assert encode_dataclass(inp) == inp
    inp = ("foo",)
    assert encode_dataclass(inp) == inp
    inp = {"bla": 1.234}
    assert encode_dataclass(inp) == inp


def test_encode_container_of_dataclasses() -> None:
    bar = Bar("foo", TestCasing.lower, "bar")
    foo = Foo([bar])
    d = encode_dataclass([foo, foo])
    dec = decode_dataclass(d)
    assert isinstance(dec, list)
    assert all(isinstance(x, Foo) for x in dec)
    assert all(isinstance(x.bars[0], Bar) for x in dec)
    assert all(isinstance(x.bars[0].casing, TestCasing) for x in dec)


@pytest.mark.parametrize(
    "casing",
    [
        TestCasing.lower,
        TestCasing.upper,
        TestCasing.original,
    ],
)
def test_endecode__dataclass(casing: TestCasing) -> None:
    bar = Bar("foo", casing, "bar")
    foo = Foo([bar])
    d = encode_dataclass(foo)

    dec = decode_dataclass(d)
    assert isinstance(dec, Foo), f"{dec=}"
    assert isinstance(dec.bars[0], Bar), f"{dec=}"
    assert isinstance(dec.bars[0].casing, TestCasing), f"{dec=}"

    dec = decode_dataclass([d, d])
    assert isinstance(dec, list)
    assert all(isinstance(x, Foo) for x in dec)
    assert all(isinstance(x.bars[0], Bar) for x in dec)
    assert all(isinstance(x.bars[0].casing, TestCasing) for x in dec)


def test_deserialze_casing_with_int() -> None:
    bar = Bar("foo", TestCasing.upper, "bar")
    d = encode_dataclass(bar)
    d["casing"]["value"] = int(d["casing"]["value"])
    dec: Bar = decode_dataclass(d)
    assert isinstance(dec.casing, TestCasing)


def test_endeserialize_dataclass() -> None:
    bar = Bar("foo", TestCasing.lower, "bar")
    foo = Foo([bar])
    d = serialize_dataclass(foo)
    dec = deserialize_dataclass(d)
    assert isinstance(dec, Foo), f"{dec=}"
    assert isinstance(dec.bars[0], Bar), f"{dec=}"


# TODO
# def test_sparse_dataclass():
#     bar = Bar("foo", TestCasing.lower, "bar")
#     foo = Foo([bar, bar])
#     # d = encode_dataclass(foo,sparse=True)
#     write_file("test.json", serialize_dataclass(foo, sparse=True))
#     # pprint(f"{d=}")
#     # dec = decode_dataclass(d)
#     # assert isinstance(dec, list)
#     # assert all([isinstance(x, Foo) for x in dec])
#     # assert all([isinstance(x.bars[0], Bar) for x in dec])
#     # assert all([isinstance(x.bars[0].casing, TestCasing) for x in dec])


if __name__ == "__main__":
    """
    for some stupid reason this doesn't work form pycharm but form command-line
    even though I set working-directory in run-configs
    """
    o = TestDataClass(
        foo="foo",
        bla={"foo": [AnotherDataClass("foo") for k in range(3)], "bar": None},
    )
    assert o.__class__.__module__ == "__main__"
    s = encode_dataclass(o)
    o2 = decode_dataclass(s)
    print(o2)
