import json

from nested_dataclass_serialization.dataclass_json_decoding import (
    DataclassDecoderObjectHook,
)
from nested_dataclass_serialization.dataclass_json_encoding import DataclassEncoder
from nested_dataclass_serialization.dataclass_serialization_utils import (
    CLASS_REF_KEY,
    ID_KEY,
    NODE_ID_KEY,
    NODES_KEY,
    Dataclass,
    JsonLoadsOutput,
    NeStr,
)


def decode_dataclass(
    o: dict | (list | tuple),
    class_ref_key: str = CLASS_REF_KEY,
    is_sparse: bool = False,
) -> Dataclass | JsonLoadsOutput:
    # TODO:  why did I want this Base64-stuff?
    # o = json.dumps(o) # there must be a better way than, json.dumps twice!
    # o = json.loads(o, cls=Base64Decoder)
    return _json_loads_decode_dataclass(
        json.dumps(o),
        class_ref_key,
        is_sparse=is_sparse,
    )


def deserialize_dataclass(
    o: NeStr,
    class_ref_key: str = CLASS_REF_KEY,
    is_sparse: bool = False,
) -> Dataclass | JsonLoadsOutput:
    # TODO: why did I want this Base64-stuff?
    # o = json.loads(o, cls=Base64Decoder)
    return _json_loads_decode_dataclass(o, class_ref_key, is_sparse=is_sparse)


def _json_loads_decode_dataclass(
    s: str,
    class_ref_key: str = CLASS_REF_KEY,
    is_sparse: bool = False,
) -> Dataclass | JsonLoadsOutput:
    if is_sparse:
        dct = json.loads(s)
        assert NODES_KEY in dct.keys()
        assert NODE_ID_KEY in dct.keys()
        nodeid_2_objid = {
            node_id: node[ID_KEY] for node_id, node in dct[NODES_KEY].items()
        }
    else:
        nodeid_2_objid = None
    object_hook = DataclassDecoderObjectHook(
        class_ref_key=class_ref_key,
        nodeid_2_objid=nodeid_2_objid,
    )
    return json.loads(s, object_hook=object_hook)


def serialize_dataclass(  # noqa: PLR0913
    d: str | Dataclass,  # TODO: WTF why str?
    class_reference_key: str = CLASS_REF_KEY,
    skip_undefined: bool = True,
    skip_keys: list[str] | None = None,
    encode_for_hash: bool = False,
    sparse: bool = False,
    indent: int | None = None,  # use indent =4 for "pretty json"
) -> str:
    return json.dumps(
        encode_dataclass(
            d,
            class_reference_key,
            skip_undefined,
            skip_keys,
            sparse=sparse,
            encode_for_hash=encode_for_hash,
        ),
        ensure_ascii=False,
        indent=indent,  # use indent =4 for "pretty json"
    )


def encode_dataclass(  # noqa: PLR0913
    d: Dataclass | JsonLoadsOutput | tuple,
    class_reference_key: str = CLASS_REF_KEY,
    skip_undefined: bool = True,
    skip_keys: list[str] | None = None,
    sparse: bool = False,
    encode_for_hash: bool = False,
) -> dict | (list | (tuple | set)):
    """
    # TODO: bad naming! cause it not only handles Dataclasses but also JsonLoadsOutput
    encode in the sense that the dictionary representation can be decoded to the nested dataclasses object again
    """
    DataclassEncoder.class_reference_key = class_reference_key
    DataclassEncoder.skip_undefined = skip_undefined
    DataclassEncoder.skip_keys = skip_keys
    DataclassEncoder.sparse = sparse
    DataclassEncoder.encode_for_hash = encode_for_hash
    return DataclassEncoder().default(d)
