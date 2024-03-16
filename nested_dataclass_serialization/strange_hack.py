import inspect
import os
from typing import Any


def fix_module_if_class_in_same_file_as_main(obj: Any) -> str:
    assert (
        "PYTHONPATH" in os.environ
    ), "do export PYTHONPATH=${PWD} if you run script from __main__"
    prefixes = os.environ["PYTHONPATH"].split(":")
    prefixes = [  # which are not prefix of another one
        p for p in prefixes if not any(pp.startswith(p) and p != pp for pp in prefixes)
    ]
    file_path = os.path.abspath(inspect.getsourcefile(obj.__class__))  # noqa: PTH100
    file_path = file_path.replace(".py", "")
    assert any(
        (file_path.startswith(p) for p in prefixes),
    ), f"{file_path=}, {prefixes=}, set PYTHONPATH if you run script from __main__"
    for p in prefixes:
        file_path = file_path.replace(p, "")
    return file_path.strip("/").replace("/", ".")
