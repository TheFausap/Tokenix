"""An offline, deterministic English corpus: Python standard-library docstrings."""

from __future__ import annotations

import importlib
import inspect

MODULES = [
    "argparse", "asyncio", "calendar", "collections", "concurrent.futures", "configparser",
    "contextlib", "csv", "dataclasses", "datetime", "decimal", "difflib", "email", "enum",
    "fractions", "functools", "heapq", "http.client", "inspect", "ipaddress", "itertools",
    "json", "logging", "mailbox", "multiprocessing", "numbers", "optparse", "os", "pathlib",
    "pickle", "pprint", "random", "re", "shutil", "socket", "sqlite3", "statistics", "string",
    "subprocess", "tarfile", "tempfile", "textwrap", "threading", "tkinter", "turtle",
    "typing", "unittest", "urllib.parse", "uuid", "xml.dom.minidom", "zipfile",
]


def stdlib_docstrings(modules: list[str] = MODULES) -> str:
    """Concatenate module, class and function docstrings of ``modules``."""
    seen: set[int] = set()
    parts: list[str] = []

    def add(obj) -> None:
        doc = inspect.getdoc(obj)
        if doc and id(doc) not in seen:
            seen.add(id(doc))
            parts.append(doc)

    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        add(mod)
        for _, obj in sorted(vars(mod).items()):
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj):
                add(obj)
            if inspect.isclass(obj):
                for _, meth in sorted(vars(obj).items()):
                    if inspect.isfunction(meth):
                        add(meth)
    return "\n\n".join(parts)
