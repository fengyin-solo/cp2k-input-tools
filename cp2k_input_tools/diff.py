"""Semantic comparison of two CP2K input files.

Both inputs are parsed with the same CP2K XML input definition using the canonical
parser, so differences in formatting, indentation, casing, quoting, Fortran-style
floating point notation (``1.0d-8``), comments or the way ``@include`` files were
split up do not show up in the comparison: only the resulting option tree is
compared.

The produced paths uniquely identify a value in the input tree:

* sections are joined with ``/``, e.g. ``force_eval[0]/dft/mgrid/cutoff``
* repeatable sections carry their zero-based index: ``kind[1]``
* the section parameter (the value after a section opening line such as
  ``&XC_FUNCTIONAL PBE``) is addressed with ``_``
* elements of multi-value keywords carry an array index: ``cell/a[0]``
* values of repeatable keywords are addressed with ``[<occurrence>]``

The public entry points are:

* :func:`diff_trees` -- compare two trees as returned by
  :class:`cp2k_input_tools.parser.CP2KInputParser`
* :func:`diff_inputs` -- compare two open file handles (parsing both before comparing)
* :func:`diff_paths` -- convenience wrapper comparing two files by path
"""

import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, TextIO, Tuple, Union

from .lineiterator import LineContinuationError
from .parser import CP2KInputParser
from .parser_errors import ParserError
from .tokenizer import TokenizerError

#: operation marking a value only present in the right ("new") input
OP_ADDED = "added"
#: operation marking a value only present in the left ("old") input
OP_REMOVED = "removed"
#: operation marking a value present in both inputs, but with different values
OP_CHANGED = "changed"

_MISSING = object()

_SEGMENT_MATCH = re.compile(r"^(?P<name>[^\[\]]*)(?P<indices>(?:\[\d+\])*)$")
_INDICES_MATCH = re.compile(r"\[(\d+)\]")


@dataclass
class DiffOptions:
    """Options steering the semantic comparison.

    :param abs_tol: absolute tolerance accepted when comparing two floating point values
    :param rel_tol: relative tolerance accepted when comparing two floating point values
    :param ignore_section_order: do not consider the order of repeatable sections significant
    """

    abs_tol: float = 0.0
    rel_tol: float = 0.0
    ignore_section_order: bool = False


@dataclass
class DiffEntry:
    """A single semantic difference between the two inputs.

    :param op: one of :data:`OP_ADDED`, :data:`OP_REMOVED` or :data:`OP_CHANGED`
    :param path: the stable path identifying the value, e.g. ``force_eval[0]/dft/mgrid/cutoff``
    :param value_left: value in the left input (``None`` for additions)
    :param value_right: value in the right input (``None`` for removals)
    """

    op: str
    path: str
    value_left: Any = None
    value_right: Any = None


@dataclass
class DiffResult:
    """The result of a semantic comparison."""

    entries: List[DiffEntry] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """return True if the two inputs are semantically identical"""
        return not self.entries

    @property
    def added(self) -> List[DiffEntry]:
        return [e for e in self.entries if e.op == OP_ADDED]

    @property
    def removed(self) -> List[DiffEntry]:
        return [e for e in self.entries if e.op == OP_REMOVED]

    @property
    def changed(self) -> List[DiffEntry]:
        return [e for e in self.entries if e.op == OP_CHANGED]


@dataclass
class ParseFailure:
    """A failed parse attempt of one of the two compared inputs."""

    side: str
    source: str
    message: str
    filename: Optional[str] = None
    linenr: Optional[int] = None


class CP2KDiffParseError(Exception):
    """raised when at least one of the two inputs could not be parsed.

    All failures are collected so that no partial diff is ever reported.
    """

    def __init__(self, failures: Sequence[ParseFailure]):
        self.failures = list(failures)
        super().__init__("; ".join(str(f) for f in self.failures))


def _path_sort_key(path: str) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """numeric aware sort key so that ``[2]`` sorts before ``[10]``"""
    key = []
    for segment in path.split("/"):
        match = _SEGMENT_MATCH.match(segment)
        indices = tuple(int(i) for i in _INDICES_MATCH.findall(match.group("indices")))
        key.append((match.group("name"), indices))
    return tuple(key)


def _join(prefix: Tuple[str, ...], *segments: str) -> str:
    return "/".join(prefix + segments)


def _scalars_equal(left: Any, right: Any, options: DiffOptions) -> bool:
    # booleans are a subclass of int in Python, but a logical keyword changing
    # between True and 1 must never be considered equal
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=options.rel_tol, abs_tol=options.abs_tol)

    return left == right


def _value_leaves(value: Any) -> List[Tuple[Tuple[int, ...], Any]]:
    """flatten a keyword/section-parameter value into ``(index-tuple, scalar)`` leaves"""
    if isinstance(value, list):  # repeatable keyword: list of scalar or tuple values
        leaves: List[Tuple[Tuple[int, ...], Any]] = []
        for occurrence, item in enumerate(value):
            if isinstance(item, tuple):
                leaves += [((occurrence, index), scalar) for index, scalar in enumerate(item)]
            else:
                leaves.append(((occurrence,), item))
        return leaves

    if isinstance(value, tuple):  # multi-value keyword
        return [((index,), scalar) for index, scalar in enumerate(value)]

    return [((), value)]


def _suffix_path(prefix: Tuple[str, ...], name: str, indices: Tuple[int, ...]) -> str:
    return _join(prefix, name + "".join(f"[{i}]" for i in indices))


def _diff_value(
    left: Any,
    right: Any,
    prefix: Tuple[str, ...],
    name: str,
    options: DiffOptions,
) -> List[DiffEntry]:
    """compare two keyword (or section-parameter) values"""
    left_leaves = {} if left is _MISSING else dict(_value_leaves(left))
    right_leaves = {} if right is _MISSING else dict(_value_leaves(right))

    entries = []
    for indices in sorted(set(left_leaves) | set(right_leaves)):
        path = _suffix_path(prefix, name, indices)
        left_value = left_leaves.get(indices, _MISSING)
        right_value = right_leaves.get(indices, _MISSING)

        if left_value is _MISSING:
            entries.append(DiffEntry(OP_ADDED, path, None, right_value))
        elif right_value is _MISSING:
            entries.append(DiffEntry(OP_REMOVED, path, left_value, None))
        elif not _scalars_equal(left_value, right_value, options):
            entries.append(DiffEntry(OP_CHANGED, path, left_value, right_value))

    return entries


def _partition(section: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """split a canonical section dict into parameters, sections and keywords"""
    parameters: Dict[str, Any] = {}
    sections: Dict[str, Any] = {}
    keywords: Dict[str, Any] = {}

    for key, value in section.items():
        if key == "_":  # section parameter
            parameters[key] = value
        elif key.startswith("+"):  # sections always carry the '+' prefix in the canonical tree
            sections[key[1:]] = value
        else:
            keywords[key] = value

    return parameters, sections, keywords


def _collect_section(section: Dict[str, Any], prefix: Tuple[str, ...]) -> List[Tuple[str, Any]]:
    """return all ``(path, scalar)`` leaves of a section (used for one-sided additions/removals)"""
    leaves: List[Tuple[str, Any]] = []
    parameters, sections, keywords = _partition(section)

    for name, value in parameters.items():
        leaves += [(_suffix_path(prefix, name, indices), scalar) for indices, scalar in _value_leaves(value)]

    for name, value in keywords.items():
        leaves += [(_suffix_path(prefix, name, indices), scalar) for indices, scalar in _value_leaves(value)]

    for name, value in sorted(sections.items()):
        leaves += _collect_section_value(name, value, prefix)

    return leaves


def _collect_section_value(name: str, value: Any, prefix: Tuple[str, ...]) -> List[Tuple[str, Any]]:
    if isinstance(value, dict):
        return _collect_section(value, prefix + (name,))

    return [leaf for index, subsection in enumerate(value) for leaf in _collect_section(subsection, prefix + (f"{name}[{index}]",))]


def _diff_section_list(
    name: str,
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    prefix: Tuple[str, ...],
    options: DiffOptions,
) -> List[DiffEntry]:
    """compare two lists of repeatable sections, honoring the order option"""
    entries: List[DiffEntry] = []

    if not options.ignore_section_order:
        for index in range(max(len(left), len(right))):
            segment = f"{name}[{index}]"
            section_prefix = prefix + (segment,)
            if index < len(left) and index < len(right):
                entries += _diff_section(left[index], right[index], section_prefix, options)
            elif index < len(left):
                entries += [
                    DiffEntry(OP_REMOVED, path, value, None) for path, value in _collect_section(left[index], section_prefix)
                ]
            else:
                entries += [
                    DiffEntry(OP_ADDED, path, None, value) for path, value in _collect_section(right[index], section_prefix)
                ]
        return entries

    # order-insensitive comparison: pair sections using a greedy minimum-cost
    # (maximum similarity) one-to-one assignment based on the recursive diff size
    candidates = []
    for left_index, left_section in enumerate(left):
        for right_index, right_section in enumerate(right):
            cost = len(_diff_section(left_section, right_section, (), options))
            candidates.append((cost, left_index, right_index))

    candidates.sort()

    pairing: Dict[int, int] = {}
    used_left: Set[int] = set()
    used_right: Set[int] = set()
    for _cost, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        pairing[left_index] = right_index
        used_left.add(left_index)
        used_right.add(right_index)
        if len(pairing) == min(len(left), len(right)):
            break

    for left_index, left_section in enumerate(left):
        segment = f"{name}[{left_index}]"
        if left_index in pairing:
            # paths of matched sections use the left-side index
            entries += _diff_section(left_section, right[pairing[left_index]], prefix + (segment,), options)
        else:
            entries += [
                DiffEntry(OP_REMOVED, path, value, None) for path, value in _collect_section(left_section, prefix + (segment,))
            ]

    for right_index, right_section in enumerate(right):
        if right_index not in pairing.values():
            segment = f"{name}[{right_index}]"
            entries += [
                DiffEntry(OP_ADDED, path, None, value) for path, value in _collect_section(right_section, prefix + (segment,))
            ]

    return entries


def _diff_section_value(
    name: str,
    left: Any,
    right: Any,
    prefix: Tuple[str, ...],
    options: DiffOptions,
) -> List[DiffEntry]:
    if left is _MISSING:
        return [DiffEntry(OP_ADDED, path, None, value) for path, value in _collect_section_value(name, right, prefix)]

    if right is _MISSING:
        return [DiffEntry(OP_REMOVED, path, value, None) for path, value in _collect_section_value(name, left, prefix)]

    if isinstance(left, dict) and isinstance(right, dict):
        return _diff_section(left, right, prefix + (name,), options)

    if isinstance(left, list) and isinstance(right, list):
        return _diff_section_list(name, left, right, prefix, options)

    # same schema on both sides makes this unreachable in practice,
    # handle it defensively as a removal/addition:
    entries = [DiffEntry(OP_REMOVED, path, value, None) for path, value in _collect_section_value(name, left, prefix)]
    entries += [DiffEntry(OP_ADDED, path, None, value) for path, value in _collect_section_value(name, right, prefix)]
    return entries


def _diff_section(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefix: Tuple[str, ...],
    options: DiffOptions,
) -> List[DiffEntry]:
    """recursively compare two canonical parsed section dictionaries"""
    left_params, left_sections, left_keywords = _partition(left)
    right_params, right_sections, right_keywords = _partition(right)

    entries: List[DiffEntry] = []

    # section parameter
    if "_" in left_params or "_" in right_params:
        entries += _diff_value(left_params.get("_", _MISSING), right_params.get("_", _MISSING), prefix, "_", options)

    # keywords (their order is semantically irrelevant in CP2K, compare by name)
    for keyword_name in sorted(set(left_keywords) | set(right_keywords)):
        entries += _diff_value(
            left_keywords.get(keyword_name, _MISSING),
            right_keywords.get(keyword_name, _MISSING),
            prefix,
            keyword_name,
            options,
        )

    # subsections
    for section_name in sorted(set(left_sections) | set(right_sections)):
        entries += _diff_section_value(
            section_name,
            left_sections.get(section_name, _MISSING),
            right_sections.get(section_name, _MISSING),
            prefix,
            options,
        )

    return entries


def diff_trees(
    tree_left: Dict[str, Any],
    tree_right: Dict[str, Any],
    options: Optional[DiffOptions] = None,
) -> DiffResult:
    """compare two canonical trees as returned by :class:`CP2KInputParser`.

    :param tree_left: parsed left ("old") input
    :param tree_right: parsed right ("new") input
    :param options: comparison options (tolerances, section-order handling)
    :return: a :class:`DiffResult` with entries sorted in a stable order
    """
    if options is None:
        options = DiffOptions()

    entries = _diff_section(tree_left, tree_right, (), options)
    entries.sort(key=lambda entry: _path_sort_key(entry.path))

    return DiffResult(entries)


def parse_input(
    fhandle: TextIO,
    base_dir: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]] = ".",
    var_values: Optional[Dict[str, str]] = None,
    xmlspec: Union[str, pathlib.Path, None] = None,
) -> Dict[str, Any]:
    """parse a CP2K input using the canonical parser.

    :param fhandle: open file handle of the input
    :param base_dir: directory or list of directories searched for ``@include`` files
    :param var_values: optional mapping of preprocessor variable values
    :param xmlspec: optional alternative ``cp2k_input.xml`` schema location
    :raises CP2KDiffParseError: if the input cannot be parsed
    """
    source = getattr(fhandle, "name", "<stream>")
    parser = CP2KInputParser(xmlspec=xmlspec, base_dir=base_dir, key_trafo=str.lower)

    try:
        return parser.parse(fhandle, dict(var_values) if var_values is not None else None)
    except (ParserError, TokenizerError, LineContinuationError, OSError) as exc:
        filename = None
        linenr = None
        message = None

        context = getattr(exc, "args", None)
        if context and len(context) > 1 and hasattr(context[1], "filename"):
            message = context[0]
            filename = context[1].filename
            linenr = context[1].linenr

        failure = ParseFailure(
            side="",
            source=str(source),
            message=message or str(exc),
            filename=filename,
            linenr=linenr,
        )
        raise CP2KDiffParseError([failure]) from exc


def _parse_side(
    side: str,
    fhandle: TextIO,
    base_dir: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]],
    var_values: Optional[Dict[str, str]],
    xmlspec: Union[str, pathlib.Path, None],
) -> Tuple[Optional[Dict[str, Any]], Optional[ParseFailure]]:
    try:
        return parse_input(fhandle, base_dir=base_dir, var_values=var_values, xmlspec=xmlspec), None
    except CP2KDiffParseError as exc:
        failure = exc.failures[0]
        failure.side = side
        return None, failure


def diff_inputs(
    fhandle_left: TextIO,
    fhandle_right: TextIO,
    options: Optional[DiffOptions] = None,
    *,
    base_dir_left: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]] = ".",
    base_dir_right: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]] = ".",
    var_values_left: Optional[Dict[str, str]] = None,
    var_values_right: Optional[Dict[str, str]] = None,
    xmlspec: Union[str, pathlib.Path, None] = None,
) -> DiffResult:
    """parse and compare two open CP2K input files.

    Both inputs are parsed in full before the comparison starts: if one of them
    cannot be parsed, a :class:`CP2KDiffParseError` containing every failure
    (and naming the offending file, including included files) is raised instead
    of a partial diff.
    """
    tree_left, failure_left = _parse_side("left", fhandle_left, base_dir_left, var_values_left, xmlspec)
    tree_right, failure_right = _parse_side("right", fhandle_right, base_dir_right, var_values_right, xmlspec)

    failures = [failure for failure in (failure_left, failure_right) if failure is not None]
    if failures:
        raise CP2KDiffParseError(failures)

    # both trees are complete here, parsing succeeded for both sides
    assert tree_left is not None and tree_right is not None
    return diff_trees(tree_left, tree_right, options)


def diff_paths(
    path_left: Union[str, pathlib.Path],
    path_right: Union[str, pathlib.Path],
    options: Optional[DiffOptions] = None,
    *,
    base_dir_left: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]] = ".",
    base_dir_right: Union[str, pathlib.Path, Sequence[Union[str, pathlib.Path]]] = ".",
    var_values_left: Optional[Dict[str, str]] = None,
    var_values_right: Optional[Dict[str, str]] = None,
    xmlspec: Union[str, pathlib.Path, None] = None,
) -> DiffResult:
    """parse and compare two CP2K input files given by path"""
    with open(path_left, "r") as fhandle_left, open(path_right, "r") as fhandle_right:
        return diff_inputs(
            fhandle_left,
            fhandle_right,
            options,
            base_dir_left=base_dir_left,
            base_dir_right=base_dir_right,
            var_values_left=var_values_left,
            var_values_right=var_values_right,
            xmlspec=xmlspec,
        )
