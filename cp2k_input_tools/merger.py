"""
Merge a CP2K input (the "base") with another CP2K input (the "override").

Both inputs are expected to have been parsed against the same CP2K input XML
definition (see :class:`CP2KInputParser`). The merge works on the canonical
nested-dict representation produced by :class:`CP2KInputParser`:

* ordinary (non-repeating) sections are merged recursively,
* scalar keywords are replaced by the override value,
* repeating sections with a section parameter (like ``&KIND H``) are matched
  by their parameter and then merged recursively (unmatched entries are
  appended),
* repeating sections without a parameter follow an explicit replace/append
  policy,
* repeating keywords (arrays) follow an explicit replace/append policy,
* with ``only_existing_paths=True`` the override may only touch paths already
  present in the base.

The merged tree is always validated with the :class:`CP2KInputGenerator` and
re-parsed before it is handed back, so anything returned from this module is
guaranteed to be renderable into a valid CP2K input.
"""

import contextlib
import io
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

from . import DEFAULT_CP2K_INPUT_XML
from .generator import (
    CP2KInputGenerator,
    GeneratorError,
    InvalidBooleanDataError,
    InvalidKeywordDataError,
    InvalidSectionDataError,
    KeywordNotFoundError,
    SectionNotFoundError,
    SectionParametersNotFoundError,
    SimplifiedSectionAmbiguityError,
)
from .parser import CP2KInputParser
from .parser_errors import ParserError
from .tokenizer import TokenizerError

GENERATOR_VALIDATION_ERRORS = (
    GeneratorError,
    SectionNotFoundError,
    KeywordNotFoundError,
    SectionParametersNotFoundError,
    InvalidSectionDataError,
    InvalidKeywordDataError,
    InvalidBooleanDataError,
    SimplifiedSectionAmbiguityError,
)


class MergeError(Exception):
    """raised for any structural error encountered while merging"""


class MergeConflictError(MergeError):
    """raised when the override cannot be matched unambiguously (e.g. duplicate section parameters)"""


class MergePathNotFoundError(MergeError):
    """raised in ``only_existing_paths`` mode for paths not present in the base"""


class MergeValidationError(MergeError):
    """raised when the merged tree fails generator validation or the re-parse check"""


class RepeatPolicy(str, Enum):
    """strategy for repeated sections or repeated keywords (arrays)"""

    APPEND = "append"
    REPLACE = "replace"


@dataclass(frozen=True)
class MergePolicy:
    """the policies applied when merging repeated structures"""

    repeated_sections: RepeatPolicy = RepeatPolicy.APPEND
    repeated_keywords: RepeatPolicy = RepeatPolicy.APPEND


DEFAULT_MERGE_POLICY = MergePolicy()

TreeDict = Dict[str, Any]


class CP2KInputMerger:
    def __init__(
        self,
        xmlspec: Union[str, Path] = DEFAULT_CP2K_INPUT_XML,
        policy: MergePolicy = DEFAULT_MERGE_POLICY,
        only_existing_paths: bool = False,
    ):
        """
        :param xmlspec: path to the ``cp2k_input.xml`` definition, must be the same one
                        used to parse (and used to re-validate) the inputs
        :param policy: replace/append policies for repeated sections and keywords
        :param only_existing_paths: if True, the override may only contain paths that
                                    already exist in the base tree
        """
        self._xmlspec = xmlspec
        self._spec = ET.parse(xmlspec)
        self._policy = policy
        self._only_existing = only_existing_paths

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _find_section_node(parent_node: ET.Element, name: str) -> Optional[ET.Element]:
        """return the SECTION XML node matching the given name/alias or None"""
        upper = name.upper()
        for section in parent_node.iterfind("./SECTION"):
            if upper in [e.text.upper() for e in section.iterfind("./NAME") if e.text]:
                return section
        return None

    @staticmethod
    def _find_keyword_node(parent_node: ET.Element, name: str) -> Optional[ET.Element]:
        """return the KEYWORD/DEFAULT_KEYWORD XML node matching the given name/alias or None"""
        if name == "*":
            return parent_node.find("./DEFAULT_KEYWORD")

        upper = name.upper()
        for kw in parent_node.iterfind("./KEYWORD"):
            if upper in [e.text.upper() for e in kw.iterfind("./NAME") if e.text]:
                return kw
        return None

    @staticmethod
    def _canonical_section_name(section_node: ET.Element) -> str:
        return section_node.find("./NAME").text.lower()

    @staticmethod
    def _canonical_keyword_name(kw_node: ET.Element) -> str:
        if kw_node.tag == "DEFAULT_KEYWORD":
            return "*"
        return kw_node.find("./NAME[@type='default']").text.lower()

    # ------------------------------------------------------------- key helpers
    @staticmethod
    def _is_section_key(key: str) -> bool:
        return key.startswith(("+", "&"))

    def _normalize_result_keys(self, xml_node: ET.Element, content: TreeDict, path: str) -> TreeDict:
        """rename alias keys to the canonical schema names so the merge is spelling independent"""
        normalized: TreeDict = {}

        for key, value in content.items():
            if key == "_":
                normalized["_"] = value
                continue

            ckey = key
            if self._is_section_key(key):
                sec_node = self._find_section_node(xml_node, key[1:])
                if sec_node is not None:
                    ckey = f"+{self._canonical_section_name(sec_node)}"
            else:
                kw_node = self._find_keyword_node(xml_node, key)
                if kw_node is not None:
                    ckey = self._canonical_keyword_name(kw_node)

            if ckey in normalized:
                raise MergeConflictError(f"multiple aliases for the same path '{self._join(path, ckey)}'")

            normalized[ckey] = deepcopy(value)

        return normalized

    @staticmethod
    def _join(path: str, key: str) -> str:
        return f"{path}/{key}" if path else f"/{key}"

    @staticmethod
    def _as_section_list(value: Any, path: str) -> List[TreeDict]:
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise MergeError(f"expected a section or a list of sections at '{path}'")
        return value

    @staticmethod
    def _as_keyword_list(value: Any) -> list:
        return value if isinstance(value, list) else [value]

    # ------------------------------------------------------------------ merge
    def merge_trees(self, base: TreeDict, override: TreeDict) -> TreeDict:
        """recursively merge the override tree into the base tree and return a new tree"""
        return self._merge_section(self._spec.getroot(), base, override, "")

    def _merge_section(self, xml_node: ET.Element, base: TreeDict, override: TreeDict, path: str) -> TreeDict:
        if not isinstance(base, dict) or not isinstance(override, dict):
            raise MergeError(f"can only merge section contents (dictionaries) at '{path or '/'}'")

        result = self._normalize_result_keys(xml_node, base, path)

        for key, over_value in override.items():
            if key == "_":
                # section parameter: only reached for matched/new parameterized sections
                result["_"] = deepcopy(over_value)
                continue

            if self._is_section_key(key):
                self._merge_section_value(xml_node, result, key, over_value, path)
            else:
                self._merge_keyword_value(xml_node, result, key, over_value, path)

        return result

    def _merge_section_value(self, xml_node: ET.Element, result: TreeDict, key: str, over_value: Any, path: str) -> None:
        sec_node = self._find_section_node(xml_node, key[1:])
        if sec_node is None:
            # without schema metadata the merge cannot decide how to combine this key;
            # a valid (parsed) tree can never end up here
            raise MergeError(f"unknown section '{key[1:]}' at '{path or '/'}'")

        ckey = f"+{self._canonical_section_name(sec_node)}"
        cpath = self._join(path, ckey)
        repeats = sec_node.get("repeats") == "yes"

        if not repeats:
            if ckey in result:
                result[ckey] = self._merge_section(sec_node, result[ckey], self._as_dict(over_value, cpath), cpath)
            else:
                if self._only_existing:
                    raise MergePathNotFoundError(f"section '{cpath}' is not present in the base input")
                result[ckey] = self._merge_section(sec_node, {}, self._as_dict(over_value, cpath), cpath)
            return

        base_entries = self._as_section_list(result.get(ckey, []), cpath)
        over_entries = self._as_section_list(over_value, cpath)

        if sec_node.find("./SECTION_PARAMETERS") is not None:
            result[ckey] = self._merge_parameterized_sections(sec_node, base_entries, over_entries, cpath)
        else:
            result[ckey] = self._merge_unparameterized_sections(sec_node, base_entries, over_entries, cpath)

    @staticmethod
    def _as_dict(value: Any, path: str) -> TreeDict:
        if not isinstance(value, dict):
            raise MergeError(f"expected section contents at '{path}'")
        return value

    def _merge_parameterized_sections(
        self, sec_node: ET.Element, base_entries: List[TreeDict], over_entries: List[TreeDict], path: str
    ) -> List[TreeDict]:
        merged = [deepcopy(entry) for entry in base_entries]

        # duplicate parameters within the override itself cannot be applied deterministically
        seen = set()
        for entry in over_entries:
            param = entry.get("_")
            if param in seen:
                raise MergeConflictError(f"section '{path}' contains multiple override entries with the same parameter {param!r}")
            seen.add(param)

        for entry in over_entries:
            param = entry.get("_")
            matches = [idx for idx, existing in enumerate(merged) if existing.get("_") == param]

            if len(matches) > 1:
                raise MergeConflictError(
                    f"parameter {param!r} for section '{path}' matches {len(matches)} sections in the base input,"
                    " cannot merge unambiguously"
                )

            if matches:
                merged[matches[0]] = self._merge_section(sec_node, merged[matches[0]], entry, f"{path}[{param!r}]")
            elif self._only_existing:
                raise MergePathNotFoundError(f"section '{path}' with parameter {param!r} is not present in the base input")
            else:
                merged.append(self._merge_section(sec_node, {}, entry, f"{path}[{param!r}]"))

        return merged

    def _merge_unparameterized_sections(
        self, sec_node: ET.Element, base_entries: List[TreeDict], over_entries: List[TreeDict], path: str
    ) -> List[TreeDict]:
        if self._only_existing:
            # without a parameter there is no stable identity to match an override against
            raise MergePathNotFoundError(f"section '{path}' is repeated without parameters: existing paths cannot be identified")

        if self._policy.repeated_sections == RepeatPolicy.REPLACE:
            merged: List[TreeDict] = []
        else:
            merged = [deepcopy(entry) for entry in base_entries]

        for entry in over_entries:
            merged.append(self._merge_section(sec_node, {}, entry, path))

        return merged

    def _merge_keyword_value(self, xml_node: ET.Element, result: TreeDict, key: str, over_value: Any, path: str) -> None:
        kw_node = self._find_keyword_node(xml_node, key)

        if kw_node is None:
            # unknown keyword: nothing structurally mergeable, pass it through and let the
            # mandatory generator validation reject it
            if self._only_existing:
                raise MergePathNotFoundError(f"keyword '{self._join(path, key)}' is not present in the base input")
            result[key] = deepcopy(over_value)
            return

        ckey = self._canonical_keyword_name(kw_node)
        cpath = self._join(path, ckey)
        repeats = kw_node.get("repeats") == "yes"

        if ckey not in result:
            if self._only_existing:
                raise MergePathNotFoundError(f"keyword '{cpath}' is not present in the base input")
            result[ckey] = deepcopy(self._as_keyword_list(over_value) if repeats else over_value)
            return

        if not repeats:
            # scalar replacement
            result[ckey] = deepcopy(over_value)
            return

        base_values = self._as_keyword_list(result[ckey])
        over_values = self._as_keyword_list(over_value)

        if self._policy.repeated_keywords == RepeatPolicy.REPLACE:
            result[ckey] = deepcopy(over_values)
        else:
            result[ckey] = base_values + deepcopy(over_values)

    # ------------------------------------------------------------- validation
    def render(self, tree: TreeDict) -> List[str]:
        """run the merged tree through the generator, raising on any schema violation"""
        generator = CP2KInputGenerator(self._xmlspec)
        try:
            return list(generator.line_iter(tree))
        except GENERATOR_VALIDATION_ERRORS as exc:
            raise MergeValidationError(f"merged input failed generator validation: {exc}") from exc

    def verify(self, tree: TreeDict) -> List[str]:
        """generate the output text and verify it re-parses into exactly the merged tree"""
        lines = self.render(tree)

        stream = io.StringIO("\n".join(lines))
        stream.name = "<merged>"  # required for parser error contexts

        parser = CP2KInputParser(self._xmlspec, key_trafo=str.lower)
        try:
            reparsed = parser.parse(stream)
        except (ParserError, TokenizerError) as exc:
            raise MergeValidationError(f"merged output could not be re-parsed: {exc}") from exc

        # the generator emits repeated sections in (reversed) stack order, so their
        # list order is inverted in the textual output; this is a deterministic,
        # purely positional effect — undo it before the structural comparison
        self._restore_section_order(self._spec.getroot(), reparsed)

        if reparsed != tree:
            raise MergeValidationError("merged output does not re-parse into the merged tree")

        return lines

    def _restore_section_order(self, xml_node: ET.Element, content: TreeDict) -> None:
        """reverse the repeated-section lists introduced by the generator's stack-based rendering"""
        if not isinstance(content, dict):
            return

        for key, value in content.items():
            if not self._is_section_key(key):
                continue

            sec_node = self._find_section_node(xml_node, key[1:])
            if sec_node is None:
                continue

            if sec_node.get("repeats") == "yes" and isinstance(value, list):
                value.reverse()
                entries = value
            elif isinstance(value, dict):
                entries = [value]
            else:
                continue

            for entry in entries:
                self._restore_section_order(sec_node, entry)


def merge_inputs(
    base_fhandle,
    override_fhandle,
    *,
    xmlspec: Union[str, Path, None] = None,
    base_dir: Union[str, Path] = ".",
    initial_variable_values: Optional[dict] = None,
    policy: MergePolicy = DEFAULT_MERGE_POLICY,
    only_existing_paths: bool = False,
) -> Tuple[TreeDict, List[str]]:
    """
    Parse two CP2K input handles against the same XML definition, merge them and
    validate the result.

    :return: the merged canonical tree and the validated lines of the merged CP2K input
    :raises ParserError/TokenizerError: if either input is invalid against the schema
    :raises MergeError: for structural merge conflicts or validation failures
    """
    if xmlspec is None:
        xmlspec = DEFAULT_CP2K_INPUT_XML

    base_parser = CP2KInputParser(xmlspec, base_dir=base_dir, key_trafo=str.lower)
    base_tree = base_parser.parse(base_fhandle, initial_variable_values)

    override_parser = CP2KInputParser(xmlspec, base_dir=base_dir, key_trafo=str.lower)
    override_tree = override_parser.parse(override_fhandle, initial_variable_values)

    merger = CP2KInputMerger(xmlspec, policy=policy, only_existing_paths=only_existing_paths)
    merged_tree = merger.merge_trees(base_tree, override_tree)
    lines = merger.verify(merged_tree)

    return merged_tree, lines


def write_lines_atomic(path: Union[str, Path], lines: List[str]) -> None:
    """
    Write the given CP2K input lines to ``path`` atomically: the content is written
    to a temporary file in the same directory, fsync'ed and then moved into place.
    On error the destination is left untouched and the temporary file is removed.
    """
    destination = Path(path)
    parent = destination.parent if str(destination.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmpname = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(parent))
    tmp_path = Path(tmpname)

    try:
        with os.fdopen(fd, "w") as fhandle:
            for line in lines:
                fhandle.write(f"{line}\n")
            fhandle.flush()
            os.fsync(fhandle.fileno())
        os.replace(tmp_path, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise
