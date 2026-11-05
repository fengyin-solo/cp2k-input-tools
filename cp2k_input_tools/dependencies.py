"""Scan CP2K input files for their ``@INCLUDE``/``@XCTYPE`` dependency graph.

Unlike :class:`cp2k_input_tools.parser.CP2KInputParser` this module does neither parse the
CP2K XML format specification nor does it validate the keyword contents of an input file.
It merely follows the preprocessor directives (``@INCLUDE``, ``@XCTYPE``, ``@SET`` and
``@IF``/``@ENDIF``) and records which file references which other file, allowing to detect:

* missing include targets,
* cyclic includes and
* the set of files that would end up in the final preprocessor expansion
  (``reachable``: reached from the entry through enabled branches only).

Relative targets are resolved first against the directory of the referencing file and
afterwards against the additionally provided search directories. Every resolved real path
generates exactly one node, independently of how often (or via how many symlinks) it is
referenced.
"""

import enum
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

from .lineiterator import ContinuationLineIterator, LineContinuationError
from .parser_errors import PreprocessorError
from .preprocessor import (
    _CONDITIONAL_MATCH,
    _INCLUDE_MATCH,
    _SET_MATCH,
    _VALID_VAR_NAME_MATCH,
    _ConditionalBlock,
    _Variable,
    resolve_variables,
)
from .tokenizer import COMMENT_CHARS, Context, TokenizerError, tokenize

#: node id used for entries read from a stream without a file name (standard input)
STDIN_NODE_ID = "<stdin>"

#: exit status returned by the check mode when a reachable target is missing
EXIT_MISSING = 2
#: exit status returned by the check mode when a dependency cycle was found
EXIT_CYCLE = 3
#: exit status returned by the check mode when both problems were found
EXIT_MISSING_AND_CYCLE = 4


class NodeKind(str, enum.Enum):
    """the kind of a dependency graph node"""

    ENTRY = "entry"
    INCLUDE = "include"
    XCTYPE = "xctype"
    MISSING = "missing"


class ReferenceType(str, enum.Enum):
    """the preprocessor directive an edge was created with"""

    INCLUDE = "include"
    XCTYPE = "xctype"


class DependencyScanError(Exception):
    """raised when the scanned input contains directives that cannot be evaluated

    This covers, for example, undefined variables used in enabled branches, unknown
    preprocessor directives, unbalanced ``@IF``/``@ENDIF`` blocks or an entry file
    that cannot be opened. It does *not* cover missing include targets and cycles,
    which are reported as part of the :class:`DependencyGraph` instead.
    """

    def __init__(self, message: str, filename: Optional[str] = None, linenr: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.filename = filename
        self.linenr = linenr


@dataclass
class DependencyNode:
    """a node in the dependency graph

    ``id`` and ``path`` are identical for files on disk: they contain the resolved real
    path. Missing targets are identified by the normalized, absolute path of the first
    location they were looked up at.
    """

    id: str
    kind: NodeKind
    path: str


@dataclass
class DependencyEdge:
    """a directed edge from the referencing file (``source``) to a ``target`` node

    ``reference`` contains the filename expression as written in the source file (the
    ``@SET`` variable substitution is applied to *locate* the target, but not to this
    field). ``target`` is ``None`` for references located in a disabled ``@IF`` branch
    whose filename expression could not be evaluated (for example because of an
    undefined variable); such references are never followed.
    """

    source: str
    target: Optional[str]
    type: ReferenceType
    line: int
    disabled: bool
    reference: str


@dataclass
class DependencyGraph:
    """the dependency graph resulting from a scan"""

    entry: str
    nodes: Dict[str, DependencyNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    reachable: set = field(default_factory=set)

    @property
    def missing_nodes(self) -> List[DependencyNode]:
        """all missing target nodes, ordered by their id"""
        return sorted((n for n in self.nodes.values() if n.kind is NodeKind.MISSING), key=lambda n: n.id)

    @property
    def active_missing_nodes(self) -> List[DependencyNode]:
        """missing targets referenced from at least one enabled branch of a reachable file"""
        missing_ids = {node.id for node in self.missing_nodes}
        active_ids = {
            edge.target for edge in self.edges if not edge.disabled and edge.target in missing_ids and edge.source in self.reachable
        }
        return [node for node in self.missing_nodes if node.id in active_ids]

    @property
    def check_status(self) -> int:
        """non-zero status if a reachable target is missing or a cycle exists (0 otherwise)"""
        has_missing = bool(self.active_missing_nodes)
        has_cycle = bool(self.cycles)
        if has_missing and has_cycle:
            return EXIT_MISSING_AND_CYCLE
        if has_missing:
            return EXIT_MISSING
        if has_cycle:
            return EXIT_CYCLE
        return 0

    def _references_for(self, node_id: str) -> List[Dict[str, Any]]:
        references = [
            {
                "source": edge.source,
                "line": edge.line,
                "type": edge.type.value,
                "disabled": edge.disabled,
            }
            for edge in self.edges
            if edge.target == node_id
        ]
        return sorted(references, key=lambda r: (r["source"], r["line"], r["type"]))

    def to_dict(self) -> Dict[str, Any]:
        """return a deterministic, JSON-serializable representation of the graph"""
        nodes = [
            {"id": node.id, "kind": node.kind.value, "path": node.path} for node in sorted(self.nodes.values(), key=lambda n: n.id)
        ]
        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type.value,
                "line": edge.line,
                "disabled": edge.disabled,
                "reference": edge.reference,
            }
            for edge in sorted(
                self.edges,
                key=lambda e: (e.source, e.line, e.type.value, e.target or "", e.reference),
            )
        ]
        missing = [{"id": node.id, "path": node.path, "references": self._references_for(node.id)} for node in self.missing_nodes]
        return {
            "entry": self.entry,
            "nodes": nodes,
            "edges": edges,
            "missing": missing,
            "cycles": sorted(self.cycles),
            "reachable": sorted(self.reachable),
        }

    def to_json(self, indent: int = 2) -> str:
        """return the graph as a stable JSON document (trailing newline included)"""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False) + "\n"

    def to_text(self) -> str:
        """return a human-readable report listing reachable files, missing targets and cycles"""
        data = self.to_dict()
        lines = [f"Entry: {self.entry}", ""]

        lines.append(f"Reachable files ({len(data['reachable'])}):")
        lines.extend(f"  {path}" for path in data["reachable"])
        lines.append("")

        lines.append(f"Missing files ({len(data['missing'])}):")
        if not data["missing"]:
            lines.append("  none")
        for node in data["missing"]:
            lines.append(f"  {node['path']}")
            for ref in node["references"]:
                directive = "XCTYPE" if ref["type"] == "xctype" else "INCLUDE"
                suffix = " [disabled @IF branch]" if ref["disabled"] else ""
                lines.append(f"    referenced via @{directive} at {ref['source']}:{ref['line']}{suffix}")
        lines.append("")

        lines.append(f"Dependency cycles ({len(data['cycles'])}):")
        if not data["cycles"]:
            lines.append("  none")
        else:
            for cycle in data["cycles"]:
                lines.append("  " + " -> ".join(cycle))

        return "\n".join(lines) + "\n"


class DependencyScanner:
    """Scan CP2K input streams for their include dependencies

    :param search_dirs: additional directories searched for relative targets, after the
                        directory of the referencing file (may be given multiple times)
    :param initial_variable_values: preset values for preprocessor variables
    """

    def __init__(self, search_dirs=(), initial_variable_values=None):
        self._search_dirs = [Path(path) for path in search_dirs]
        self._initial_variable_values = initial_variable_values
        self._reset()

    def _reset(self):
        self._varstack: Dict[str, _Variable] = {}
        if self._initial_variable_values:
            self._varstack.update({k.upper(): _Variable(str(v), None) for k, v in self._initial_variable_values.items()})

        self._graph = DependencyGraph(entry="")
        self._stack: List[str] = []
        self._conditional: Optional[_ConditionalBlock] = None

    def scan_path(self, path: Union[str, os.PathLike]) -> DependencyGraph:
        """scan an entry file located at *path* and return its dependency graph

        Re-invoking this method (or :meth:`scan_stream`) on the same scanner resets all
        scan state while keeping the configured search directories and preset variables,
        therefore producing identical results on repeated runs.
        """
        self._reset()

        path = Path(path)
        if not path.is_file():
            raise DependencyScanError(f"entry file '{path}' could not be opened")

        node_id = str(path.resolve())
        self._graph.entry = node_id
        self._add_node(node_id, NodeKind.ENTRY)

        with path.open("r") as fhandle:
            self._scan_file(fhandle, node_id, path.resolve().parent)

        self._assert_conditional_closed()
        return self._graph

    def scan_stream(self, fhandle: TextIO, name: Optional[str] = None) -> DependencyGraph:
        """scan an entry read from an open stream (for example standard input)

        Relative targets are resolved against the current working directory unless the
        stream reports the name of an existing file via its ``name`` attribute.
        """
        self._reset()

        if name is None:
            name = getattr(fhandle, "name", None)

        if name and name != STDIN_NODE_ID:
            named_path = Path(name)
            if named_path.is_file():
                node_id = str(named_path.resolve())
                source_dir = named_path.resolve().parent
            else:
                node_id = STDIN_NODE_ID
                source_dir = Path.cwd()
        else:
            node_id = STDIN_NODE_ID
            source_dir = Path.cwd()

        self._graph.entry = node_id
        self._add_node(node_id, NodeKind.ENTRY)

        self._scan_file(fhandle, node_id, source_dir)

        self._assert_conditional_closed()
        return self._graph

    def _assert_conditional_closed(self):
        if self._conditional is not None:
            raise DependencyScanError("conditional block not closed at end of file")

    def _add_node(self, node_id: str, kind: NodeKind) -> DependencyNode:
        node = self._graph.nodes.get(node_id)
        if node is None:
            node = DependencyNode(id=node_id, kind=kind, path=node_id)
            self._graph.nodes[node_id] = node
        return node

    def _scan_file(self, fhandle: TextIO, source_id: str, source_dir: Path):
        self._stack.append(source_id)
        self._graph.reachable.add(source_id)

        line_iter = ContinuationLineIterator(fhandle)
        try:
            for line in line_iter:
                # ignore empty lines and comments, identical to CP2KPreprocessor.__next__
                if not line or line.startswith(COMMENT_CHARS) or not line.startswith("@"):
                    continue

                try:
                    self._parse_directive(line, source_id, source_dir, line_iter.line_range[1])
                except (PreprocessorError, TokenizerError) as exc:
                    raise DependencyScanError(exc.args[0], filename=source_id, linenr=line_iter.line_range[1]) from None
        except LineContinuationError as exc:
            raise DependencyScanError(str(exc), filename=source_id) from None
        finally:
            self._stack.pop()

    def _parse_directive(self, line: str, source_id: str, source_dir: Path, linenr: int):
        conditional_match = _CONDITIONAL_MATCH.match(line)
        ctx = Context(line=line)

        if conditional_match:
            stmt = conditional_match.group("stmt")
            condition = conditional_match.group("cond").strip()

            if stmt.upper() == "ENDIF":
                if self._conditional is None:
                    raise PreprocessorError("found @ENDIF without a previous @IF", ctx)

                if condition and not condition.startswith(COMMENT_CHARS):
                    ctx.colnr = conditional_match.start("cond")
                    ctx.ref_colnr = conditional_match.end("cond")
                    raise PreprocessorError("garbage found after @ENDIF", ctx)

                self._conditional = None
            else:
                if self._conditional is not None:
                    ctx.ref_line = self._conditional.ctx.line
                    raise PreprocessorError("nested @IF are not allowed", ctx)

                condition = resolve_variables(condition, self._varstack)

                if not condition or condition == "0":
                    enabled = False
                elif "==" in condition:
                    lhs, rhs = [s.strip() for s in condition.split("==", maxsplit=1)]
                    enabled = lhs == rhs
                elif "/=" in condition:
                    lhs, rhs = [s.strip() for s in condition.split("/=", maxsplit=1)]
                    enabled = lhs != rhs
                else:
                    enabled = True

                self._conditional = _ConditionalBlock(enabled, ctx)
            return

        disabled = self._conditional is not None and not self._conditional.condition

        set_match = _SET_MATCH.match(line)
        if set_match:
            if disabled:
                return

            key = set_match.group("var")
            value = resolve_variables(set_match.group("value"), self._varstack)

            if not _VALID_VAR_NAME_MATCH.match(key):
                raise PreprocessorError(f"invalid variable name '{key}'", ctx) from None

            self._varstack[key.upper()] = _Variable(value, ctx)
            return

        include_match = _INCLUDE_MATCH.match(line)
        if include_match:
            inctype = include_match.group("type")
            ref_type = ReferenceType.XCTYPE if inctype.upper() == "XCTYPE" else ReferenceType.INCLUDE
            file_expr = include_match.group("file")
            raw_reference = file_expr.strip()
            col_offset = include_match.start("file")

            try:
                filename = self._parse_include_filename(inctype, file_expr, col_offset)
            except (PreprocessorError, TokenizerError):
                # an expression that cannot be evaluated (e.g. an undefined variable) is
                # only tolerable inside a disabled branch: record it without target there
                if disabled:
                    self._add_edge(source_id, None, ref_type, linenr, True, raw_reference)
                    return
                raise

            node = self._resolve_target_node(filename, ref_type, source_dir)
            edge = self._add_edge(source_id, node.id, ref_type, linenr, disabled, raw_reference)

            # the same source line was already recorded earlier (the file is referenced
            # more than once): nothing new to follow or detect
            if edge is None:
                return

            if disabled or node.kind is NodeKind.MISSING:
                return

            if node.id in self._stack:
                start = self._stack.index(node.id)
                self._graph.cycles.append(self._stack[start:] + [node.id])
                return

            target_path = Path(node.path)
            try:
                with target_path.open("r") as fhandle:
                    self._scan_file(fhandle, node.id, target_path.parent)
            except OSError as exc:
                raise DependencyScanError(
                    f"specified INCLUDE/XCTYPE {filename} could not be opened: {exc}",
                    filename=source_id,
                    linenr=linenr,
                ) from None
            return

        if disabled:
            # unknown directives inside disabled branches are skipped by CP2K as well
            return

        raise PreprocessorError("unknown preprocessor directive found", ctx)

    def _parse_include_filename(self, inctype: str, file_expr: str, col_offset: int) -> str:
        try:
            filename = resolve_variables(file_expr, self._varstack)
        except PreprocessorError as exc:
            exc.args[1].colnr += col_offset
            exc.args[1].ref_colnr += col_offset
            raise

        if filename.startswith(("'", '"')):
            try:
                tokens = tokenize(filename)
            except TokenizerError as exc:
                exc.args[1].colnr += col_offset
                exc.args[1].ref_colnr += col_offset
                raise

            if len(tokens) != 1:
                raise PreprocessorError(
                    "@INCLUDE requires exactly one argument",
                    Context(colnr=col_offset, ref_colnr=col_offset + len(file_expr)),
                )

            filename = tokens[0].strip("'\"")

        if not filename.strip():
            raise PreprocessorError(
                f"@{inctype} requires exactly one argument",
                Context(colnr=col_offset, ref_colnr=col_offset + len(file_expr)),
            )

        filename = filename.strip("'\"")

        if inctype.upper() == "XCTYPE":
            filename = f"xc_section/{filename}.sec"

        return filename

    def _resolve_target_node(self, filename: str, ref_type: ReferenceType, source_dir: Path) -> DependencyNode:
        target = Path(filename)

        # search order: directory of the referencing file first, then the additional
        # search directories in the order they were configured
        if target.is_absolute():
            candidates = [target]
        else:
            candidates = [source_dir / target]
            candidates.extend(search_dir / target for search_dir in self._search_dirs)

        for candidate in candidates:
            if candidate.is_file():
                node_id = str(candidate.resolve())
                return self._add_node(
                    node_id,
                    NodeKind.XCTYPE if ref_type is ReferenceType.XCTYPE else NodeKind.INCLUDE,
                )

        # nothing found: identify the missing target by the normalized absolute path of
        # the first location it was looked up at (deterministic, no file system access)
        missing_id = os.path.normpath(str(candidates[0]))
        return self._add_node(missing_id, NodeKind.MISSING)

    def _add_edge(self, source_id, target_id, ref_type, linenr, disabled, reference) -> Optional[DependencyEdge]:
        # references of the same source line are identical: a multiply-included file must
        # not contribute its transitive edges or cycles more than once, while distinct
        # include sites (different line numbers) remain distinct edges
        for existing in self._graph.edges:
            if (
                existing.source == source_id
                and existing.line == linenr
                and existing.type is ref_type
                and existing.target == target_id
            ):
                return None

        edge = DependencyEdge(
            source=source_id,
            target=target_id,
            type=ref_type,
            line=linenr,
            disabled=disabled,
            reference=reference,
        )
        self._graph.edges.append(edge)
        return edge


def scan_dependencies(
    entry: Union[str, os.PathLike, TextIO, None] = None,
    *,
    search_dirs=(),
    initial_variable_values=None,
) -> DependencyGraph:
    """Scan *entry* for its include dependency graph.

    :param entry: path to the entry file or an open text stream; ``None`` means standard input
    :param search_dirs: additional search roots for relative targets, after the referencing
                        file's directory
    :param initial_variable_values: preset preprocessor variables (``-E key=value`` on the CLI)
    :raises DependencyScanError: for directives that cannot be evaluated or unreadable entries
    """
    scanner = DependencyScanner(search_dirs=search_dirs, initial_variable_values=initial_variable_values)

    if entry is None:
        return scanner.scan_stream(sys.stdin)
    if isinstance(entry, (str, os.PathLike)):
        return scanner.scan_path(entry)
    return scanner.scan_stream(entry)
