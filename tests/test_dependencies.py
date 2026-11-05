import io
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from cp2k_input_tools.cli.deps import cp2k_deps
from cp2k_input_tools.dependencies import (
    EXIT_CYCLE,
    EXIT_MISSING,
    EXIT_MISSING_AND_CYCLE,
    DependencyScanError,
    DependencyScanner,
    NodeKind,
    ReferenceType,
    scan_dependencies,
)

from . import TEST_DIR

INPUTS_DIR = TEST_DIR / "inputs"


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def edge_map(graph):
    """map (source-name, target-name/None, directive type) -> list of (line, disabled)"""
    names = {node.id: Path(node.id).name for node in graph.nodes.values()}
    names[None] = None
    result = {}
    for edge in graph.edges:
        result.setdefault((names[edge.source], names.get(edge.target) if edge.target else None, edge.type.value), []).append(
            (edge.line, edge.disabled)
        )
    return result


def node_kind(graph, name):
    return next(node.kind for node in graph.nodes.values() if Path(node.id).name == name)


# ---------------------------------------------------------------------------------------
# library API
# ---------------------------------------------------------------------------------------


def test_multilevel_includes(tmp_path):
    write(tmp_path / "level2.inp", "&END\n")
    write(tmp_path / "level1.inp", "!\n@INCLUDE level2.inp\n")
    entry = write(tmp_path / "entry.inp", "@INCLUDE  level1.inp\n&GLOBAL\n&END GLOBAL\n")

    graph = scan_dependencies(entry)
    data = graph.to_dict()

    assert graph.entry == str(entry.resolve())
    assert node_kind(graph, "entry.inp") is NodeKind.ENTRY
    assert node_kind(graph, "level1.inp") is NodeKind.INCLUDE
    assert node_kind(graph, "level2.inp") is NodeKind.INCLUDE

    edges = edge_map(graph)
    assert edges[("entry.inp", "level1.inp", "include")] == [(1, False)]
    assert edges[("level1.inp", "level2.inp", "include")] == [(2, False)]

    assert data["missing"] == []
    assert data["cycles"] == []
    assert data["reachable"] == sorted(str(p.resolve()) for p in [entry, tmp_path / "level1.inp", tmp_path / "level2.inp"])
    assert graph.check_status == 0


def test_xctype_mapping_from_referrers_directory(tmp_path):
    secdir = tmp_path / "xc_section"
    write(secdir / "PBE.sec", "&XC\n&END XC\n")
    entry = write(tmp_path / "entry.inp", "@XCTYPE PBE\n")

    graph = scan_dependencies(entry)

    assert node_kind(graph, "PBE.sec") is NodeKind.XCTYPE
    edge = next(edge for edge in graph.edges if Path(edge.source).name == "entry.inp")
    assert edge.type is ReferenceType.XCTYPE
    assert edge.target == str((secdir / "PBE.sec").resolve())
    assert edge.reference == "PBE"
    assert edge.disabled is False
    assert str((secdir / "PBE.sec").resolve()) in graph.reachable


def test_xctype_with_shipped_section():
    graph = scan_dependencies(INPUTS_DIR / "He_PBE.inp")
    data = graph.to_dict()

    assert any(node["kind"] == "xctype" and node["id"].endswith("xc_section/PBE.sec") for node in data["nodes"])
    assert any(edge["type"] == "xctype" and edge["reference"] == "PBE" for edge in data["edges"])


def test_xctype_missing(tmp_path):
    entry = write(tmp_path / "entry.inp", "@XCTYPE DOES_NOT_EXIST\n")

    graph = scan_dependencies(entry)

    missing = graph.missing_nodes
    assert len(missing) == 1
    assert missing[0].path.endswith(os.path.join("xc_section", "DOES_NOT_EXIST.sec"))
    assert graph.active_missing_nodes == missing
    assert graph.check_status == EXIT_MISSING


def test_multiple_search_roots_and_priority(tmp_path):
    work = tmp_path / "work"
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    for directory in (work, root1, root2):
        directory.mkdir()

    write(root2 / "shared.inp", "! from root2\n")
    entry = write(work / "entry.inp", "@INCLUDE shared.inp\n")

    # not found without search roots (referrer directory has no such file)
    graph = scan_dependencies(entry)
    assert graph.check_status == EXIT_MISSING

    # found in the second search root
    graph = scan_dependencies(entry, search_dirs=[root1, root2])
    assert graph.check_status == 0
    assert edge_map(graph)[("entry.inp", "shared.inp", "include")] == [(1, False)]
    assert Path(graph.missing_nodes[0].path if graph.missing_nodes else root2 / "shared.inp").name == "shared.inp"
    assert str((root2 / "shared.inp").resolve()) in graph.reachable

    # the referencing file's directory takes precedence over the search roots
    local = write(work / "local.inp", "! local wins\n")
    write(root2 / "local.inp", "! root2 copy\n")
    entry2 = write(work / "entry2.inp", "@INCLUDE local.inp\n")
    graph = scan_dependencies(entry2, search_dirs=[root1, root2])
    edge = graph.edges[0]
    assert edge.target == str(local.resolve())


def test_same_file_referenced_multiple_times_is_one_node(tmp_path):
    write(tmp_path / "deep.inp", "!\n")
    write(tmp_path / "shared.inp", "@INCLUDE deep.inp\n")
    write(tmp_path / "a.inp", "@INCLUDE shared.inp\n")
    write(tmp_path / "b.inp", "@INCLUDE shared.inp\n")
    entry = write(tmp_path / "entry.inp", "@INCLUDE a.inp\n@INCLUDE b.inp\n@INCLUDE shared.inp\n")

    graph = scan_dependencies(entry)
    data = graph.to_dict()

    shared_ids = {node["id"] for node in data["nodes"] if node["id"].endswith("shared.inp")}
    assert len(shared_ids) == 1
    assert len(data["nodes"]) == 5  # entry, a, b, shared, deep -> one node each

    edges = edge_map(graph)
    assert edges[("entry.inp", "a.inp", "include")] == [(1, False)]
    assert edges[("entry.inp", "b.inp", "include")] == [(2, False)]
    assert edges[("entry.inp", "shared.inp", "include")] == [(3, False)]
    assert edges[("a.inp", "shared.inp", "include")] == [(1, False)]
    assert edges[("b.inp", "shared.inp", "include")] == [(1, False)]
    # the transitive edge of the thrice-included shared.inp is recorded only once
    assert edges[("shared.inp", "deep.inp", "include")] == [(1, False)]

    assert len(data["reachable"]) == 5
    assert data["missing"] == []
    assert data["cycles"] == []


def test_symlink_aliases_dedupe_by_real_path(tmp_path):
    real = write(tmp_path / "real.inp", "!\n")
    try:
        os.symlink(real, tmp_path / "alias.inp")
    except OSError:
        pytest.skip("symlinks not supported on this platform")

    entry = write(tmp_path / "entry.inp", "@INCLUDE real.inp\n@INCLUDE alias.inp\n")
    graph = scan_dependencies(entry)

    included = {node.id for node in graph.nodes.values() if node.kind is NodeKind.INCLUDE}
    assert included == {str(real.resolve())}


def test_disabled_conditional_branch(tmp_path):
    write(tmp_path / "yes.inp", "!\n")
    entry = write(
        tmp_path / "entry.inp",
        "@IF 0\n@INCLUDE ghost.inp\n@ENDIF\n@IF 1\n@INCLUDE yes.inp\n@ENDIF\n",
    )

    graph = scan_dependencies(entry)
    data = graph.to_dict()

    ghost_edge = next(edge for edge in data["edges"] if edge["reference"] == "ghost.inp")
    yes_edge = next(edge for edge in data["edges"] if edge["reference"] == "yes.inp")
    assert ghost_edge["line"] == 2
    assert ghost_edge["disabled"] is True
    assert yes_edge["line"] == 5
    assert yes_edge["disabled"] is False

    # the disabled reference is still resolved and reported as a missing target ...
    assert [Path(node["path"]).name for node in data["missing"]] == ["ghost.inp"]
    # ... but it is not active, does not fail the check and is not reachable
    assert graph.active_missing_nodes == []
    assert graph.check_status == 0
    assert set(data["reachable"]) == {str(entry.resolve()), str((tmp_path / "yes.inp").resolve())}


def test_disabled_branch_with_unevaluable_reference(tmp_path):
    entry = write(tmp_path / "entry.inp", "@IF 0\n@INCLUDE ${undefined}/frag.inp\n@ENDIF\n")

    graph = scan_dependencies(entry)
    edge = graph.edges[0]
    assert edge.disabled is True
    assert edge.target is None
    assert edge.reference == "${undefined}/frag.inp"
    assert graph.check_status == 0
    assert graph.to_dict()["edges"][0]["target"] is None


def test_set_in_disabled_branch_has_no_effect(tmp_path):
    entry = write(tmp_path / "entry.inp", "@IF 0\n@SET frag ghost\n@ENDIF\n@INCLUDE ${frag}.inp\n")

    with pytest.raises(DependencyScanError, match="undefined variable 'frag'"):
        scan_dependencies(entry)


def test_enabled_branch_with_unevaluable_reference_errors(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE ${undefined}/frag.inp\n")

    with pytest.raises(DependencyScanError) as excinfo:
        scan_dependencies(entry)
    assert excinfo.value.linenr == 1
    assert excinfo.value.filename == str(entry.resolve())


def test_preset_variable_flips_condition(tmp_path):
    write(tmp_path / "opt.inp", "!\n")
    entry = write(tmp_path / "entry.inp", "@IF ${DO_OPT}\n@INCLUDE opt.inp\n@ENDIF\n")

    disabled = scan_dependencies(entry, initial_variable_values={"DO_OPT": 0})
    assert disabled.edges[0].disabled is True
    assert disabled.check_status == 0

    enabled = scan_dependencies(entry, initial_variable_values={"DO_OPT": 1})
    assert enabled.edges[0].disabled is False
    assert enabled.check_status == 0
    assert str((tmp_path / "opt.inp").resolve()) in enabled.reachable


def test_variable_generated_filenames(tmp_path):
    sub = tmp_path / "sub"
    write(sub / "a.inp", "!\n")
    write(tmp_path / "zz.inp", "!\n")
    entry = write(
        tmp_path / "entry.inp",
        "@SET subname sub\n@INCLUDE ${subname}/a.inp\n@INCLUDE $zz\n",
    )

    graph = scan_dependencies(entry, initial_variable_values={"zz": "zz.inp"})
    data = graph.to_dict()
    refs = sorted(edge["reference"] for edge in data["edges"])
    assert refs == ["$zz", "${subname}/a.inp"]
    assert str((sub / "a.inp").resolve()) in graph.reachable
    assert str((tmp_path / "zz.inp").resolve()) in graph.reachable
    assert graph.check_status == 0


def test_missing_target_report_and_status(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE first.inp\n@INCLUDE second.inp\n")

    graph = scan_dependencies(entry)
    data = graph.to_dict()

    assert sorted(node["path"] for node in data["missing"]) == sorted([str(tmp_path / "first.inp"), str(tmp_path / "second.inp")])
    first = next(node for node in data["missing"] if node["path"].endswith("first.inp"))
    assert first["references"] == [{"source": str(entry.resolve()), "line": 1, "type": "include", "disabled": False}]
    assert graph.check_status == EXIT_MISSING


def test_dependency_cycle(tmp_path):
    write(tmp_path / "b.inp", "@INCLUDE a.inp\n")
    entry = write(tmp_path / "a.inp", "@INCLUDE b.inp\n")

    graph = scan_dependencies(entry)
    assert graph.to_dict()["cycles"] == [[str(entry.resolve()), str((tmp_path / "b.inp").resolve()), str(entry.resolve())]]
    assert graph.check_status == EXIT_CYCLE


def test_self_cycle(tmp_path):
    entry = write(tmp_path / "a.inp", "@INCLUDE a.inp\n")

    graph = scan_dependencies(entry)
    assert graph.to_dict()["cycles"] == [[str(entry.resolve()), str(entry.resolve())]]
    assert graph.check_status == EXIT_CYCLE


def test_missing_and_cycle_status(tmp_path):
    write(tmp_path / "b.inp", "@INCLUDE a.inp\n")
    entry = write(tmp_path / "a.inp", "@INCLUDE b.inp\n@INCLUDE ghost.inp\n")

    graph = scan_dependencies(entry)
    assert graph.check_status == EXIT_MISSING_AND_CYCLE


def test_cycle_reported_once_on_repeated_reference(tmp_path):
    write(tmp_path / "b.inp", "@INCLUDE a.inp\n")
    entry = write(tmp_path / "a.inp", "@INCLUDE b.inp\n@INCLUDE b.inp\n")

    graph = scan_dependencies(entry)
    assert len(graph.cycles) == 1


def test_stdin_stream():
    content = "&GLOBAL\n&END GLOBAL\n"
    graph = scan_dependencies(io.StringIO(content))

    assert graph.entry == "<stdin>"
    assert graph.to_dict()["reachable"] == ["<stdin>"]

    # a missing absolute target is reported in the graph rather than raised ...
    missing_graph = DependencyScanner().scan_stream(io.StringIO("@INCLUDE /nonexistent/path/x.inp\n"))
    assert missing_graph.check_status == EXIT_MISSING

    # ... while malformed directives still produce a scan error
    with pytest.raises(DependencyScanError, match="unknown preprocessor directive"):
        DependencyScanner().scan_stream(io.StringIO("@FOO bar\n"))


def test_stdin_with_search_root(tmp_path):
    write(tmp_path / "frag.inp", "!\n")
    stream = io.StringIO("@INCLUDE frag.inp\n")

    scanner = DependencyScanner(search_dirs=[tmp_path])
    graph = scanner.scan_stream(stream)

    assert graph.entry == "<stdin>"
    edge = graph.edges[0]
    assert edge.source == "<stdin>"
    assert edge.target == str((tmp_path / "frag.inp").resolve())
    assert set(graph.to_dict()["reachable"]) == {"<stdin>", str((tmp_path / "frag.inp").resolve())}


def test_repeated_execution_stability(tmp_path):
    write(tmp_path / "b.inp", "!\n")
    write(tmp_path / "a.inp", "@INCLUDE b.inp\n")
    entry = write(tmp_path / "entry.inp", "@INCLUDE a.inp\n@INCLUDE a.inp\n@INCLUDE missing.inp\n")

    scanner = DependencyScanner()
    first = scanner.scan_path(entry).to_json()
    second = scanner.scan_path(entry).to_json()
    third = DependencyScanner().scan_path(entry).to_json()

    assert first == second == third
    parsed = json.loads(first)
    # stable, sorted collections
    assert parsed["nodes"] == sorted(parsed["nodes"], key=lambda n: n["id"])
    assert [edge["line"] for edge in parsed["edges"] if edge["source"] == str(entry.resolve())] == [1, 2, 3]
    # re-serializing the parsed document gives the identical canonical document
    assert json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n" == first


def test_conditional_state_spans_includes(tmp_path):
    # the conditional state is global during the (depth-first) expansion, identical to
    # CP2KPreprocessor: an @ENDIF may sit inside an included file
    write(tmp_path / "child.inp", "@ENDIF\n")
    entry_open = write(tmp_path / "open.inp", "@IF 1\n@INCLUDE child.inp\n&TAIL\n")

    graph = scan_dependencies(entry_open)
    assert graph.check_status == 0
    assert set(graph.to_dict()["reachable"]) == {str(entry_open.resolve()), str((tmp_path / "child.inp").resolve())}

    # ... but an include inside a disabled branch is not expanded at all, so an @ENDIF
    # living in that file is not seen either and the block remains unclosed
    entry_closed = write(tmp_path / "closed.inp", "@IF 0\n@INCLUDE child.inp\n&TAIL\n")
    with pytest.raises(DependencyScanError, match="conditional block not closed"):
        scan_dependencies(entry_closed)


def test_disabled_include_not_expanded(tmp_path):
    write(tmp_path / "grandchild.inp", "!\n")
    write(tmp_path / "child.inp", "@INCLUDE grandchild.inp\n")
    entry = write(tmp_path / "entry.inp", "@IF 0\n@INCLUDE child.inp\n@ENDIF\n")

    graph = scan_dependencies(entry)
    names = {Path(node.id).name for node in graph.nodes.values()}
    assert names == {"entry.inp", "child.inp"}  # grandchild is not part of the graph
    assert graph.edges[0].disabled is True
    assert graph.to_dict()["reachable"] == [str(entry.resolve())]


def test_shipped_test02_input():
    graph = scan_dependencies(INPUTS_DIR / "test02.inp")
    data = graph.to_dict()
    assert any(node["id"].endswith("test02.fragment.inp") for node in data["nodes"])
    assert graph.check_status == 0


def test_quoted_include(tmp_path):
    write(tmp_path / "quoted frag.inp", "!\n")
    entry = write(tmp_path / "entry.inp", '@INCLUDE "quoted frag.inp"\n')

    graph = scan_dependencies(entry)
    assert graph.check_status == 0
    assert graph.edges[0].reference == '"quoted frag.inp"'
    assert graph.edges[0].target == str((tmp_path / "quoted frag.inp").resolve())


def test_preprocessor_error_conditions(tmp_path):
    with pytest.raises(DependencyScanError, match="found @ENDIF without a previous @IF"):
        scan_dependencies(write(tmp_path / "a.inp", "@ENDIF\n"))

    with pytest.raises(DependencyScanError, match="conditional block not closed"):
        scan_dependencies(write(tmp_path / "b.inp", "@IF 1\n"))

    with pytest.raises(DependencyScanError, match="nested @IF"):
        scan_dependencies(write(tmp_path / "c.inp", "@IF 1\n@IF 1\n@ENDIF\n@ENDIF\n"))

    with pytest.raises(DependencyScanError, match="unknown preprocessor directive"):
        scan_dependencies(write(tmp_path / "d.inp", "@FOOBAR something\n"))

    with pytest.raises(DependencyScanError, match="requires exactly one argument"):
        scan_dependencies(write(tmp_path / "e.inp", "@INCLUDE\n"))


def test_missing_entry_file(tmp_path):
    with pytest.raises(DependencyScanError, match="entry file"):
        scan_dependencies(tmp_path / "does-not-exist.inp")


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def run_cli(args, **kwargs):
    return CliRunner().invoke(cp2k_deps, args, catch_exceptions=False, **kwargs)


def test_cli_json_default(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE frag.inp\n")
    write(tmp_path / "frag.inp", "!\n")

    result = run_cli([str(entry)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["entry"] == str(entry.resolve())
    assert {Path(node["id"]).name for node in data["nodes"]} == {"entry.inp", "frag.inp"}


def test_cli_check_clean_exit_zero(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE frag.inp\n")
    write(tmp_path / "frag.inp", "!\n")

    result = run_cli(["--check", str(entry)])
    assert result.exit_code == 0


def test_cli_check_missing_exit_2(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE ghost.inp\n")

    result = run_cli(["--check", str(entry)])
    assert result.exit_code == EXIT_MISSING
    assert json.loads(result.output)["missing"][0]["path"].endswith("ghost.inp")

    # without --check a missing target does not result in a non-zero status
    result = run_cli([str(entry)])
    assert result.exit_code == 0


def test_cli_check_cycle_exit_3_and_both_4(tmp_path):
    write(tmp_path / "b.inp", "@INCLUDE a.inp\n")
    entry = write(tmp_path / "a.inp", "@INCLUDE b.inp\n")

    result = run_cli(["--check", str(entry)])
    assert result.exit_code == EXIT_CYCLE

    write(entry, "@INCLUDE b.inp\n@INCLUDE ghost.inp\n")
    result = run_cli(["--check", str(entry)])
    assert result.exit_code == EXIT_MISSING_AND_CYCLE


def test_cli_text_report(tmp_path):
    entry = write(tmp_path / "entry.inp", "@IF 0\n@INCLUDE ghost.inp\n@ENDIF\n@INCLUDE ghost.inp\n")

    result = run_cli(["--check", "-f", "text", str(entry)])
    assert result.exit_code == EXIT_MISSING
    assert "Reachable files" in result.output
    assert "Missing files (1)" in result.output
    assert "ghost.inp" in result.output
    assert "referenced via @INCLUDE at" in result.output
    assert "disabled @IF branch" in result.output
    assert "Dependency cycles (0)" in result.output


def test_cli_scan_error_exit_1(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE ${undef}.inp\n")

    result = run_cli(["--check", str(entry)])
    assert result.exit_code == 1
    assert "cp2k-deps: error" in result.output
    assert "undefined variable" in result.output
    assert "line 1" in result.output


def test_cli_nonexistent_entry_exit_1(tmp_path):
    result = run_cli([str(tmp_path / "nope.inp")])
    assert result.exit_code == 1
    assert "entry file" in result.output


def test_cli_search_dirs_and_preset_vars(tmp_path):
    root = tmp_path / "root"
    write(root / "picked.inp", "!\n")
    entry = write(tmp_path / "entry.inp", "@IF ${ON}\n@INCLUDE picked.inp\n@ENDIF\n")

    assert run_cli(["-I", str(root), "-E", "ON=0", "--check", str(entry)]).exit_code == 0

    result = run_cli(["-I", str(root), "-E", "ON=1", "--check", str(entry)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert any(node["id"].endswith("picked.inp") for node in data["nodes"])
    assert not next(edge for edge in data["edges"] if edge["reference"] == "picked.inp")["disabled"]


def test_cli_stdin(tmp_path):
    write(tmp_path / "frag.inp", "!\n")
    result = run_cli(["-I", str(tmp_path)], input="@INCLUDE frag.inp\n")

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["entry"] == "<stdin>"
    assert data["edges"][0]["source"] == "<stdin>"
    assert data["edges"][0]["target"].endswith("frag.inp")


def test_cli_stable_json_on_repeated_runs(tmp_path):
    entry = write(tmp_path / "entry.inp", "@INCLUDE b.inp\n@INCLUDE a.inp\n")
    write(tmp_path / "a.inp", "!\n")
    write(tmp_path / "b.inp", "!\n")

    first = run_cli([str(entry)]).output
    second = run_cli([str(entry)]).output
    assert first == second
