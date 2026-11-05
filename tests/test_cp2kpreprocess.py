import io
import json
from pathlib import Path

import pytest

from cp2k_input_tools.parser_errors import PreprocessorError
from cp2k_input_tools.preprocessor import (
    OutputPathConflictError,
    atomic_write_text,
    ensure_no_output_conflict,
    preprocess,
    source_map_to_json,
)

from . import TEST_DIR

INPUTS_DIR = TEST_DIR / "inputs"


# --------------------------------------------------------------------------------------------
# library API
# --------------------------------------------------------------------------------------------


def test_preprocess_simple_buffer():
    result = preprocess(io.StringIO("&GLOBAL\n  RUN_TYPE ENERGY\n&END GLOBAL\n"))

    assert result.lines == ["&GLOBAL", "RUN_TYPE ENERGY", "&END GLOBAL"]
    assert result.text == "&GLOBAL\nRUN_TYPE ENERGY\n&END GLOBAL\n"
    assert len(result.source_map) == 3


def test_preprocess_drops_comments_and_empty_lines():
    result = preprocess(io.StringIO("! a comment\n\n   # another comment\n&END\n"))

    assert result.lines == ["&END"]
    assert result.source_map[0].location.line_start == 4


def test_preprocess_variable_substitution_and_mapping():
    result = preprocess(io.StringIO("@SET X foo\nA $X B ${X}\n"))

    assert result.lines == ["A foo B foo"]
    entry = result.source_map[0]
    assert entry.output_line == 1
    assert entry.location.filename is None  # buffers have no file name
    assert entry.location.line_start == 2
    assert entry.location.line_end == 2


def test_preprocess_variable_preset_priority():
    # values preset by the caller act as defaults: an @SET in the input overrides them,
    # exactly as in the existing preprocessor / CP2K
    result = preprocess(io.StringIO("@SET X from_file\n$X\n"), initial_variable_values={"X": "from_cli"})
    assert result.lines == ["from_file"]

    # without an @SET the preset is used
    result = preprocess(io.StringIO("$X\n"), initial_variable_values={"X": "from_cli"})
    assert result.lines == ["from_cli"]


def test_preprocess_conditional_branches():
    stream = "@IF $SHOW\nVISIBLE\n@ENDIF\nALWAYS\n"

    assert preprocess(io.StringIO(stream), initial_variable_values={"SHOW": "1"}).lines == ["VISIBLE", "ALWAYS"]
    assert preprocess(io.StringIO(stream), initial_variable_values={"SHOW": "0"}).lines == ["ALWAYS"]

    # equality/inequality conditions
    stream_eq = "@IF $MODE == fast\nFAST\n@ENDIF\n@IF $MODE /= slow\nNOT_SLOW\n@ENDIF\nDONE\n"
    assert preprocess(io.StringIO(stream_eq), initial_variable_values={"MODE": "fast"}).lines == [
        "FAST",
        "NOT_SLOW",
        "DONE",
    ]
    assert preprocess(io.StringIO(stream_eq), initial_variable_values={"MODE": "slow"}).lines == ["DONE"]


def test_preprocess_include_chain():
    with (INPUTS_DIR / "test02.inp").open() as fhandle:
        result = preprocess(fhandle, base_dir=[INPUTS_DIR])

    # the @set variable in the fragment was expanded and comments/directives are gone
    assert "&FORCE_eval" in result.lines
    assert all(not line.startswith(("@", "#")) for line in result.lines)

    fragment_entries = [e for e in result.source_map if e.location.filename.endswith("test02.fragment.inp")]
    assert fragment_entries  # lines actually came from the included file

    entry = fragment_entries[0]
    assert Path(entry.location.filename).is_absolute()
    assert len(entry.include_chain) == 1
    site = entry.include_chain[0]
    assert site.filename.endswith("test02.inp")
    assert Path(site.filename).is_absolute()
    assert site.line_start == 6
    assert site.line_end == 6
    assert site.directive == '@include "test02.fragment.inp"'
    assert site.colnr == 0
    assert site.colnr_end == len(site.directive)


def test_preprocess_nested_includes(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    (dir2 / "c.inc").write_text("FROM_C\n")
    (dir1 / "b.inc").write_text("FROM_B\n@INCLUDE 'c.inc'\n")
    root = tmp_path / "root.inp"
    root.write_text("ROOT\n@INCLUDE 'b.inc'\nTAIL\n")

    with root.open() as fhandle:
        result = preprocess(fhandle, base_dir=[dir1, dir2, root.parent])

    assert result.lines == ["ROOT", "FROM_B", "FROM_C", "TAIL"]

    c_entry = next(e for e in result.source_map if e.location.filename.endswith("c.inc"))
    assert [s.filename for s in c_entry.include_chain] == [
        str(root.resolve()),
        str((dir1 / "b.inc").resolve()),
    ]
    assert c_entry.include_chain[0].directive == "@INCLUDE 'b.inc'"
    assert c_entry.include_chain[1].directive == "@INCLUDE 'c.inc'"

    b_entry = next(e for e in result.source_map if e.location.filename.endswith("b.inc"))
    assert len(b_entry.include_chain) == 1

    root_entries = [e for e in result.source_map if e.location.filename == str(root.resolve())]
    assert root_entries and all(not e.include_chain for e in root_entries)


def test_preprocess_indented_include_origin_columns(tmp_path):
    (tmp_path / "inc.inc").write_text("INCLUDED\n")
    root = tmp_path / "root.inp"
    root.write_text("   @INCLUDE inc.inc\n")

    with root.open() as fhandle:
        result = preprocess(fhandle, base_dir=[tmp_path])

    site = result.source_map[0].include_chain[0]
    assert site.colnr == 3  # leading whitespace stripped by the line iterator is accounted for
    assert site.colnr_end == 3 + len("@INCLUDE inc.inc")
    assert site.directive == "@INCLUDE inc.inc"


def test_preprocess_multiple_search_dirs(tmp_path):
    root_dir = tmp_path / "root"
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    root_dir.mkdir()
    dir1.mkdir()
    dir2.mkdir()

    (dir2 / "only_in_second.inc").write_text("FROM_SECOND_ROOT\n")
    root = root_dir / "root.inp"
    root.write_text("@INCLUDE 'only_in_second.inc'\n")

    with root.open() as fhandle:
        result = preprocess(fhandle, base_dir=[dir1, dir2])

    assert result.lines == ["FROM_SECOND_ROOT"]

    # the first root alone must not find it
    with root.open() as fhandle:
        with pytest.raises(PreprocessorError, match="could not be opened"):
            preprocess(fhandle, base_dir=[dir1])


def test_preprocess_xctype():
    with (INPUTS_DIR / "He_PBE.inp").open() as fhandle:
        result = preprocess(fhandle, base_dir=[INPUTS_DIR])

    assert any("XC_FUNCTIONAL" in line for line in result.lines)

    xc_entries = [e for e in result.source_map if e.location.filename.endswith("xc_section/PBE.sec")]
    assert xc_entries
    assert xc_entries[0].include_chain[0].directive == "@XCTYPE PBE"
    assert xc_entries[0].include_chain[0].filename.endswith("He_PBE.inp")


def test_preprocess_xctype_not_found():
    with (INPUTS_DIR / "He_PBE.inp").open() as fhandle:
        with pytest.raises(PreprocessorError, match="could not be opened"):
            preprocess(fhandle, base_dir=".")


def test_preprocess_missing_include_is_hard_error():
    with pytest.raises(PreprocessorError, match="could not be opened"):
        preprocess(io.StringIO("@INCLUDE 'definitely-not-there.inc'\n"))


def test_preprocess_unclosed_conditional():
    with pytest.raises(PreprocessorError, match="conditional block not closed at end of file"):
        preprocess(io.StringIO("@IF 1\nA\n"))


def test_preprocess_endif_without_if():
    with pytest.raises(PreprocessorError, match="found @ENDIF without a previous @IF"):
        preprocess(io.StringIO("@ENDIF\n"))


def test_preprocess_line_continuation_columns():
    result = preprocess(io.StringIO("  A B \\\n    C\n"))

    assert result.lines == ["A B C"]
    location = result.source_map[0].location
    assert location.line_start == 1
    assert location.line_end == 2
    assert location.colnrs == (2, 4)  # leading whitespace stripped from each physical line
    assert location.starts == (0, 4)  # offset of each physical segment in the output line


def test_source_map_covers_every_output_line(tmp_path):
    root = tmp_path / "simple.inp"
    root.write_text("&GLOBAL\n  PRINT_LEVEL MEDIUM\n&END GLOBAL\n")

    with root.open() as fhandle:
        result = preprocess(fhandle, base_dir=[root.parent])

    assert [e.output_line for e in result.source_map] == list(range(1, len(result.lines) + 1))

    raw_lines = root.read_text().splitlines()
    # every single output line can be mapped back and matches the original source line
    for idx, line in enumerate(result.lines):
        entry = result.source_map[idx]
        assert entry.location.filename == str(root.resolve())
        assert entry.location.line_start == entry.location.line_end == entry.output_line
        assert raw_lines[entry.location.line_start - 1].strip() == line
        assert entry.location.colnrs  # arbitrary line: column info present


def test_source_map_json_roundtrip():
    result = preprocess(io.StringIO("@SET X foo\nA $X\n"))

    document = json.loads(source_map_to_json(result.source_map))

    assert document["format"] == "cp2k-input-tools/source-map/v1"
    assert len(document["lines"]) == 1
    entry = document["lines"][0]
    assert entry["output_line"] == 1
    assert entry["file"] is None
    assert entry["line_start"] == 2
    assert entry["line_end"] == 2
    assert isinstance(entry["colnrs"], list)
    assert isinstance(entry["starts"], list)
    assert entry["include_chain"] == []


def test_atomic_write_text_replaces_existing(tmp_path):
    destination = tmp_path / "sub" / "out.txt"
    atomic_write_text(destination, "first\n")
    assert destination.read_text() == "first\n"

    atomic_write_text(destination, "second\n")
    assert destination.read_text() == "second\n"
    assert list(tmp_path.rglob("*.tmp")) == []  # no temporary files left behind


def test_output_conflict_same_path_refused(tmp_path):
    inp = tmp_path / "in.inp"
    inp.write_text("&END\n")

    with pytest.raises(OutputPathConflictError, match="refusing to overwrite the input file"):
        ensure_no_output_conflict(inp, output_path=inp)

    with pytest.raises(OutputPathConflictError, match="source map file"):
        ensure_no_output_conflict(inp, source_map_path=inp)

    # different paths are fine
    ensure_no_output_conflict(inp, output_path=tmp_path / "out.txt", source_map_path=tmp_path / "map.json")

    # no input path (stdin): nothing to check
    ensure_no_output_conflict(None, output_path=inp)


# --------------------------------------------------------------------------------------------
# command line interface
# --------------------------------------------------------------------------------------------


def test_cp2kpreprocess_stdout(script_runner):
    ret = script_runner.run(["cp2kpreprocess", str(INPUTS_DIR / "test02.inp")])

    assert ret.success
    assert ret.stderr == ""
    assert "&GLOBAL" in ret.stdout
    assert "&FORCE_eval" in ret.stdout
    assert all(not line.startswith(("@", "#")) for line in ret.stdout.splitlines())


def test_cp2kpreprocess_default_search_dir_is_input_dir(script_runner, tmp_path):
    # no -I: includes are resolved relative to the directory containing the input file
    ret = script_runner.run(["cp2kpreprocess", str(INPUTS_DIR / "test02.inp")], cwd=str(tmp_path))

    assert ret.success
    assert "&FORCE_eval" in ret.stdout


def test_cp2kpreprocess_stdin(script_runner):
    with (INPUTS_DIR / "test02.inp").open() as stdin:
        ret = script_runner.run(["cp2kpreprocess"], stdin=stdin, cwd=str(INPUTS_DIR))

    assert ret.success
    assert "&FORCE_eval" in ret.stdout


def test_cp2kpreprocess_stdin_conditions_and_vars(script_runner):
    stream = "@SET X foo\n$X\n@IF $SHOW\nVISIBLE\n@ENDIF\nDONE\n"

    ret = script_runner.run(["cp2kpreprocess", "-E", "SHOW=1"], stdin=io.StringIO(stream))
    assert ret.success
    assert ret.stdout.splitlines() == ["foo", "VISIBLE", "DONE"]

    ret = script_runner.run(["cp2kpreprocess", "-E", "SHOW=0"], stdin=io.StringIO(stream))
    assert ret.success
    assert ret.stdout.splitlines() == ["foo", "DONE"]


def test_cp2kpreprocess_output_file_atomic(script_runner, tmp_path):
    inp = tmp_path / "in.inp"
    inp.write_text("&GLOBAL\n&END GLOBAL\n")
    out = tmp_path / "out.txt"

    ret = script_runner.run(["cp2kpreprocess", str(inp), "-o", str(out)])

    assert ret.success
    assert ret.stdout == ""
    assert out.read_text() == "&GLOBAL\n&END GLOBAL\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cp2kpreprocess_source_map(script_runner, tmp_path):
    out = tmp_path / "preprocessed.inp"
    source_map = tmp_path / "map.json"

    ret = script_runner.run(
        [
            "cp2kpreprocess",
            str(INPUTS_DIR / "test02.inp"),
            "-o",
            str(out),
            "--source-map",
            str(source_map),
        ]
    )

    assert ret.success

    document = json.loads(source_map.read_text())
    assert document["format"] == "cp2k-input-tools/source-map/v1"

    output_lines = out.read_text().splitlines()
    entries = document["lines"]
    assert len(entries) == len(output_lines)
    assert [e["output_line"] for e in entries] == list(range(1, len(output_lines) + 1))
    assert all(e["file"] is None or Path(e["file"]).is_absolute() for e in entries)

    fragment_entry = next(e for e in entries if e["file"].endswith("test02.fragment.inp"))
    assert fragment_entry["line_start"] >= 1
    assert fragment_entry["line_end"] >= fragment_entry["line_start"]
    assert len(fragment_entry["include_chain"]) == 1
    site = fragment_entry["include_chain"][0]
    assert site["file"].endswith("test02.inp")
    assert site["line_start"] == site["line_end"] == 6
    assert site["directive"].startswith("@include")
    assert 0 <= site["colnr"] < site["colnr_end"]


def test_cp2kpreprocess_source_map_with_stdout(script_runner, tmp_path):
    source_map = tmp_path / "map.json"

    ret = script_runner.run(["cp2kpreprocess", str(INPUTS_DIR / "test02.inp"), "--source-map", str(source_map)])

    assert ret.success
    assert "&GLOBAL" in ret.stdout
    document = json.loads(source_map.read_text())
    assert document["lines"]


def test_cp2kpreprocess_nested_includes_and_chains(script_runner, tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir2 / "c.inc").write_text("FROM_C\n")
    (dir1 / "b.inc").write_text("FROM_B\n@INCLUDE 'c.inc'\n")
    root = tmp_path / "root.inp"
    root.write_text("ROOT\n@INCLUDE 'b.inc'\nTAIL\n")

    source_map = tmp_path / "map.json"
    ret = script_runner.run(
        [
            "cp2kpreprocess",
            str(root),
            "-I",
            str(dir1),
            "-I",
            str(dir2),
            "--source-map",
            str(source_map),
        ]
    )

    assert ret.success, ret.stderr
    assert ret.stdout.splitlines() == ["ROOT", "FROM_B", "FROM_C", "TAIL"]

    entries = json.loads(source_map.read_text())["lines"]
    c_entry = next(e for e in entries if e["file"].endswith("c.inc"))
    assert [s["file"] for s in c_entry["include_chain"]] == [str(root.resolve()), str((dir1 / "b.inc").resolve())]


def test_cp2kpreprocess_multiple_search_dirs_second_root(script_runner, tmp_path):
    root_dir = tmp_path / "root"
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    root_dir.mkdir()
    dir1.mkdir()
    dir2.mkdir()
    (dir2 / "only_here.inc").write_text("FOUND_IN_SECOND_ROOT\n")
    root = root_dir / "root.inp"
    root.write_text("@INCLUDE 'only_here.inc'\n")

    ret = script_runner.run(["cp2kpreprocess", str(root), "-I", str(dir1), "-I", str(dir2)])
    assert ret.success
    assert "FOUND_IN_SECOND_ROOT" in ret.stdout

    # without the second search root the include cannot be opened
    ret = script_runner.run(["cp2kpreprocess", str(root), "-I", str(dir1)])
    assert not ret.success
    assert "could not be opened" in ret.stdout


def test_cp2kpreprocess_xctype(script_runner, tmp_path):
    source_map = tmp_path / "map.json"
    ret = script_runner.run(
        [
            "cp2kpreprocess",
            str(INPUTS_DIR / "He_PBE.inp"),
            "-I",
            str(INPUTS_DIR),
            "--source-map",
            str(source_map),
        ]
    )

    assert ret.success, ret.stderr
    assert "&XC_FUNCTIONAL PBE" in ret.stdout

    entries = json.loads(source_map.read_text())["lines"]
    xc_entry = next(e for e in entries if e["file"].endswith("xc_section/PBE.sec"))
    assert xc_entry["include_chain"][0]["directive"] == "@XCTYPE PBE"


def test_cp2kpreprocess_missing_include(script_runner, tmp_path):
    inp = tmp_path / "in.inp"
    inp.write_text("@INCLUDE 'does-not-exist.inc'\n")

    ret = script_runner.run(["cp2kpreprocess", str(inp)])

    assert not ret.success
    assert "could not be opened" in ret.stdout
    assert "does-not-exist.inc" in ret.stdout


def test_cp2kpreprocess_undefined_variable_error(script_runner):
    ret = script_runner.run(["cp2kpreprocess"], stdin=io.StringIO("A $UNDEF\n"))

    assert not ret.success
    assert "undefined variable 'UNDEF'" in ret.stdout
    assert "line    1: A $UNDEF" in ret.stdout


def test_cp2kpreprocess_unclosed_conditional(script_runner):
    ret = script_runner.run(["cp2kpreprocess"], stdin=io.StringIO("@IF 1\nA\n"))

    assert not ret.success
    assert "conditional block not closed at end of file" in ret.stdout


def test_cp2kpreprocess_refuse_same_input_output(script_runner, tmp_path):
    inp = tmp_path / "in.inp"
    inp.write_text("&GLOBAL\n&END GLOBAL\n")
    original = inp.read_text()

    ret = script_runner.run(["cp2kpreprocess", str(inp), "-o", str(inp)])

    assert not ret.success
    assert "refusing to overwrite the input file" in ret.stderr
    assert inp.read_text() == original  # the input was never touched

    ret = script_runner.run(["cp2kpreprocess", str(inp), "--source-map", str(inp)])
    assert not ret.success
    assert "refusing to overwrite the input file" in ret.stderr
    assert inp.read_text() == original


def test_cp2kpreprocess_bad_set_argument(script_runner, tmp_path):
    inp = tmp_path / "in.inp"
    inp.write_text("&END\n")

    ret = script_runner.run(["cp2kpreprocess", "-E", "missing-equal-sign", str(inp)])

    assert not ret.success
    assert "Invalid value for '-E' / '--set'" in ret.stderr
