import io
import textwrap
from copy import deepcopy

import pytest

from cp2k_input_tools.merger import (
    CP2KInputMerger,
    MergeConflictError,
    MergePathNotFoundError,
    MergePolicy,
    MergeValidationError,
    RepeatPolicy,
    merge_inputs,
)
from cp2k_input_tools.parser import CP2KInputParser
from cp2k_input_tools.parser_errors import InvalidNameError

BASE_INPUT = """\
&GLOBAL
   PRINT_LEVEL MEDIUM
   PROJECT base-project
&END GLOBAL
&MOTION
   &CELL_OPT
      MAX_ITER 10
      SYMM_EXCLUDE_RANGE 1 2
      &LBFGS
         MAX_H_RANK 3
      &END LBFGS
   &END CELL_OPT
   &TMC
      &MOVE_TYPE proton
         SIZE 1.0
      &END MOVE_TYPE
      &MOVE_TYPE hmc
         PROB 0.5
      &END MOVE_TYPE
   &END TMC
&END MOTION
&FORCE_EVAL
   METHOD QUICKSTEP
&END FORCE_EVAL
"""

SCALAR_OVERRIDE = """\
&GLOBAL
   PRINT_LEVEL DEBUG
&END GLOBAL
"""

NESTED_OVERRIDE = """\
&MOTION
   &CELL_OPT
      KEEP_ANGLES .TRUE.
      &LBFGS
         TRUST_RADIUS 0.25
      &END LBFGS
   &END CELL_OPT
&END MOTION
"""

PARAMETERIZED_OVERRIDE = """\
&MOTION
   &TMC
      &MOVE_TYPE proton
         PROB 0.25
      &END MOVE_TYPE
      &MOVE_TYPE water
         SIZE 5.0
      &END MOVE_TYPE
   &END TMC
&END MOTION
"""

NON_PARAMETERIZED_OVERRIDE = """\
&FORCE_EVAL
   METHOD FIST
&END FORCE_EVAL
"""

ARRAY_OVERRIDE = """\
&MOTION
   &CELL_OPT
      SYMM_EXCLUDE_RANGE 3 4
   &END CELL_OPT
&END MOTION
"""

NEW_KEYWORD_OVERRIDE = """\
&GLOBAL
   RUN_TYPE ENERGY_FORCE
&END GLOBAL
"""

UNKNOWN_KEYWORD_OVERRIDE = """\
&GLOBAL
   THIS_KEYWORD_DOES_NOT_EXIST 1
&END GLOBAL
"""

CONFLICT_BASE = """\
&MOTION
   &TMC
      &MOVE_TYPE proton
         SIZE 1.0
      &END MOVE_TYPE
      &MOVE_TYPE proton
         SIZE 2.0
      &END MOVE_TYPE
   &END TMC
&END MOTION
"""

CONFLICT_OVERRIDE = """\
&MOTION
   &TMC
      &MOVE_TYPE proton
         PROB 0.1
      &END MOVE_TYPE
   &END TMC
&END MOTION
"""


def _parse(text):
    parser = CP2KInputParser(key_trafo=str.lower)
    fhandle = io.StringIO(textwrap.dedent(text))
    fhandle.name = "<test-input>"
    return parser.parse(fhandle)


def _merge(base_text, override_text, **kwargs):
    merger = CP2KInputMerger(**kwargs)
    tree = merger.merge_trees(_parse(base_text), _parse(override_text))
    lines = merger.verify(tree)
    return tree, lines


def _reparse(lines):
    parser = CP2KInputParser(key_trafo=str.lower)
    fhandle = io.StringIO("\n".join(lines))
    fhandle.name = "<merged>"
    return parser.parse(fhandle)


# --------------------------------------------------------------------- library


def test_scalar_override():
    tree, lines = _merge(BASE_INPUT, SCALAR_OVERRIDE)

    assert tree["+global"]["print_level"] == "debug"
    # unrelated scalars are retained (PROJECT is an alias of the canonical PROJECT_NAME keyword)
    assert tree["+global"]["project_name"] == "base-project"

    rendered = "\n".join(lines)
    assert "PRINT_LEVEL DEBUG" in rendered
    assert "PROJECT_NAME base-project" in rendered


def test_nested_sections_merge_recursively():
    tree, lines = _merge(BASE_INPUT, NESTED_OVERRIDE)

    cell_opt = tree["+motion"]["+cell_opt"]
    # scalar in the base section retained, scalar from the override added
    assert cell_opt["max_iter"] == 10
    assert cell_opt["keep_angles"] is True

    # deeply nested non-repeating section merged instead of being replaced
    lbfgs = cell_opt["+lbfgs"]
    assert lbfgs["max_h_rank"] == 3
    assert lbfgs["trust_radius"] == pytest.approx(0.25)

    assert any("&LBFGS" in line for line in lines)
    assert "MAX_H_RANK 3" in "\n".join(lines)
    assert "TRUST_RADIUS 0.25" in "\n".join(lines)


def test_new_non_repeating_section_is_added():
    override = """\
&MOTION
   &GEO_OPT
      MAX_ITER 42
   &END GEO_OPT
&END MOTION
"""
    tree, lines = _merge(BASE_INPUT, override)

    assert tree["+motion"]["+geo_opt"]["max_iter"] == 42
    # sibling sections from the base are retained
    assert "+cell_opt" in tree["+motion"]
    assert any("&GEO_OPT" in line for line in lines)


def test_parameterized_repeated_sections_match_by_parameter():
    tree, lines = _merge(BASE_INPUT, PARAMETERIZED_OVERRIDE)

    move_types = tree["+motion"]["+tmc"]["+move_type"]
    by_param = {entry["_"]: entry for entry in move_types}

    # matched section recursively merged, base-only values retained
    assert by_param["proton"]["size"] == pytest.approx(1.0)
    assert by_param["proton"]["prob"] == pytest.approx(0.25)
    # unmatched base section left untouched
    assert by_param["hmc"]["prob"] == pytest.approx(0.5)
    # unmatched override section appended
    assert "water" in by_param
    assert by_param["water"]["size"] == pytest.approx(5.0)
    assert [entry["_"] for entry in move_types] == ["proton", "hmc", "water"]


def test_non_parameterized_repeated_section_appends_by_default():
    tree, lines = _merge(BASE_INPUT, NON_PARAMETERIZED_OVERRIDE)

    force_evals = tree["+force_eval"]
    assert len(force_evals) == 2
    assert force_evals[0]["method"] == "quickstep"
    assert force_evals[1]["method"] == "fist"

    opens = [line for line in lines if line.lstrip().startswith("&FORCE_EVAL")]
    assert len(opens) == 2


def test_non_parameterized_repeated_section_can_be_replaced():
    policy = MergePolicy(repeated_sections=RepeatPolicy.REPLACE)
    tree, _ = _merge(BASE_INPUT, NON_PARAMETERIZED_OVERRIDE, policy=policy)

    force_evals = tree["+force_eval"]
    assert len(force_evals) == 1
    assert force_evals[0]["method"] == "fist"


def test_repeated_keyword_array_appends_by_default():
    tree, lines = _merge(BASE_INPUT, ARRAY_OVERRIDE)

    values = tree["+motion"]["+cell_opt"]["symm_exclude_range"]
    assert values == [(1, 2), (3, 4)]
    assert sum("SYMM_EXCLUDE_RANGE" in line for line in lines) == 2


def test_repeated_keyword_array_can_be_replaced():
    policy = MergePolicy(repeated_keywords=RepeatPolicy.REPLACE)
    tree, lines = _merge(BASE_INPUT, ARRAY_OVERRIDE, policy=policy)

    assert tree["+motion"]["+cell_opt"]["symm_exclude_range"] == [(3, 4)]
    assert sum("SYMM_EXCLUDE_RANGE" in line for line in lines) == 1
    assert not any(line.rstrip().endswith("1 2") for line in lines)


def test_existing_only_allows_existing_paths():
    tree, _ = _merge(BASE_INPUT, SCALAR_OVERRIDE, only_existing_paths=True)
    assert tree["+global"]["print_level"] == "debug"


def test_existing_only_rejects_new_keyword_path():
    merger = CP2KInputMerger(only_existing_paths=True)
    with pytest.raises(MergePathNotFoundError):
        merger.merge_trees(_parse(BASE_INPUT), _parse(NEW_KEYWORD_OVERRIDE))


def test_existing_only_rejects_new_section_parameter():
    merger = CP2KInputMerger(only_existing_paths=True)
    with pytest.raises(MergePathNotFoundError):
        merger.merge_trees(_parse(BASE_INPUT), _parse(PARAMETERIZED_OVERRIDE))


def test_existing_only_rejects_new_section_path():
    merger = CP2KInputMerger(only_existing_paths=True)
    # KPOINTS is not part of the base tree
    override = "&FORCE_EVAL\n   &DFT\n      &KPOINTS\n         FULL_GRID .TRUE.\n      &END KPOINTS\n   &END DFT\n&END FORCE_EVAL\n"
    with pytest.raises(MergePathNotFoundError):
        merger.merge_trees(_parse(BASE_INPUT), _parse(override))


def test_unknown_keyword_fails_schema_validation():
    # the override must validate against the same XML schema before anything is merged
    with pytest.raises(InvalidNameError):
        _merge(BASE_INPUT, UNKNOWN_KEYWORD_OVERRIDE)


def test_unknown_section_fails_schema_validation():
    override = "&THIS_SECTION_DOES_NOT_EXIST\n&END THIS_SECTION_DOES_NOT_EXIST\n"
    from cp2k_input_tools.parser_errors import InvalidSectionError

    with pytest.raises(InvalidSectionError):
        _merge(BASE_INPUT, override)


def test_duplicate_section_parameters_are_a_conflict():
    merger = CP2KInputMerger()
    with pytest.raises(MergeConflictError):
        merger.merge_trees(_parse(CONFLICT_BASE), _parse(CONFLICT_OVERRIDE))


def test_duplicate_parameters_in_override_are_a_conflict():
    override = """\
&MOTION
   &TMC
      &MOVE_TYPE water
         SIZE 1.0
      &END MOVE_TYPE
      &MOVE_TYPE water
         SIZE 2.0
      &END MOVE_TYPE
   &END TMC
&END MOTION
"""
    merger = CP2KInputMerger()
    with pytest.raises(MergeConflictError):
        merger.merge_trees(_parse(BASE_INPUT), _parse(override))


def test_generator_validation_failure_is_detected():
    # trees not coming from the parser bypass input validation, the mandatory
    # generator pass must catch the resulting invalid CP2K data
    base_tree = _parse(BASE_INPUT)

    merger = CP2KInputMerger()
    merged = merger.merge_trees(base_tree, {"+motion": {"+cell_opt": {"keep_angles": "maybe"}}})
    with pytest.raises(MergeValidationError):
        merger.verify(merged)

    merged = merger.merge_trees(base_tree, {"+global": {"bogus_keyword": 1}})
    with pytest.raises(MergeValidationError):
        merger.verify(merged)


def test_merged_output_reparses_into_the_merged_tree():
    merger = CP2KInputMerger()
    tree = merger.merge_trees(_parse(BASE_INPUT), _parse(PARAMETERIZED_OVERRIDE))
    lines = merger.verify(tree)

    reparsed = _reparse(lines)

    # the generator renders repeated sections (here MOVE_TYPE) in stack order
    assert [entry["_"] for entry in reparsed["+motion"]["+tmc"]["+move_type"]] == ["water", "hmc", "proton"]

    # after undoing this deterministic ordering effect, the output is structurally identical
    import xml.etree.ElementTree as ET

    from cp2k_input_tools import DEFAULT_CP2K_INPUT_XML

    merger._restore_section_order(ET.parse(DEFAULT_CP2K_INPUT_XML).getroot(), reparsed)
    assert reparsed == tree


def test_merge_does_not_mutate_input_trees():
    base_tree = _parse(BASE_INPUT)
    override_tree = _parse(PARAMETERIZED_OVERRIDE)
    base_before = deepcopy(base_tree)
    override_before = deepcopy(override_tree)

    merger = CP2KInputMerger()
    merger.verify(merger.merge_trees(base_tree, override_tree))

    assert base_tree == base_before
    assert override_tree == override_before


def test_merge_inputs_from_handles():
    base_fh = io.StringIO(textwrap.dedent(BASE_INPUT))
    base_fh.name = "<base>"
    override_fh = io.StringIO(textwrap.dedent(SCALAR_OVERRIDE))
    override_fh.name = "<override>"

    tree, lines = merge_inputs(base_fh, override_fh)

    assert tree["+global"]["print_level"] == "debug"
    assert any("PRINT_LEVEL DEBUG" in line for line in lines)


# ------------------------------------------------------------------------- CLI


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


def test_cp2k_merge_cli_writes_validated_output(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", PARAMETERIZED_OVERRIDE)
    output = tmp_path / "merged.inp"

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(output)])

    assert ret.success, ret.stderr
    assert output.exists()

    reparsed = _reparse(output.read_text().splitlines())
    move_types = sorted(entry["_"] for entry in reparsed["+motion"]["+tmc"]["+move_type"])
    assert move_types == ["hmc", "proton", "water"]
    assert "PRINT_LEVEL MEDIUM" in output.read_text()


def test_cp2k_merge_cli_stdout(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", SCALAR_OVERRIDE)

    ret = script_runner.run(["cp2k-merge", str(base), str(override)])

    assert ret.success, ret.stderr
    assert "PRINT_LEVEL DEBUG" in ret.stdout
    assert "PROJECT_NAME base-project" in ret.stdout


def test_cp2k_merge_cli_append_and_replace_section_policies(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", NON_PARAMETERIZED_OVERRIDE)
    output_append = tmp_path / "append.inp"
    output_replace = tmp_path / "replace.inp"

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(output_append)])
    assert ret.success, ret.stderr
    assert output_append.read_text().count("&FORCE_EVAL") == 2  # opening lines of both sections

    ret = script_runner.run(["cp2k-merge", "--repeated-sections", "replace", str(base), str(override), "-o", str(output_replace)])
    assert ret.success, ret.stderr
    assert output_replace.read_text().count("&FORCE_EVAL") == 1
    assert "METHOD FIST" in output_replace.read_text()
    assert "METHOD QUICKSTEP" not in output_replace.read_text()


def test_cp2k_merge_cli_keyword_replace_policy(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", ARRAY_OVERRIDE)
    output = tmp_path / "merged.inp"

    ret = script_runner.run(["cp2k-merge", "--repeated-keywords", "replace", str(base), str(override), "-o", str(output)])
    assert ret.success, ret.stderr

    text = output.read_text()
    assert "SYMM_EXCLUDE_RANGE 3 4" in text
    assert "SYMM_EXCLUDE_RANGE 1 2" not in text


def test_cp2k_merge_cli_existing_only(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)

    allowed = _write(tmp_path, "allowed.inp", SCALAR_OVERRIDE)
    ret = script_runner.run(["cp2k-merge", "--existing-only", str(base), str(allowed), "-o", str(tmp_path / "ok.inp")])
    assert ret.success, ret.stderr

    denied = _write(tmp_path, "denied.inp", NEW_KEYWORD_OVERRIDE)
    output = tmp_path / "denied.out.inp"
    ret = script_runner.run(["cp2k-merge", "--existing-only", str(base), str(denied), "-o", str(output)])
    assert not ret.success
    assert "not present in the base input" in ret.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_cp2k_merge_cli_rejects_inputs_being_the_same_file(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    output = tmp_path / "merged.inp"

    ret = script_runner.run(["cp2k-merge", str(base), str(base), "-o", str(output)])
    assert not ret.success
    assert "same file" in ret.stderr
    assert not output.exists()


def test_cp2k_merge_cli_rejects_output_equal_to_inputs(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", SCALAR_OVERRIDE)

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(base)])
    assert not ret.success
    assert "output path must be different" in ret.stderr

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(override)])
    assert not ret.success
    assert "output path must be different" in ret.stderr


def test_cp2k_merge_cli_unknown_field_is_nonzero_and_writes_nothing(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", UNKNOWN_KEYWORD_OVERRIDE)
    output = tmp_path / "merged.inp"

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(output)])

    assert not ret.success
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_cp2k_merge_cli_conflict_is_nonzero_and_writes_nothing(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", CONFLICT_BASE)
    override = _write(tmp_path, "override.inp", CONFLICT_OVERRIDE)
    output = tmp_path / "merged.inp"

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(output)])

    assert not ret.success
    assert "multiple" in ret.stderr or "ambiguously" in ret.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_cp2k_merge_cli_replaces_output_atomically(script_runner, tmp_path):
    base = _write(tmp_path, "base.inp", BASE_INPUT)
    override = _write(tmp_path, "override.inp", SCALAR_OVERRIDE)
    output = _write(tmp_path, "merged.inp", "previous content\n")

    ret = script_runner.run(["cp2k-merge", str(base), str(override), "-o", str(output)])

    assert ret.success, ret.stderr
    assert "PRINT_LEVEL DEBUG" in output.read_text()
    assert "previous content" not in output.read_text()
    assert not list(tmp_path.glob(".*.tmp"))
