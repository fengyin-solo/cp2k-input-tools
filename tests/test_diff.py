import io
import json

import pytest

from cp2k_input_tools.diff import (
    OP_ADDED,
    OP_CHANGED,
    OP_REMOVED,
    CP2KDiffParseError,
    DiffOptions,
    diff_inputs,
    diff_paths,
    diff_trees,
)

from . import TEST_DIR

DIFF_DIR = TEST_DIR.joinpath("inputs/diff")
INC_DIR = DIFF_DIR.joinpath("includes")


def _diff_pair(left, right, options=None, **kwargs):
    with open(DIFF_DIR.joinpath(left)) as fleft, open(DIFF_DIR.joinpath(right)) as fright:
        return diff_inputs(fleft, fright, options, **kwargs)


def _entries_by_path(result):
    return {entry.path: entry for entry in result.entries}


# ----------------------------------------------------------------------------
# library API
# ----------------------------------------------------------------------------


def test_identical_inputs_ignore_cosmetic_differences():
    # comments, indentation, casing, quoting, numeric notation (400 vs 4.0e2,
    # 50.0 vs 50, 1.0E-10 vs 1.0d-10) and keyword ordering must be ignored
    result = _diff_pair("base.inp", "cosmetic.inp")

    assert result.identical
    assert result.entries == []


def test_scalar_changes():
    result = _diff_pair("base.inp", "scalar_changed.inp")

    assert not result.identical
    assert [e.op for e in result.entries] == [OP_CHANGED, OP_CHANGED]

    by_path = _entries_by_path(result)
    assert by_path["force_eval[0]/dft/mgrid/cutoff"].value_left == 400.0
    assert by_path["force_eval[0]/dft/mgrid/cutoff"].value_right == 500.0
    assert by_path["global/print_level"].value_left == "medium"
    assert by_path["global/print_level"].value_right == "high"


def test_section_parameter_change():
    result = _diff_pair("base.inp", "section_param_changed.inp")

    assert len(result.entries) == 1
    entry = result.entries[0]

    assert entry.op == OP_CHANGED
    assert entry.path == "force_eval[0]/dft/xc/xc_functional/_"
    assert entry.value_left == "PBE"
    assert entry.value_right == "BLYP"


def test_array_element_change():
    # only the second component of the A cell vector changes
    result = _diff_pair("base.inp", "array_element.inp")

    assert len(result.entries) == 1
    entry = result.entries[0]

    assert entry.op == OP_CHANGED
    assert entry.path == "force_eval[0]/subsys/cell/a[1]"
    assert entry.value_left == 0.0
    assert entry.value_right == 0.5


def test_added_and_removed_values_and_subtrees():
    result = _diff_pair("base.inp", "added_removed.inp")
    by_path = _entries_by_path(result)

    assert by_path["force_eval[0]/dft/mgrid/rel_cutoff"].op == OP_REMOVED
    assert by_path["force_eval[0]/dft/mgrid/rel_cutoff"].value_left == 50.0
    # whole removed section is reported leaf by leaf
    assert by_path["force_eval[0]/dft/qs/eps_default"].op == OP_REMOVED
    # added keyword
    assert by_path["force_eval[0]/dft/scf/eps_scf"].op == OP_ADDED
    assert by_path["force_eval[0]/dft/scf/eps_scf"].value_right == 1e-06
    # added repeated section including parameter and keywords
    assert by_path["force_eval[0]/subsys/kind[2]/_"].op == OP_ADDED
    assert by_path["force_eval[0]/subsys/kind[2]/_"].value_right == "C"
    assert by_path["force_eval[0]/subsys/kind[2]/element"].value_right == "C"

    assert len(result.added) == 4
    assert len(result.removed) == 2
    assert result.changed == []


def test_repeated_sections_are_indexed_and_order_significant_by_default():
    result = _diff_pair("base.inp", "kinds_reordered.inp")

    # the same two KIND sections swapped must be reported positionally
    assert not result.identical
    by_path = _entries_by_path(result)
    assert by_path["force_eval[0]/subsys/kind[0]/_"].value_left == "H"
    assert by_path["force_eval[0]/subsys/kind[0]/_"].value_right == "O"
    assert by_path["force_eval[0]/subsys/kind[1]/_"].value_left == "O"
    assert by_path["force_eval[0]/subsys/kind[1]/_"].value_right == "H"


def test_repeated_section_order_can_be_ignored():
    result = _diff_pair("base.inp", "kinds_reordered.inp", DiffOptions(ignore_section_order=True))
    assert result.identical

    # a swapped AND modified section must be matched up instead
    result = _diff_pair("base.inp", "kinds_reordered_changed.inp", DiffOptions(ignore_section_order=True))
    assert len(result.changed) == 1
    entry = result.changed[0]
    assert entry.path == "force_eval[0]/subsys/kind[0]/basis_set[0]"
    assert entry.value_left == "DZVP-MOLOPT"
    assert entry.value_right == "TZVP-MOLOPT"


def test_float_tolerance_absolute():
    options_exact = DiffOptions()
    options_loose = DiffOptions(abs_tol=1e-3)
    options_tight = DiffOptions(abs_tol=1e-6)

    assert not _diff_pair("base.inp", "tolerance_close.inp", options_exact).identical
    assert _diff_pair("base.inp", "tolerance_close.inp", options_loose).identical
    assert not _diff_pair("base.inp", "tolerance_close.inp", options_tight).identical
    # a difference larger than the tolerance is still reported
    assert not _diff_pair("base.inp", "tolerance_far.inp", options_loose).identical


def test_float_tolerance_relative():
    result_equal = diff_trees({"v": 400.0}, {"v": 400.0001}, DiffOptions(rel_tol=1e-6))
    assert result_equal.identical

    result_different = diff_trees({"v": 400.0}, {"v": 400.0001}, DiffOptions(rel_tol=1e-9))
    assert not result_different.identical


def test_integer_and_float_notation_normalize():
    left = io.StringIO("&GLOBAL\nPRINT_LEVEL MEDIUM\n&END\n" "&FORCE_EVAL\n&DFT\n&MGRID\nCUTOFF 400\n&END\n&END\n&END\n")
    right = io.StringIO("&GLOBAL\nPRINT_LEVEL medium\n&END\n" "&FORCE_EVAL\n&DFT\n&MGRID\nCUTOFF 4.0e2\n&END\n&END\n&END\n")

    result = diff_inputs(left, right)
    assert result.identical


def _kpoints_stream(value):
    return io.StringIO(f"&FORCE_EVAL\n&DFT\n&KPOINTS\nFULL_GRID {value}\n&END KPOINTS\n&END DFT\n&END FORCE_EVAL\n")


def test_logical_values_normalize():
    result = diff_inputs(_kpoints_stream(".FALSE."), _kpoints_stream("False"))
    assert result.identical

    result = diff_inputs(_kpoints_stream(".FALSE."), _kpoints_stream(".TRUE."))
    assert len(result.changed) == 1
    assert result.changed[0].path == "force_eval[0]/dft/kpoints/full_grid"
    assert result.changed[0].value_left is False
    assert result.changed[0].value_right is True


def test_string_case_is_significant():
    # project names are strings: their case matters (e.g. on case-sensitive filesystems)
    left = io.StringIO("&GLOBAL\nPROJECT Calc\n&END GLOBAL\n")
    right = io.StringIO("&GLOBAL\nPROJECT calc\n&END GLOBAL\n")

    result = diff_inputs(left, right)
    assert not result.identical
    assert result.changed[0].path == "global/project_name"


def test_include_expansion_is_transparent():
    result = diff_paths(
        INC_DIR.joinpath("main.inp"),
        DIFF_DIR.joinpath("included_inline.inp"),
        base_dir_left=INC_DIR,
    )
    assert result.identical


def test_per_side_variable_values():
    options = DiffOptions()

    # same template, same variables -> identical
    result = _diff_pair(
        "vars.inp",
        "vars.inp",
        options,
        var_values_left={"RUN": "ENERGY", "CUT": "400"},
        var_values_right={"RUN": "ENERGY", "CUT": "400"},
    )
    assert result.identical

    # same template, different per-side variable values -> semantic difference
    result = _diff_pair(
        "vars.inp",
        "vars.inp",
        options,
        var_values_left={"RUN": "ENERGY", "CUT": "400"},
        var_values_right={"RUN": "ENERGY", "CUT": "500"},
    )
    assert len(result.changed) == 1
    assert result.changed[0].path == "force_eval[0]/dft/mgrid/cutoff"
    assert result.changed[0].value_left == 400.0
    assert result.changed[0].value_right == 500.0


def test_parse_error_names_file_and_side_and_no_partial_diff():
    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("base.inp", "broken.inp")

    failures = excinfo.value.failures
    assert len(failures) == 1
    failure = failures[0]
    assert failure.side == "right"
    assert failure.source.endswith("broken.inp")
    assert failure.linenr == 3
    assert "NOT_A_SECTION" in failure.message

    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("broken.inp", "base.inp")
    assert excinfo.value.failures[0].side == "left"


def test_parse_error_in_included_file_names_included_file():
    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("base.inp", "broken_include.inp", base_dir_right=INC_DIR)

    failure = excinfo.value.failures[0]
    assert failure.filename.endswith("force_eval_broken.inc")
    assert failure.linenr == 4
    assert "NOTAKEYWORD" in failure.message


def test_missing_include_is_reported_with_filename():
    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("base.inp", "missing_include.inp")

    failure = excinfo.value.failures[0]
    assert failure.source.endswith("missing_include.inp")
    assert "does_not_exist.inc" in failure.message


def test_both_inputs_are_parsed_before_diffing():
    # both sides broken -> both failures must be collected instead of stopping at the first
    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("broken.inp", "broken.inp")

    assert [f.side for f in excinfo.value.failures] == ["left", "right"]


def test_invalid_section_parameter_error():
    with pytest.raises(CP2KDiffParseError) as excinfo:
        _diff_pair("base.inp", "broken_section_param.inp")

    failure = excinfo.value.failures[0]
    assert failure.source.endswith("broken_section_param.inp")
    assert "section parameter" in failure.message


def test_results_are_in_stable_order_with_numeric_aware_indices():
    left = {"+s": [{"k": i} for i in range(11)]}
    right = [{"k": i} for i in range(11)]
    right[1]["k"] = -1
    right[10]["k"] = -1
    right = {"+s": right}

    first = diff_trees(left, right)
    second = diff_trees(left, right)

    assert [e.path for e in first.entries] == [e.path for e in second.entries]
    # s[1] must sort before s[10] (numeric aware, not lexicographic)
    assert first.entries[0].path == "s[1]/k"
    assert first.entries[1].path == "s[10]/k"


# ----------------------------------------------------------------------------
# command line interface
# ----------------------------------------------------------------------------


def test_cli_identical(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "cosmetic.inp")])

    assert ret.success
    assert ret.returncode == 0
    assert ret.stdout == ""
    assert ret.stderr == ""


def test_cli_reports_differences_with_exit_code_1(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "scalar_changed.inp")])

    assert not ret.success
    assert ret.returncode == 1
    assert "force_eval[0]/dft/mgrid/cutoff: 400.0 -> 500.0" in ret.stdout
    assert "global/print_level" in ret.stdout


def test_cli_section_parameter_and_array_paths(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "section_param_changed.inp")])
    assert ret.returncode == 1
    assert "xc_functional/_" in ret.stdout

    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "array_element.inp")])
    assert ret.returncode == 1
    assert "cell/a[1]: 0.0 -> 0.5" in ret.stdout


def test_cli_json_output(script_runner):
    ret = script_runner.run(["cp2k-diff", "--json", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "scalar_changed.inp")])

    assert ret.returncode == 1
    payload = json.loads(ret.stdout)
    assert payload["identical"] is False
    assert payload["left"].endswith("base.inp")
    assert payload["right"].endswith("scalar_changed.inp")

    paths = {entry["path"]: entry for entry in payload["differences"]}
    assert paths["force_eval[0]/dft/mgrid/cutoff"] == {
        "op": "changed",
        "path": "force_eval[0]/dft/mgrid/cutoff",
        "left": 400.0,
        "right": 500.0,
    }


def test_cli_json_output_identical(script_runner):
    ret = script_runner.run(["cp2k-diff", "--json", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "cosmetic.inp")])

    assert ret.returncode == 0
    payload = json.loads(ret.stdout)
    assert payload["identical"] is True
    assert payload["differences"] == []


def test_cli_parse_error_exit_code_2_without_partial_diff(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "broken.inp")])

    assert ret.returncode == 2
    assert not ret.success
    assert "Error parsing right input" in ret.stderr
    assert "broken.inp" in ret.stderr
    assert "line 3" in ret.stderr
    # no partial diff on stdout
    assert ret.stdout == ""


def test_cli_missing_include_names_include(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "missing_include.inp")])

    assert ret.returncode == 2
    assert "does_not_exist.inc" in ret.stderr
    assert ret.stdout == ""


def test_cli_broken_include_names_included_file(script_runner):
    ret = script_runner.run(
        [
            "cp2k-diff",
            "--base-dir-right",
            str(INC_DIR),
            str(DIFF_DIR / "base.inp"),
            str(DIFF_DIR / "broken_include.inp"),
        ]
    )

    assert ret.returncode == 2
    assert "force_eval_broken.inc" in ret.stderr


def test_cli_include_search_directory(script_runner):
    ret = script_runner.run(
        [
            "cp2k-diff",
            "--base-dir",
            str(INC_DIR),
            str(INC_DIR / "main.inp"),
            str(DIFF_DIR / "included_inline.inp"),
        ]
    )

    assert ret.returncode == 0
    assert ret.success


def test_cli_ignore_section_order(script_runner):
    args = [str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "kinds_reordered.inp")]

    assert script_runner.run(["cp2k-diff"] + args).returncode == 1
    assert script_runner.run(["cp2k-diff", "--ignore-section-order"] + args).returncode == 0


def test_cli_tolerance_options(script_runner):
    args = [str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "tolerance_close.inp")]

    assert script_runner.run(["cp2k-diff"] + args).returncode == 1
    assert script_runner.run(["cp2k-diff", "--abs-tol", "1e-3"] + args).returncode == 0
    assert script_runner.run(["cp2k-diff", "--rel-tol", "1e-9"] + args).returncode == 1
    assert script_runner.run(["cp2k-diff", "--rel-tol", "1e-6"] + args).returncode == 0


def test_cli_common_and_per_side_variables(script_runner):
    args = [str(DIFF_DIR / "vars.inp"), str(DIFF_DIR / "vars.inp")]

    ret = script_runner.run(["cp2k-diff", "-E", "RUN=ENERGY", "-E", "CUT=400"] + args)
    assert ret.returncode == 0

    ret = script_runner.run(["cp2k-diff", "-E", "RUN=ENERGY", "--left-set", "CUT=400", "--right-set", "CUT=500"] + args)
    assert ret.returncode == 1
    assert "mgrid/cutoff: 400.0 -> 500.0" in ret.stdout


def test_cli_invalid_set_argument(script_runner):
    ret = script_runner.run(
        [
            "cp2k-diff",
            "-E",
            "missing-equal-sign",
            str(DIFF_DIR / "base.inp"),
            str(DIFF_DIR / "cosmetic.inp"),
        ]
    )

    assert ret.returncode == 2
    assert "Invalid value for '-E'" in ret.stderr


def test_cli_missing_file_argument(script_runner):
    ret = script_runner.run(["cp2k-diff", str(DIFF_DIR / "base.inp"), str(DIFF_DIR / "does-not-exist.inp")])

    assert ret.returncode == 2
    assert "does not exist" in ret.stderr
