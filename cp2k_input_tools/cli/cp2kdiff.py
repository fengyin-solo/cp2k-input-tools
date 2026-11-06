import json
import pathlib
import sys

import click

from cp2k_input_tools.diff import (
    OP_ADDED,
    OP_CHANGED,
    CP2KDiffParseError,
    DiffOptions,
    diff_paths,
)

from . import click_validate_kv

_EXISTING_DIR = click.Path(exists=True, file_okay=False, path_type=pathlib.Path)


def _merge_dirs(common, specific):
    """side-specific include directories take precedence over the common ones"""
    dirs = tuple(specific) + tuple(common)
    return dirs if dirs else (pathlib.Path("."),)


def _merge_vars(common, specific):
    # side-specific variable values take precedence over the common ones
    return {**dict(common), **dict(specific)}


def _format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def _render_text(result, path_left, path_right):
    if result.identical:
        return ""

    groups = (
        ("Changed values", result.changed),
        ("Added", result.added),
        ("Removed", result.removed),
    )

    lines = [f"semantic differences between {path_left} (left) and {path_right} (right):"]
    for title, entries in groups:
        if not entries:
            continue

        lines.append(f"{title}:")
        for entry in entries:
            if entry.op == OP_CHANGED:
                lines.append(f"  {entry.path}: {_format_value(entry.value_left)} -> {_format_value(entry.value_right)}")
            elif entry.op == OP_ADDED:
                lines.append(f"  {entry.path}: {_format_value(entry.value_right)}")
            else:  # OP_REMOVED
                lines.append(f"  {entry.path}: {_format_value(entry.value_left)}")

    return "\n".join(lines) + "\n"


def _render_json(result, path_left, path_right):
    return (
        json.dumps(
            {
                "left": str(path_left),
                "right": str(path_right),
                "identical": result.identical,
                "differences": [
                    {
                        "op": entry.op,
                        "path": entry.path,
                        "left": entry.value_left,
                        "right": entry.value_right,
                    }
                    for entry in result.entries
                ],
            },
            indent=2,
        )
        + "\n"
    )


def _render_parse_errors(error: CP2KDiffParseError):
    lines = []
    for failure in error.failures:
        lines.append(f"Error parsing {failure.side} input '{failure.source}': {failure.message}")
        if failure.filename and failure.filename != failure.source:
            location = f"  in included file {failure.filename}"
            if failure.linenr is not None:
                location += f", line {failure.linenr}"
            lines.append(location)
        elif failure.linenr is not None:
            lines.append(f"  at line {failure.linenr}")

    return "\n".join(lines) + "\n"


@click.command(name="cp2k-diff")
@click.argument("file_left", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.argument("file_right", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option(
    "-b",
    "--base-dir",
    "base_dirs",
    multiple=True,
    type=_EXISTING_DIR,
    help="directory searched for relative @include's in both inputs (can be given multiple times)",
)
@click.option(
    "--base-dir-left",
    multiple=True,
    type=_EXISTING_DIR,
    help="directory searched for relative @include's in the left input only",
)
@click.option(
    "--base-dir-right",
    multiple=True,
    type=_EXISTING_DIR,
    help="directory searched for relative @include's in the right input only",
)
@click.option(
    "-E",
    "--set",
    "var_values",
    metavar="key=value",
    multiple=True,
    type=click.UNPROCESSED,
    callback=click_validate_kv,
    help="preset the value of a CP2K preprocessor variable in both inputs",
)
@click.option(
    "--left-set",
    metavar="key=value",
    multiple=True,
    type=click.UNPROCESSED,
    callback=click_validate_kv,
    help="preset the value of a CP2K preprocessor variable in the left input only",
)
@click.option(
    "--right-set",
    metavar="key=value",
    multiple=True,
    type=click.UNPROCESSED,
    callback=click_validate_kv,
    help="preset the value of a CP2K preprocessor variable in the right input only",
)
@click.option("--abs-tol", type=float, default=0.0, show_default=True, help="absolute tolerance for numeric comparisons")
@click.option("--rel-tol", type=float, default=0.0, show_default=True, help="relative tolerance for numeric comparisons")
@click.option(
    "--ignore-section-order",
    is_flag=True,
    default=False,
    help="do not treat the order of repeatable sections as significant",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="emit the differences as JSON")
def cp2kdiff(
    file_left,
    file_right,
    base_dirs,
    base_dir_left,
    base_dir_right,
    var_values,
    left_set,
    right_set,
    abs_tol,
    rel_tol,
    ignore_section_order,
    as_json,
):
    """Compare two CP2K inputs semantically using the same CP2K XML input definition.

    Formatting, indentation, casing, comments and the way @include files are
    split up are ignored: only values that can change the calculation are
    reported. Exit status is 0 when the inputs are equivalent, 1 when they
    differ and 2 on (parse) errors.
    """

    options = DiffOptions(abs_tol=abs_tol, rel_tol=rel_tol, ignore_section_order=ignore_section_order)

    try:
        result = diff_paths(
            file_left,
            file_right,
            options,
            base_dir_left=_merge_dirs(base_dirs, base_dir_left),
            base_dir_right=_merge_dirs(base_dirs, base_dir_right),
            var_values_left=_merge_vars(var_values, left_set),
            var_values_right=_merge_vars(var_values, right_set),
        )
    except CP2KDiffParseError as error:
        sys.stderr.write(_render_parse_errors(error))
        sys.exit(2)

    if as_json:
        click.echo(_render_json(result, file_left, file_right), nl=False)
    else:
        click.echo(_render_text(result, file_left, file_right), nl=False)

    if not result.identical:
        sys.exit(1)
