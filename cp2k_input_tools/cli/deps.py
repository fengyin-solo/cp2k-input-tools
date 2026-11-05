import pathlib
import sys

import click

from cp2k_input_tools.dependencies import DependencyScanError, scan_dependencies

from . import var_values_option


@click.command()
@click.argument(
    "fpath",
    metavar="[<file>]",
    type=click.Path(allow_dash=True, dir_okay=False, path_type=pathlib.Path),
    default="-",
)
@click.option(
    "-I",
    "--search-dir",
    "search_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help=(
        "additional directory searched for relative @INCLUDE/@XCTYPE targets, "
        "after the directory of the referencing file; may be given multiple times"
    ),
)
@var_values_option
@click.option(
    "--check",
    "check",
    is_flag=True,
    help="exit with a non-zero status if a reachable include target is missing "
    "or a dependency cycle exists (2: missing, 3: cycle, 4: both)",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(("json", "text")),
    default="json",
    show_default=True,
    help="output format",
)
def cp2k_deps(fpath, search_dirs, var_values, check, fmt):
    """Scan a CP2K input file for its @INCLUDE/@XCTYPE dependency graph

    The scan follows @INCLUDE, @XCTYPE, @SET and @IF/@ENDIF directives without parsing
    the CP2K input schema. Nodes distinguish the entry file, plain includes, XCTYPE files
    and missing targets; edges record the reference type, the source line and whether the
    reference is located in a disabled @IF branch.

    Relative targets are resolved first against the directory of the referencing file and
    afterwards against the directories given via -I/--search-dir. The report lists missing
    files, dependency cycles and the files reachable from the entry through enabled
    branches (i.e. those entering the final expansion). JSON output is stable and suitable
    for diffing. With --check the exit status is non-zero on problems: 2 for missing
    targets, 3 for cycles, 4 for both. Read from standard input if no file is given.
    """

    try:
        if str(fpath) == "-":
            graph = scan_dependencies(sys.stdin, search_dirs=search_dirs, initial_variable_values=var_values)
        else:
            graph = scan_dependencies(fpath, search_dirs=search_dirs, initial_variable_values=var_values)
    except DependencyScanError as exc:
        location = ""
        if exc.filename:
            location += f", in {exc.filename}"
        if exc.linenr is not None:
            location += f", line {exc.linenr}"
        click.echo(f"cp2k-deps: error: {exc.message}{location}", err=True)
        sys.exit(1)

    if fmt == "text":
        click.echo(graph.to_text(), nl=False)
    else:
        click.echo(graph.to_json(), nl=False)

    if check:
        sys.exit(graph.check_status)
