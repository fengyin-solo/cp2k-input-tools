import re
import sys
from pathlib import Path

import click

from cp2k_input_tools.lineiterator import LineContinuationError
from cp2k_input_tools.parser_errors import PreprocessorError
from cp2k_input_tools.preprocessor import (
    OutputPathConflictError,
    atomic_write_text,
    ensure_no_output_conflict,
    preprocess,
    source_map_to_json,
)
from cp2k_input_tools.tokenizer import TokenizerError

from . import fhandle_argument, var_values_option


def _print_preprocessor_error(exc):
    ctx = exc.args[1]
    line = ctx.line.rstrip() if ctx.line else ""
    filename = ctx.filename or "<stdin>"
    linenr = getattr(ctx, "linenr", None)  # attached dynamically, missing for end-of-file errors

    print(f"Preprocessing error: {exc.args[0]}, in {filename}:")

    if exc.__cause__:
        print(f"              {exc.__cause__}")

    if linenr is not None:
        print(f"line {ctx.linenr:>4}: {line}")

        if ctx.colnr is not None:
            count = 0  # number of underline chars after (positiv) or before (negative) the marker if ref_colnr given
            nchars = ctx.colnr  # relevant line length

            if ctx.ref_colnr is not None:
                count = ctx.ref_colnr - ctx.colnr
                nchars = min(ctx.ref_colnr, ctx.colnr)  # correct if ref comes before

            if ctx.colnrs:
                # shift by the number of left-stripped ws
                # ctx.colnrs contains the left shift for each possibly continued line
                nchars += ctx.colnrs[0]  # assume no line-continuation for now

            # replace all non-ws chars with spaces:
            # - assuming a monospace font
            # - preserving other whitespace we don't know the width
            underline = re.sub(r"\S", " ", ctx.line[:nchars])

            if count >= 0:
                print(f"{str():>9}  {underline}^{str():~>{count}}")
            else:
                print(f"{str():>9}  {underline}{str():~>{-count}}^")

    if ctx.ref_line is not None:
        print("previous definition:")
        print(f"line {str():>4}: {ctx.ref_line.rstrip()}")


@click.command()
@fhandle_argument
@var_values_option
@click.option(
    "-I",
    "--include-dir",
    "include_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="directory searched for relative @INCLUDE/@XCTYPE files, may be repeated "
    "(searched in the given order, before the default search directory)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="write the preprocessed CP2K input to this file instead of standard output " "(the file is replaced atomically)",
)
@click.option(
    "--source-map",
    "source_map",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="write a JSON source map with, for every output line, the originating absolute file path, "
    "original line range, columns and the include chain (the file is replaced atomically)",
)
def cp2kpreprocess(fhandle, var_values, include_dirs, output, source_map):
    """
    Expand the CP2K preprocessor directives (@INCLUDE, @XCTYPE, @IF/@ENDIF, @SET and
    ${variable}/$variable substitution) of the given CP2K input file (or standard input)
    and write the resulting plain CP2K input to standard output or a file.

    No CP2K schema validation is performed and includes that cannot be opened are a hard error.

    The default search directory for relative @INCLUDE/@XCTYPE files is the directory of the
    input file (or the current directory when reading from standard input).
    """

    input_path = Path(fhandle.name) if getattr(fhandle, "name", None) and fhandle.name != "<stdin>" else None

    try:
        ensure_no_output_conflict(input_path, output, source_map)
    except OutputPathConflictError as exc:
        raise click.ClickException(str(exc)) from None

    search_dirs = list(include_dirs)
    search_dirs.append(input_path.parent if input_path is not None else Path("."))

    try:
        result = preprocess(fhandle, base_dir=search_dirs, initial_variable_values=dict(var_values))
    except (PreprocessorError, TokenizerError) as exc:
        _print_preprocessor_error(exc)
        sys.exit(1)
    except LineContinuationError as exc:
        print(f"Preprocessing error: {exc}")
        sys.exit(1)

    if output is not None:
        atomic_write_text(output, result.text)
    else:
        click.echo(result.text, nl=False)

    if source_map is not None:
        atomic_write_text(source_map, source_map_to_json(result.source_map))
