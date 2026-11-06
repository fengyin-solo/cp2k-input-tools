import sys
from pathlib import Path

import click

from cp2k_input_tools.merger import (
    DEFAULT_MERGE_POLICY,
    MergeError,
    MergePolicy,
    RepeatPolicy,
    merge_inputs,
    write_lines_atomic,
)
from cp2k_input_tools.parser_errors import ParserError
from cp2k_input_tools.tokenizer import TokenizerError

from . import base_dir_option, var_values_option, xml_option


def _is_same_file(path1: Path, path2: Path) -> bool:
    """check whether two paths refer to the same file (by resolved path or by inode)"""
    try:
        if path1.resolve() == path2.resolve():
            return True
    except OSError:
        pass

    try:
        stat1, stat2 = path1.stat(), path2.stat()
        return (stat1.st_dev, stat1.st_ino) == (stat2.st_dev, stat2.st_ino)
    except OSError:
        return False


@click.command(name="cp2k-merge")
@click.argument("base_input", metavar="<base-input>", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("override_input", metavar="<override-input>", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="path of the merged CP2K input to write (default: write to stdout)",
)
@click.option(
    "--repeated-sections",
    type=click.Choice([policy.value for policy in RepeatPolicy]),
    default=DEFAULT_MERGE_POLICY.repeated_sections.value,
    show_default=True,
    help="how to merge repeated sections without a section parameter",
)
@click.option(
    "--repeated-keywords",
    type=click.Choice([policy.value for policy in RepeatPolicy]),
    default=DEFAULT_MERGE_POLICY.repeated_keywords.value,
    show_default=True,
    help="how to merge repeated keywords (arrays)",
)
@click.option(
    "--existing-only",
    is_flag=True,
    default=False,
    help="only allow override paths (sections, keywords and section parameters) already present in the base input",
)
@base_dir_option
@var_values_option
@xml_option
def cp2k_merge(
    base_input,
    override_input,
    output,
    repeated_sections,
    repeated_keywords,
    existing_only,
    base_dir,
    var_values,
    xml,
):
    """
    Merge a base CP2K input with an override CP2K input into a new, validated CP2K input.

    Ordinary sections are merged recursively, scalar keywords are replaced by the
    override value, parameterized repeated sections (e.g. '&KIND H') are matched by
    their parameter and merged recursively. Repeated sections without a parameter and
    repeated keyword arrays follow the selected replace/append policies.

    The merged result is regenerated and re-parsed against the CP2K XML definition
    before anything is written and the output file is replaced atomically.
    """

    if _is_same_file(base_input, override_input):
        raise click.ClickException("the base input and the override input are the same file")

    if output is not None and (_is_same_file(output, base_input) or _is_same_file(output, override_input)):
        raise click.ClickException("the output path must be different from both input paths")

    policy = MergePolicy(
        repeated_sections=RepeatPolicy(repeated_sections),
        repeated_keywords=RepeatPolicy(repeated_keywords),
    )

    try:
        with base_input.open("r") as base_fhandle, override_input.open("r") as override_fhandle:
            _, lines = merge_inputs(
                base_fhandle,
                override_fhandle,
                xmlspec=xml,
                base_dir=base_dir,
                initial_variable_values=dict(var_values),
                policy=policy,
                only_existing_paths=existing_only,
            )
    except (TokenizerError, ParserError) as exc:
        message = exc.args[0] if exc.args else "invalid input"
        context = exc.args[1] if len(exc.args) > 1 else None
        if context is not None and getattr(context, "filename", None):
            click.echo(f"Error: invalid CP2K input in {context.filename} (line {context.linenr}): {message}", err=True)
        else:
            click.echo(f"Error: invalid CP2K input: {message}", err=True)
        sys.exit(1)
    except MergeError as exc:
        raise click.ClickException(str(exc)) from exc

    if output is None:
        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\n")
    else:
        # the merged tree has already been fully generated and re-parsed successfully,
        # so the atomic replace is the only step touching the destination
        write_lines_atomic(output, lines)
        click.echo(f"merged CP2K input written to '{output}'", err=True)
