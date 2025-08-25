```python
import click
import subprocess
from pathlib import Path

@click.command()
@click.option('--bibmerge/--no-bibmerge', default=False, help='Run bibmerge.py to merge bibliography files.')
@click.option('--verbose/--no-verbose', default=False, help='Enable verbose output.')
def makerepref(bibmerge, verbose):
    """
    Builds the manuscript document.
    """
    if bibmerge:
        click.echo("Running bibmerge...")
        # Import and run bibmerge directly
        from bibmerge import BibMerger
        bibpaths_file = 'manuscript/build/.bibpaths'
        bib_merger = BibMerger(bibpaths_file)
        bib_files = bib_merger.get_bib_files_from_paths()
        if len(bib_files) == 0:
            print("No *.bib files found to merge.")
        else:
            # Merge and dump contents of individual bib files
            try:
                output_file = bib_merger.merge_bib_files(bib_files)
                print(f"Merged {len(bib_files)} *.bib files into {output_file}.")
            except Exception as e:
                print(f"Found {len(bib_files)} *.bib files.")
                print(f"Error occurred when merging: {e}")

    click.echo("Concatenating markdown files...")
    front_md = Path('manuscript/front.md').read_text()
    body_md = Path('manuscript/body.md').read_text()
    refs_md = Path('manuscript/refs/refs.md').read_text()
    yaml_content = Path('manuscript/build/head.yml').read_text()

    qmd_content = f"---\n{yaml_content}\n---\n\n{front_md}\n\n{body_md}\n\n{refs_md}"
    qmd_path = Path('manuscript/build/manuscript.qmd')
    qmd_path.write_text(qmd_content)

    click.echo("Rendering with quarto...")
    cmd = ['quarto', 'render', 'manuscript.qmd', '--to', 'docx']
    if verbose:
        click.echo(f"Command: {' '.join(cmd)}")
        click.echo(f"Rendering {qmd_path} to docx with metadata from manuscript/build/head.yml")
    else:
        cmd.append('--quiet')

    result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd='manuscript/build')
    if verbose:
        click.echo(result.stdout)
        click.echo(result.stderr)

    click.echo("Manuscript build complete.")

if __name__ == '__main__':
    makerepref()
```
