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
        cmd = ['python', 'manuscript/build/bibmerge.py', 'manuscript/build/.bibpaths']
        if verbose:
            click.echo(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    click.echo("Concatenating markdown files...")
    front_md = Path('manuscript/front.md').read_text()
    body_md = Path('manuscript/body.md').read_text()
    refs_md = Path('manuscript/refs/refs.md').read_text()

    temp_md_content = f"{front_md}\n\n{body_md}\n\n{refs_md}"
    temp_md_path = Path('manuscript.md')
    temp_md_path.write_text(temp_md_content)

    click.echo("Rendering with quarto...")
    cmd = ['quarto', 'render', str(temp_md_path), '--to', 'docx', '--output', 'manuscript.docx', '--metadata-file=manuscript/build/head.yml']
    if verbose:
        click.echo(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if verbose:
        click.echo(result.stdout)
        click.echo(result.stderr)

    click.echo("Cleaning up temporary files...")
    temp_md_path.unlink()

    click.echo("Manuscript build complete.")

if __name__ == '__main__':
    makerepref()
```
