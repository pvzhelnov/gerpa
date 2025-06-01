#!/usr/bin/env python3
"""
LLM Prototyping Monorepo CLI
A minimal CLI for fast LLM prototyping with versioned prompts, responses, and evaluations.
"""

import os
import click
import subprocess
from pathlib import Path
from typing import Optional, Any
import re
import importlib.resources
import toml
from enum import Enum

def _infer_package_name() -> str:
    """Infer package name from current file"""
    return Path(__file__).parent.name or "gerpa"

def _read_config_file(
        config_file_name: Optional[str] = "pyproject.toml") -> dict[str, Any]:
    """Attempt to read config file"""
    try:
        pyproject_path = importlib.resources.files(_infer_package_name()).parent / config_file_name
        if os.path.exists(pyproject_path):
            with open(pyproject_path, "r") as f:
                pyproject_data = toml.load(f)
    except:
        pyproject_data = {}
    return pyproject_data

def _get_package_name() -> str:
    """Attempt to get package name from config file"""
    try:
        pyproject_data = _read_config_file()
        packages = pyproject_data.get("tool", {}).get("poetry", {}).get("packages")
        if packages:
            for package in packages:
                if "include" in package:
                    package_name = package["include"]
                    break  # Take only the first one found
    except:
        package_name = _infer_package_name()
    return package_name

def _get_version():
    """Attempt to get version from config file"""
    try:
        pyproject_data = _read_config_file()
        version = pyproject_data.get("tool", {}).get("poetry", {}).get("version")
    except:
        version = None
    return version

class TemplateType(Enum):
    """Specifies subdir within templates dir."""
    CONFIG = "configs"
    MODULE = "modules"
    DOCS = "docs"
    NONE = None

class Template():
    """Specifies Markdown templates reused below."""
    def __init__(
            self,
            template_slug: str,
            template_type: Optional[str] = None):
        self.template = None
        self.template_dir = 'templates'
        self.template_filename = 'TEMPLATE.md'
        self._compile_md_code_block()
        self._load_template(template_slug, template_type)
         
    def _compile_md_code_block(self):
        pattern = r'```([a-z0-9]*)\n(.*)\n```'
        flags=re.DOTALL | re.IGNORECASE
        self.md_code_block = re.compile(pattern, flags)

    def _load_template(self,
                       template_slug: str,
                       template_type: Optional[TemplateType] = None) -> str | None:
        self.template_slug = template_slug
        self.template_type = template_type
        try:
            # Get the path to the 'gerpa' package
            package_path = importlib.resources.files(_get_package_name())

            # Construct the full path to the template file
            if ((template_type is None) or
                (template_type is TemplateType.NONE)):
                template_path = package_path / self.template_dir / template_slug / self.template_filename
            else:
                template_path = package_path / self.template_dir / template_type.value / template_slug / self.template_filename

            # Open the template file
            with template_path.open('r') as f:
                content = f.read()
                # Use regex to find the code block for the specified language
                match = re.search(self.md_code_block, content)
                if match:
                    lang = match.group(1).strip()
                    template = match.group(2).strip()
                    self.template = template
        except ModuleNotFoundError:
            print(f"Error: '{_get_package_name()}' package not found.")
        except FileNotFoundError:
            print(f"Error: Template file '{template_path}' not found.")
        except Exception as e:
            print(f"An error occurred while loading the template: {e}")
    
    def __str__(self) -> str:
        return str(self.template or '')

# Template files content
### docs
ENV_TEMPLATE = Template('dotenv/example', TemplateType.DOCS)
LICENSE_TEMPLATE = Template('license/apache-2.0', TemplateType.DOCS)
### configs
GITIGNORE_TEMPLATE = Template('git/gitignore', TemplateType.CONFIG)
CONDA_TEMPLATE = Template('conda/environment', TemplateType.CONFIG)
POETRY_TEMPLATE = Template('poetry/pyproject', TemplateType.CONFIG)
PIP_REQUIREMENTS = Template('pip/requirements', TemplateType.CONFIG)
MAKEFILE_TEMPLATE = Template('make/file', TemplateType.CONFIG)
### modules
LLM_PROVIDER_CODE = Template('python/llm_provider', TemplateType.MODULE)
EVALUATOR_CODE = Template('python/evaluator', TemplateType.MODULE)
### other
NOTEBOOK_CODE = Template('notebook/experiment', TemplateType.NONE)

# CLI Interface
@click.group()
def cli():
    """LLM Prototyping Monorepo CLI"""
    pass


@cli.command()
@click.argument('project_name')
@click.option('--git/--no-git', default=True, help='Initialize git repository')
def init(project_name: str, git: bool):
    """Initialize a new LLM prototyping project"""
    project_path = Path(project_name)
    
    if project_path.exists():
        click.echo(f"Error: Directory {project_name} already exists")
        return
        
    # Create project structure
    project_path.mkdir()
    
    # Create directories
    directories = [
        'logs', 'responses', 'prompts', 'eval_results', 'untracked'
    ]
    for dir_name in directories:
        (project_path / dir_name).mkdir()

    # To be replaced with proper yaml handler
    yaml_lines = str(CONDA_TEMPLATE).strip().split('\n')
    if yaml_lines and yaml_lines[0].startswith('name:'):
        yaml_lines[0] = f"name: {project_name}"
    edited_conda_template = '\n'.join(yaml_lines)

    # Update pyproject.toml values for generated project
    pyproject_data = toml.loads(str(POETRY_TEMPLATE))
    # Update fields in the loaded dict
    pyproject_data['tool']['poetry']['name'] = project_name
    pyproject_data['tool']['poetry']['description'] = f"Generated with {_get_package_name().upper()} {_get_version()}".rstrip()
    pyproject_data['tool']['poetry']['authors'] = ["Unknown Author <anonymous@example.com>"]
    pyproject_data['tool']['poetry']['packages'] = [{"include": "modules"}]
    # If you want to convert back to TOML string
    edited_poetry_template = toml.dumps(pyproject_data)

    # Update Makefile
    makefile_lines = str(MAKEFILE_TEMPLATE).strip().split('\n')
    try:
        for i, line in enumerate(makefile_lines):
            if line.startswith('CONDA_ENV_NAME = '):
                makefile_lines[i] = f'CONDA_ENV_NAME = {project_name}'
        edited_makefile_template = '\n'.join(makefile_lines)
    except:
        edited_makefile_template = str(MAKEFILE_TEMPLATE)
        
    # Create project-level files
    project_files = {
        '.gitignore': GITIGNORE_TEMPLATE,
        'requirements.txt': PIP_REQUIREMENTS,
        'environment.yml': edited_conda_template,
        'pyproject.toml': edited_poetry_template,
        'experiment.ipynb': NOTEBOOK_CODE,
        '.env.example': ENV_TEMPLATE,
        'Makefile': edited_makefile_template,
    }
    for filename, template in project_files.items():
        file_path = project_path / filename
        file_path.write_text(str(template))

    # Create subdir for package
    package_path = (project_path / 'modules')
    package_path.mkdir()

    # Create modules
    package_files = {
        'llm_provider.py': LLM_PROVIDER_CODE,
        'evaluator.py': EVALUATOR_CODE,
    }
    for filename, template in package_files.items():
        file_path = package_path / filename
        file_path.write_text(str(template))

    # Create empty .env file
    (project_path / '.env').touch()

    # Create Apache-2.0 license file
    (project_path / 'LICENSE').write_text(str(LICENSE_TEMPLATE))

    # Create README.md file with initial content
    (project_path / 'README.md').write_text(
        f"# {project_name}\nGenerated with {_get_package_name().upper()} {_get_version()}".rstrip() + "\n"
    )
    
    # Initialize git if requested
    if git:
        try:
            subprocess.run(['git', 'init'], cwd=project_path, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.'], cwd=project_path, check=True, capture_output=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=project_path, check=True, capture_output=True)
            click.echo(f"✅ Git repository initialized in {project_name}")
        except subprocess.CalledProcessError as e:
            click.echo(f"⚠️  Warning: Git initialization failed: {e}")
            
    click.echo(f"✅ Project {project_name} created successfully!")
    click.echo(f"\nNext steps:")
    click.echo(f"1. cd {project_name}")
    click.echo(f"2. make conda poetry")
    click.echo(f"3. conda activate {project_name}")
    click.echo(f"4. Copy .env.example to .env and add your API keys")
    click.echo(f"5. jupyter notebook experiment.ipynb")
    click.echo(f"6. python -m modules.evaluator")

@cli.command()
def version():
    """Show version information"""
    __version__ = _get_version()
    if __version__ is not None:
        click.echo(f"{_get_package_name().upper()} CLI v{__version__}".strip())
    else:
        click.echo(f"{_get_package_name().upper()} CLI version not found".lstrip())

if __name__ == '__main__':
    cli()
