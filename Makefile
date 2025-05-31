# Makefile for setting up Python project using Conda + Poetry + requirements.txt

PYTHON_VERSION = 3.12.9  # FYI only - hardcoded in config files below
PIP_VERSION = 25.1       # FYI only - hardcoded in config files below
POETRY_VERSION = 2.1.3   # FYI only - hardcoded in config files below
CONDA_ENV_NAME = gerpa   # Will be used to create Conda environment

# Specifies runtime and environment
CONDA_CONFIG_TEMPLATE = ./gerpa/templates/CONDA.md

# Specifies Python packaging
POETRY_CONFIG_TEMPLATE = ./gerpa/templates/POETRY.md

# Lists Python dependencies - for GERPA itself
PIP_REQUIREMENTS = ./requirements.txt

.PHONY: install conda poetry uninstall

install: conda poetry

conda:
	@echo "📦 Creating Conda environment from file: $(CONDA_CONFIG_FILE)"
	@sed '1d;$$d' $(CONDA_CONFIG_TEMPLATE) > ./environment.yml
	@conda env create -f environment.yml -n $(CONDA_ENV_NAME)
	@conda env export -n $(CONDA_ENV_NAME) > environment.yml
	@echo "✅ Conda environment created."

poetry:
	@echo "🛠️ Configuring Poetry to use Conda env"
	@sed '1d;$$d' $(POETRY_CONFIG_TEMPLATE) > ./pyproject.toml
	@conda run -n $(CONDA_ENV_NAME) poetry config virtualenvs.create false
	@echo "🔄 Adding dependencies from $(PIP_REQUIREMENTS) via Poetry"
	@conda run -n $(CONDA_ENV_NAME) poetry add $(shell cat $(PIP_REQUIREMENTS))
	@echo "🚀 Installing dependencies via Poetry"
	@conda run -n $(CONDA_ENV_NAME) poetry install
	@echo "✅ Environment ready!"

uninstall:
	@echo "🧹 Removing Conda environment $(CONDA_ENV_NAME)"
	@conda env remove -n $(CONDA_ENV_NAME) --yes
