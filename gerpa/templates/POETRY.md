```ini
[tool.poetry]
name = "GERPA"
version = "0.0.1"
description = "Generator of Environments for Rapid Prototyping of Agents"
authors = ["Pavel Zhelnov <pzhelnov@p1m.org>"]
license = "Apache-2.0"
readme = "README.md"
packages = [{ include = "gerpa" }]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.dependencies]
python = "^3.12.9"
```
