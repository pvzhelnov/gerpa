# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Prompts with long parts would fail because could not check if they were a path/URL

## [0.0.1] - 2025-06-05

<!--- This section was initially generated with Gemini 2.5 Pro (preview) on 2025-06-05, see chats/2025/06/05/gemini/ --->

### Added

-   Initial project scaffolding including main script, environment configuration (`environment.yml`), and `.gitignore`.

-   `gerpa` script for global project launch with embedded instructions.

-   Full "Paris" example demonstrating functionality with integrated evaluations.

-   Installation instructions to `README.md`.

-   Conda and Poetry configuration files (generated post `make install`).

-   Support for `List[str]` as prompt input, with automatic file uploading (from local path or HTTP URL) to the Gemini API. Unsupported file types will result in an API error.

-   Support for system instructions, passable as `kwargs` to the agent.

-   `poetry bump version` command for streamlined version management.

-   Initial `CHANGELOG.md` file, structured according to Keep a Changelog principles.

-   `git-cliff` (v2.9.1) integration for automated changelog generation, initialized with the `keepachangelog` template.

-   Project licensed under Apache 2.0.

-   Foundational claude chat integration (initial version).

### Changed

-   **[MAJOR]** Core application logic refactored from a monolithic `main` script into a structured system utilizing `Template` and `TemplateType` classes.

    -   Evaluator logic and Command Line Interface (CLI) interactions are now fully encapsulated within these templates, eliminating the need for dynamic module imports for evaluators.

    -   Generated projects are now Poetry-enabled (though `requirements.txt` remains for flexibility and alternative Conda-based setup).

    -   Redundant import statements in the main script's preamble were removed.

    -   Implemented logic for replacing default parameters within template files during project generation.

-   **[MAJOR]** Dependency management migrated to Poetry, complemented by a Makefile for Conda environment setup and Poetry installation. `requirements.txt` has been streamlined to include only essential CLI tool packages, with LLM/data-related packages managed within project templates.

-   **[MAJOR]** OpenRouter and Ollama providers have been retired to `NotImplemented` status, to concentrate development efforts on the Gemini provider.

-   **[MAJOR]** A response schema is now mandatory for agent interactions to enforce structured output, leveraging this as a key feature for prototyping.

-   **[MAJOR]** LLM response dump format changed to JSON (from Pythonic object representation). This simplifies YAML loading for evaluations and enhances log clarity by removing class definitions from the output.

-   **[MAJOR]** The `model` and `provider` attributes were removed from the `LLMResponse` object, as this information is already available within the agent class instance.

-   **[MAJOR]** Evaluation system significantly enhanced:

    -   Nested evaluations are now supported, allowing manual evaluations to mirror the structure of complex response schemas.

    -   Manual evaluation logic overhauled: now requires a `pass` status and a score greater than 0 for a "pass" outcome; otherwise, it's marked as "fail" (unless explicitly skipped). The system returns a clear label to the user, indicating the expectation and the resulting score. Manual evaluations can be defined either as a detailed dictionary (with `manual_result`, `manual_score`, `manual_reason`) or as a concise one-liner (`pass` or `fail`, with scores defaulting to 1.0 for pass and 0.0 for fail if not specified).

-   **[MAJOR]** Agent interaction model refined:

    -   The `model` argument was renamed to `model_name` for clarity.

    -   Extended `kwargs` support enables more flexible parameter passing from the agent.

    -   Introduced a `BaseLLM` class providing a minimum configuration baseline and including settings for content safety/censorship.

-   Main script code updated for compatibility with the newer `google-genai` library, including the addition of structured output formatting.

-   A Pydantic model is now hardcoded in the agent's response mechanism.

-   The environment variable name referenced in the main script has been replaced with the project name.

-   YAML file extensions standardized to `.yml` (from `.yaml`) for common practice.

-   Response data now includes the full schema definition, not just the schema name, providing more context.

-   The "Paris" example's full response schema was updated to align with recent structural changes.

-   The `gerpa` script example in `README.md` was reworked to include more comprehensive setup and execution instructions for macOS, Linux, and WSL2 (Windows Subsystem for Linux), including guidance on avoiding root access needs.

-   `.gitignore` file updated using a standard template from `github/gitignore` for improved standardization and reliability (e.g., ensuring `poetry.lock` is not ignored).

-   The Makefile and the project template directory structure were refactored for better organization. Templates are now designed to feature their own classes.

-   The `GeminiProvider` now defaults to using the `gemini-2.0-flash` model, following Google's model lifecycle updates.

-   Enabled Unicode character support in YAML dumps and log outputs.

-   Documentation: A note was added indicating that `response_json_schema` is not yet implemented in the Gemini provider configuration.

-   `README.md` improvements:

    -   Updated with specific instructions for Conda users.

    -   PATH rewrite instructions replaced with an `alias` command for enhanced security.

    -   Expanded with additional details for overall clarity.

-   Internal imports within the `llmprovider` were expanded to allow for more flexible and comfortable modeling.

-   The `git-cliff` template was modified to better parse and categorize the project's specific commit message styles.

### Fixed

-   Corrected a typo in a request patch number within the main script; the fix was previously in `environment.yml` but had not been propagated.

-   Resolved an issue with the evaluator loading mechanism.

-   In `README.md` commands, `sh` was corrected to `bash` for Conda hook compatibility, and a missing colon was fixed.

-   Addressed issues in the initial Claude chat export functionality.

[unreleased]: https://github.com/pvzhelnov/gerpa/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/pvzhelnov/gerpa/releases/tag/v0.0.1
