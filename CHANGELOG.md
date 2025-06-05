# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add initial claude chat\
Did not test anything yet.

- Add gerpa script for global launch\
Instructions within.

- Add full Paris example with evals\
From test generated environment.\
This all seems to work!!

- Add chat on file handling implem\
Testing on test repo yet - not yet integrated in main.\
As before, had to copy and paste all code chunks from claude.ai manually.

- Add gemini chat on meta notimplemented class\
Still in testing - not migrated to main yet.\
However, this seems to have been frutiful.

- Add installation instructions to README

- Add Conda/Poetry config after Makefile\
Just ran make install and committing results.

- Add support for List[str] as prompt, including files\
So if a string that looks like a path or HTTP URL is passed, this will now attempt to upload this file to Gemini.\
Note that unsupported files are not specifially excepted here, so they will just return an API error.

- Add support for system instruction\
Can be passed to agent as part of kwargs.

- Add poetry bump version\
For easier version bumps.\
Changelog will follow.


### Changed

- Fix initial claude chat export\
Perhaps the extension is out of date again.\
There were only two versions in fact, so I removed v3 file.

- Create main, init, env & upd gitignore\
Main copied and pasted from chat, just restored missing closing triple single risk on one line.\
Gitignore - same, copied from chat but left untracked ignore in.\
Environment - copied initially but then removed all pinned versions and installed latest, then copied pinned versions manually from conda (auto export was very long).

- Upd gitignore and env in main\
Copied and pasted back from this repo's.

- Fix main code for new google-genai\
I already fixed the dep in prior commit but only fixing code now.\
Also, this adds format structured output.

- Hardcode pydantic model in agent response\
I tested this on a test repo and this works.

- Replace env name with project name in main\
Replacement code is clumsy but ok for now.

- Fix typo in request patch num in main\
Fixed in env yml but forgot to copy in main.

- Move deps to requirments\
For easier updates and usability.\
Changed both in repo and in main.

- Fix evaluator loading\
Did not test yet.\
Chat export initially automatic but did not include code snippets, so I had to copy and paste them manually.

- Test evals and add manual eval\
Tested on a sample repo and seems to work.

- Replace yaml with yml extensions\
This is more common afaik.

- Save response schema instead of name\
Response data now include full schema.\
I tested this on test repo and it works.

- Upd full response schema in paris example\
To conform with edit from parent commit.

- License under Apache 2.0

- Log chat with manual eval implement\
Lots copied and pasted manually because no code chunks are captured by extension.

- Rework gerpa example to README.md\
Also added more extensive instructions to make gerpa executable on macOS, linux, or WSL2 under Windows while also removing root access needs.

- Reorder chats for better structure

- Upd gitignore with relevant github/gitignore\
Makes it a lot more standardized and reliable.\
For example, the LLM-generated version had poetry.lock ignored by default.

- Migrate to poetry (+ conda + pip requirements)\
This is to use the good potential of poetry in managing packaging (including poetry.lock for reproducbility).\
So how this works now is that there is a Makefile with install and uninstall commands:\
- make install sets a new conda env, installs poetry there, and then adds all packages from requirements.txt and installs them through poetry. - make uninstall removes entire conda env.\
This set-up makes use of poetry but keeps the local dir nice and clean.\
The templates for CONDA and POETRY are taken from the new templates dir within package, which will now also be reused by the package.\
Also, requirements.txt is now lean and only includes packages that are actually used by the CLI tool, and all the LLM/data related packages will remain within REQUIREMENTS template (to be added).

- Upd README for conda users

- Refactor Makefile and templates tree\
Templates will feature their own classes - to be added.

- [MAJOR] Refactor of main into templates\
All code chunks moved to their own templates now, with Template and TemplateType classes introduced for more structure.\
Also, all evaluator logic, including CLI, has been now moved entirely to the template so as to remove the need to import evaluator as module dynamically and deal with deps import. In fact, there is no need to call evals from gerpa.\
This also makes the generated project poetry enabled - optional because requirements.txt is still in place, and make conda can also be called without calling make poetry\
I also removed most redundant imports in the preamble of main.\
There is also some logic now for replacing default params in template files at generation.

- Default to gemini-2.0-flash in GeminiProvider\
Google said it retired 1.5:\
Model ID | Release date | Retirement date | Recommended upgrade ---|---|---|--- `gemini-1.5-pro-002`* | September 24, 2024 | September 24, 2025 | `gemini-2.0-flash` `gemini-1.5-flash-002`* | September 24, 2024 | September 24, 2025 | `gemini-2.0-flash-lite` `text-embedding-004` | May 14, 2024 | November 18, 2025 | `gemini-embedding-001`\
https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#legacy-stable

- Allow unicode in yaml dump and logs\
Why not?

- [MAJOR] Retire OpenRouter and Ollama providers to NotImplemented\
Only working on Gemini for now to keep it focused but will come back to these soon.

- Note not implemented response_json_schema in Gemini config\
See attached pasted.md for details (closer to the end of the file).

- [MAJOR] Mandate response schema for agent\
I think this is great as forces structured output, and ultimately it is the killer feature for this prototyping package.

- [MAJOR] change response dump format to json\
This leads to the LLM response being saved (ultimately in the YAML file) as a simple JSON object, not a Pythonic object.\
It is thus not fully possible to load a Pythonic var after this change - it is, however, simpler to load the YAML for other purposes such as evals, and it looks cleaner because class definitions are absent.\
This change is not visible with simple schemas such as the one from example 1.

- [MAJOR] Remove model and provider from LLMResponse\
I figured that they are not needed there because they are part of agent class instance anyway.

- Replace PATH rewrite with alias in README\
For more security - see attached chat.

- Expand README a bit more for clarity

- Fix sh -> bash and colon in README\
So for bash, it just makes sense because the conda hook works for bash.

- [MAJOR] Allow nested evals and change manual eval logic\
Previously this would throw an exception. Yet nested evals are very helpful to replicate the structure of the response schema in manual evals.\
In terms of the new manual eval logic, now this expects a pass and a score above zero, otherwise it will put a fail (unless it is a skip), and this now returns a clear label to the user that shows the expectation and the score.\
Also, it is now possible to either define a dict with explicit manual_result, manual_score, manual_reason or stick to a one-liner pass or fail (score will be set to 1.0 if pass and to 0.0 if fail if not set).

- [MAJOR] Change model arg to model_name and add more kwargs\
This now supports passing any kwargs from agent.\
Also, there is a BaseLLM now with a minimum config, and there is also settings for safety/censorship.

- Expand imports for llmprovider\
These allow for more comfortable modeling.

- Init git-cliff=2.9.1 with keepachangelog\
git-cliff --init keepachangelog

- Modify cliff template to support my commits\
This seems to work, at least partially.


[unreleased]: https://github.com/pvzhelnov/gerpa/commits/main 

<!-- generated by git-cliff -->
