```python
# bibmerge.py v0.1.2
# Version Summary:
# 0.1.1: This is basically the first committed version of the script taken from the Final Paper Outline repo, but incremented the patch number for adding these comment lines referring to the script version. Needless to say that SemVer is used.
# 0.1.2: Fixes the name of the bib file for the .bibpaths case (makes it .bib instead of .bibpaths.bib).
# Bumps from 2025-07-25: Some changes suggested by various LLMs (Claude Sonnet 4, Qwen3-Coder, Gemini Diffusion, ChatGPT-4o) but with a handful of manual coding also.
# 0.1.3 (2025-07-25): fixes the bug when script is launched from another dir, output bib is saved in script dir instead of same dir as bibpaths.
# 0.1.4 (2025-07-25): adds support for commenting out lines in bibpaths by adding hash to start of line.
# 0.1.5 (2025-07-25): adds support for ignoring subdirs while including parent dir; also a major refactor of code, hopefully without lost functionality - works fine after a few runs.

# Credit: ChatGPT September 25 Version (https://help.openai.com/en/articles/6825453-chatgpt-release-notes). Free Research Preview.

# Initial prompt:
# you will write a script in python. it takes from the command line argument a path to a file named *.bibpaths, which is a text file containing a list of directory paths separated by new lines (it can also contain paths to *.bib files). if no arguments are specified, the script presumes the default file is located in the same dir where the script is located and is called ".bibpaths" (i.e., has an empty filename - only the extension). then the script goes recursively over any directories listed in the *.bibpaths file, as well as any their nested directories, looking for *.bib files. any paths to *.bib files listed in the *.bibpaths are also included. then, all of the *.bib files identified are merged into one *.bib file, separated by two new lines, with one new line at the end of file, and saved in the same dir where the script file is located, taking as name the name of the *.bibpaths file (if no argument was specified and a default local .bibpaths file was presumed, then the output file won't have a name, either, only the extension). when reading the contents of the *.bib files identified for merging, you must ensure that utf-8 encoding is used. you also need to make sure that if the dir where the script is located is present among the *.bibpaths entries, any *.bib files from it are processed except for the bib file with the same name as bibpaths (or, in the case of default, the .bib file without the filename).

# Final output, with modifications:

# Here's a Python script that accomplishes the task you described:

import os
import sys

class BibMerger():
    """
    Example usage:
    ```
    bibpaths = ["/path/to/dir1", "/path/to/dir2", "!/path/to/dir2/subdir_exclude", "/path/to/file.bib", "#/path/to/commented_out"]
    bibpaths_file = "/path/to/.bibpaths"
    with open(bibpaths_file, "w") as f:
        for path in bibpaths:
            f.write(path + "\n")
    bib_merger = BibMerger(bibpaths_file)
    bib_files = bib_merger.get_bib_files_from_paths()
    print(f"Found {len(bib_files)} *.bib files.")
    output_file = "/path/to/some_output_file_name.bib"
    bib_merger.merge_bib_files(bib_files, output_file)
    ```
    """

    def __init__(self, bibpaths_file: str) -> list[str]:
        self.bibpaths_file = bibpaths_file
        self.bibpaths = self._read_bibpaths_file(bibpaths_file)
        self.bib_files = []

    def _infer_default_bib_file(self) -> str:
        bibpaths_file = self.bibpaths_file
        default_bib_file_name = os.path.splitext(os.path.basename(bibpaths_file))[0]
        default_bib_file_name = '' if default_bib_file_name.startswith('.') else default_bib_file_name
        default_bib_file_dir = os.path.dirname(bibpaths_file)
        return os.path.join(default_bib_file_dir, f"{default_bib_file_name}.bib")

    def _read_bibpaths_file(self, bibpaths_file: str) -> list[str]:
        bibpaths = []
        if os.path.isfile(bibpaths_file):
            with open(bibpaths_file, "r") as f:
                bibpaths = [line.strip() for line in f if not line.strip().startswith('#')]
        else:
            print(f"Error: {bibpaths_file} not found. If you want to use a custom *.bibpaths file, pass the path as a script argument.")
        return bibpaths

    def _find_bib_files(self, directory: str, excluded_paths: list[str]) -> list[str]:
        bib_files = []

        # Preprocess excluded paths by removing '!' prefix
        normalized_excluded = [path[1:] if path.startswith('!') else path
                            for path in excluded_paths]

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories in-place
            dirs[:] = [d for d in dirs
                    if not any(os.path.join(root, d).startswith(excluded)
                                for excluded in normalized_excluded)]

            # Collect .bib files
            bib_files.extend(os.path.join(root, file)
                            for file in files if file.endswith('.bib'))

        return bib_files

    def get_bib_files_from_paths(self, bibpaths: list[str] | None = None) -> list[str]:
        if len(self.bib_files) > 0:
            return self.bib_files

        bibpaths = bibpaths if bibpaths else self.bibpaths

        included_paths = [path for path in bibpaths if not path.startswith('!')]
        excluded_paths = [path for path in bibpaths if path.startswith('!')]

        # Preprocess excluded paths for comparison
        normalized_excluded = [path[1:] for path in excluded_paths]

        bib_files = []
        for path in included_paths:
            # Skip if path is explicitly excluded
            if any(path.startswith(excluded) for excluded in normalized_excluded):
                continue

            if path.endswith('.bib'):
                bib_files.append(path)
            elif os.path.isdir(path):
                bib_files.extend(self._find_bib_files(path, excluded_paths))

        # Remove the default bib file from the list
        default_bib_file = self._infer_default_bib_file()
        bib_files = [file for file in bib_files if os.path.basename(file) != os.path.basename(default_bib_file)]

        self.bib_files = bib_files

        return bib_files

    def merge_bib_files(self, bib_files: list[str], output_file: str | None = None) -> str:
        if output_file is None:
            output_file = self._infer_default_bib_file()
        with open(output_file, "w", encoding="utf-8") as output:
            for bib_file in bib_files:
                with open(bib_file, "r", encoding="utf-8") as input_file:
                    output.write(input_file.read().strip("\n") + "\n\n")
        return output_file

def main():
    if len(sys.argv) > 1:
        bibpaths_file = sys.argv[1]
    else:
        bibpaths_file = ".bibpaths"

    # Collect paths to individual bib files (excluding default file)
    bib_merger = BibMerger(bibpaths_file)
    bib_files = bib_merger.get_bib_files_from_paths()
    if len(bib_files) == 0:
        print("No *.bib files found to merge.")
        return

    # Merge and dump contents of individual bib files
    try:
        output_file = bib_merger.merge_bib_files(bib_files)
        print(f"Merged {len(bib_files)} *.bib files into {output_file}.")
    except Exception as e:
        print(f"Found {len(bib_files)} *.bib files.")
        print(f"Error occurred when merging: {e}")

if __name__ == "__main__":
    main()

# Save this script to a .py file, and you can run it from the command line. Make sure you have Python installed on your system. You can run it with a specific .bibpaths file as an argument, or it will default to looking for a file named .bibpaths in the same directory as the script. It will merge all the .bib files found and save the result in a file with the same name as the .bibpaths file (or just .bib for the default).
```
