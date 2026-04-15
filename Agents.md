# Agents Guide

## Code 
- Place code in a subdirectory exp/<name_of_experiment> . In this subdirectory place a Readme.md describing the CLI, the intent of the experiment, the pipeline and methods employed. Describe the outputs that are produced and how to interpret the outputs. The main routine should be run.py, related helper code can be in a seperate python file in this directory if that helps keep things modular and organized.  
- Keep the Readme.md up to date for each of the experiments as code in the exp/ subdirectories is modified. 

## Intent
- Keep the code concise and readable over defensive generality.
- Rely on guarantees from earlier pipeline stages rather than repeating checks.
- Minimize type checking; assume inputs follow how the repo calls the functions.
- Remove dead/unused code; keep docs aligned with reality.
- when writing a program always include feedback to the user (INFO print statements, progress bars)
so that it is clear what is going on, what files are written and that execution time is tracked.

## Data Integrity Assumptions
- Validate at the boundary where data is created or written.
- Downstream stages should trust upstream formats and shapes.
- Avoid duplicating shape, key, or type checks unless they prevent a real failure.

## Type Checking
- Prefer direct usage over defensive casting or validation.
- Only coerce types when a specific call site requires it.
- Do not broaden APIs to handle hypothetical cases.

## Code Hygiene
- Delete unused functions, flags, and blocks that are not called.
- Keep docstrings and comments up to date when code changes.
- Update `readme.md` when the pipeline order or CLI interfaces change.
- Always include progress bars, or CLI messages to ensure the user knows what the program is doing
- Measure timing of key functions so that a post-hoc analysis can reveal what functions are taking time, and how the program scales to larger datasets


## Documentation
- Document the pipeline in the order commands are run.
- Prefer short, concrete descriptions over exhaustive edge cases.

## Data Root Convention
- Commands should accept a `--data_root` CLI flag to define where data is found.
- Resolution order:
  1) `--data_root` CLI flag  
  2) `./data` (repo-relative)



## Analysis Outputs
- All analysis artifacts (images, figures,`.csv`,`.txt`, `.json`, and any numpy files when appropriate) should be written under a subdirectory of `results/`.
- Always generate a figure (`.png`) when producing outputs. The figure should start with a concise title that explains what the data shows (a conclusion). All axes need to be clearly labeled with units. A meaningful colourmap should be used with a key. Where appropriate separate 0 values with a separate colour. Place multiple graphs on a figure where possible aligning x axes or y axes where appropriate so that comparisons can easily be made.
- When appropriate, include a `README.md` (or another `.md`) in that subdirectory describing:
  - what was run
  - a summary of files present
  - conclusions from the analysis
  - suggested next steps aligned with repository goals
  - the steps used in processing the data for analysis
  - the README.md should be comprehensive
  - make sure that the text of the title of the plot wraps so that it does not overlap any adjacent plots or text. Text overlapping is un-readable.
  - ensure that inside a figure  the x-axes and y-axes are aligned with each other on different plots where appropriate

- Notes:
  - `.pt` files are modeling data, not analysis outputs.
## Packages
- Programs are run with uv run, when adding python packages update the pyproject.toml file.
- check to see if packages are installed by consulting the pyproject.toml file.
