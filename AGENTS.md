# Agent Instructions

## Project Structure

```
wild_animal_image_classification/
  data/
    raw/                     # downloaded iWildCam images and metadata (not committed)
    processed/               
  report.md                  # methodology and results only
```


## Code Style and Complexity

### Variable and File Naming
- Name files and variables using snakecase
- Name constants using caps snakecase

### Avoid Over-Engineering
- Do not create unnecessary function wrappers
- Do not add features that weren't requested
- Prefer straightforward TensorFlow and `tf.data` pipelines over custom framework abstractions
- Keep scripts assignment-focused; avoid turning the project into a full application unless requested
- Do not use verbose output formats:
  - No caps-locked headers with `=*80` dividers (e.g., "=== HEADER ===")
  - No markdown-style headers with equals (e.g., "==Header==")
  - If separation is needed, use dashes followed by header with approximately 80-100 dashes for full screen width:
    - Format: `"-" * 80 + "\n" + "Header"`
    - Example: `"--------------------------------------------------------------------------------\nSection Title"`
- Keep code at the appropriate level for the assignment context

### Comments
- **Do not add author or date comments** unless explicitly asked to
- Write comments above major code blocks or complex lines
- Keep comments concise and relevant

## Package Management

### UV
- This project uses **UV** as the package manager
- **Never use pip, pip install, or any pip commands**
- **Never use conda or other package managers**
- To add dependencies, use: `uv add <package>`
- Target Python version is **3.13**
- To run Python scripts, use: `.venv/Scripts/python.exe <script>` or the appropriate venv path
- Do not attempt to modify package management - UV handles everything

## Report (`report.md`)

The `report.md` serves as the project report draft for code-related content only. Do **not** write cover pages, table of contents, etc., those are handled outside of code. Only write the methodology and results.

### Report Guidelines
- Every decision must be justified (dataset filtering, image resizing, augmentation, model/architecture choices, feature selection, etc.).
- Add "Insert example.jpg here" when appropriate for plots/artifacts produced by the code.

## Modeling Guidance

- Use TensorFlow/Keras for image classification unless explicitly asked otherwise
- Prefer transfer learning baselines before training a custom CNN from scratch
- Use validation-aware training with checkpoints and early stopping for long-running experiments
- Track dataset assumptions clearly, especially class imbalance, missing files, and metadata filtering

## File Operations

### Be Cautious with File Generation
- **Do not run code that downloads or creates multiple files** without explicit permission
- Always review what files will be created before running code
- Ask before running scripts that perform file I/O operations if uncertain

### File Paths for I/O
- Use the location of the current file when working with I/O operations
- **Do not use `os.getcwd()`** as it causes issues depending on how the file is run
- Use `os.path.dirname(os.path.abspath(__file__))` to get the script's directory
- For notebook compatibility, read from `content/data` in Colab and from `data` at the project root locally
