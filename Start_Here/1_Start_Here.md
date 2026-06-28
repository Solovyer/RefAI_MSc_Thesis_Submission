# RefAI: Start Here

This is the complete Windows installation and startup guide for RefAI. You do
not need to open the Python code before using the startup menu.

Official repository:
<https://github.com/Solovyer/RefAI_MSc_Thesis_Submission>

Canonical online version of this numbered guide:
<https://github.com/Solovyer/RefAI_MSc_Thesis_Submission/blob/main/Start_Here/1_Start_Here.md>

**Use the current `main` version of the complete repository. Do not combine
scripts, requirements or reference files from different downloads or older
copies, because the components are tested as one package.**

## Which File Should I Open?

There are two important files in this folder:

- `1_Start_Here.md` is the written instruction manual you are reading now.
- `2_Start_Refai.ps1` is the file that starts RefAI.

**Every time you want to install, check or run RefAI, return to this folder,
right-click `2_Start_Refai.ps1`, and select `Run with PowerShell`.**

The numbered files `3` to `8` are used by the startup menu. New users normally
do not need to open them separately.

## Before the First Installation

### Step 1: Download and Extract the Package

Download or clone the complete repository. If you downloaded a ZIP file:

1. Save the ZIP file locally.
2. Right-click the ZIP file and select `Properties`.
3. If an `Unblock` checkbox is shown, select it and click `Apply`.
4. Extract the complete ZIP file.
5. Do not run RefAI from inside the ZIP preview.

When possible, use a short local path such as `C:\RefAI` or
`C:\Users\YourName\RefAI`. Deeply nested Downloads, OneDrive or network paths
increase the risk of Windows and Excel path-length problems.

Keep the folder structure intact. The `Start_Here`, `Code`, `Pdf`,
`Reference_Docs`, `Templates` and `Output` folders must remain together.

### Step 2: Run the RefAI Setup Even If Python Packages Are Already Installed

The RefAI setup is mandatory on a new computer or a newly downloaded copy of
the repository. **Do not skip menu options `1` and `2`, even if Python,
JupyterLab, PyTorch or the other listed requirements are already installed.**

RefAI uses its own project-local `.venv` environment. Packages installed in a
different Python environment are therefore not automatically available to the
pipeline. Menu option `1` also selects a supported Python version, installs the
tested dependency ranges and registers the `RefAI` Jupyter kernel. Menu option
`2` then checks the packages and the package-relative PDFs, templates and
reference files. Skipping these steps can cause RefAI not to start, to use the
wrong Python version or kernel, or to produce incomplete results.

It is safe to run option `1` when the requirements are already present: the
installer reuses the valid local environment and only installs or updates what
is required. Continue only after option `2` reports `0 failure(s)`. A missing
Anthropic API key is an expected warning when you only intend to use SLM-only.

### Step 3: Install Python 3.12

RefAI supports 64-bit Python 3.10, 3.11 and 3.12. **Python 3.12 is strongly
recommended. Python 3.13 and 3.14 are not currently supported** because the
OCR dependency does not provide compatible packages for those versions.

To install Python 3.12 with Windows Package Manager, open PowerShell and run:

```powershell
winget install --id Python.Python.3.12 -e
```

Alternatively, download 64-bit Python 3.12 from `python.org`. Select
`Add Python to PATH` during installation. Close and reopen PowerShell after
installing Python.

Confirm the installation:

```powershell
py -3.12 --version
```

The result should begin with `Python 3.12`.

## Start RefAI

### Normal Method

1. Open the extracted `Start_Here` folder.
2. Right-click `2_Start_Refai.ps1`.
3. Select `Run with PowerShell`.
4. Choose an option from the menu.

Use this same method whenever you want to start RefAI again. Closing the
PowerShell window does not remove RefAI or its installed environment.

### If Windows Blocks the Script

This normally only needs to be solved once. Open the `Start_Here` folder,
right-click an empty area, select `Open in Terminal`, and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\*.ps1
.\2_Start_Refai.ps1
```

The execution-policy change applies only to that PowerShell window. After the
files have been unblocked, the normal right-click method should work.

## What Do the Menu Options Mean?

1. **Install or repair RefAI**: use this first. It creates `.venv`, downloads
   dependencies and registers the RefAI Jupyter kernel. Internet is required.
2. **Check the installation**: verifies Python packages and required project
   files. It also downloads and loads ArcheoBERTje to prove that the local model
   is usable before extraction begins. The first check can therefore take time.
   Run this after option 1.
3. **Open JupyterLab**: opens the development notebook for interactive use.
4. **Run SLM-only**: processes the seven configured test PDFs locally. This
   does not require an Anthropic API key.
5. **Run SLM + LLM**: runs local extraction followed by Anthropic review. An
   Anthropic API key and internet connection are required.
6. **Open this guide**: opens `1_Start_Here.md`.
0. **Exit**: closes the startup menu.

## Recommended First-Time Sequence

1. Right-click `2_Start_Refai.ps1` and select `Run with PowerShell`.
2. Choose option `1` and wait for installation to finish, even if the listed
   requirements are already installed elsewhere on the computer.
3. Choose option `2`. A missing Anthropic key is only a warning for SLM-only
   use; the important result is `0 failure(s)`. On the first computer this step
   may download approximately 440 MB for ArcheoBERTje.
4. Choose option `4` for the first test run without LLM costs.
5. Wait until PowerShell reports that the run has finished, then check the
   newly created date-time folder inside `Output`.

For every later session, return to `Start_Here`, right-click
`2_Start_Refai.ps1`, select `Run with PowerShell`, and then choose the required
menu option. Setup normally does not need to be repeated after the local
environment has passed option `2`, unless the environment is missing, damaged
or moved to a different computer.

## JupyterLab Use

Choose menu option `3`. In JupyterLab, open:

```text
Code/Refai_V7.ipynb
```

Select the `RefAI` kernel if Jupyter asks which kernel to use. Execute notebook
cells from top to bottom. The notebook contains the tested seven-PDF batch
configuration.

## Where Results Appear

Every run creates a folder such as `Output\Run_YYYYMMDD_HHMMSS`. **Always open
the newest completed run folder created by the run you just performed, and
select the result according to the run type:**

| Run type | Open this folder first | What to inspect |
| --- | --- | --- |
| SLM-only, menu option `4` | `Reviewed_Outputs_01` | The `.xlsx` or matching `.csv` files are the main local SLM/rulebook results. `Llm_Review_Outputs_02` is intentionally empty or not relevant for this run. |
| SLM + LLM, menu option `5` | `Llm_Review_Outputs_02` | These files contain the LLM-reviewed result and audit information. Use `Reviewed_Outputs_01` as the pre-LLM baseline when comparing what the LLM changed. |
| Either run type | `Quality_Control_00` | `suspect_records` and `batch_quality_summary` contain review and batch-quality information. |

The remaining folders are supporting outputs, not the best starting point for
checking whether extraction worked:

- `Raw_Outputs_03` contains the larger technical export;
- `Logs_04` contains processing and decision logs for troubleshooting.

New runs use compact folder and file names so that Excel can open the results
even when the repository was extracted inside a longer Downloads path.
If an Excel file cannot be opened, inspect the matching `.csv` file first.
Do not assess a new run from an older date-time folder: it can contain output
created with a different configuration or an earlier version of the pipeline.

### How to Recognise a Valid or Failed Run

- `RUN_COMPLETED_SUCCESSFULLY.txt` exists only after the complete batch and
  quality-control stage have finished. It records the mode, total record count,
  output path and every completed PDF as `1/7`, `2/7`, and so on.
- `RUN_INCOMPLETE.txt` means the folder must **not** be treated as valid output.
  It records how many PDFs were completed and which file or post-processing
  stage failed.
- `Logs_04\Run_Diagnostics.json` contains Python and package versions, input and
  installation-check status, ArcheoBERTje availability, completed files and the
  relevant traceback when a run fails.
- `Output\Installation_Check_Latest.txt` contains the most recent result from
  menu option `2`.

PowerShell also prints a message after each file, for example:

```text
FILE_COMPLETED_SUCCESSFULLY: Test_1_1958_P284_287.pdf (1/7 completed; 12 records)
```

The final message must say `RUN_COMPLETED_SUCCESSFULLY`. If it does not, inspect
`RUN_INCOMPLETE.txt` and `Logs_04\Run_Diagnostics.json` instead of reviewing the
generated spreadsheets.

## Problems Observed During Installation Tests

### The PowerShell Window Immediately Disappears

Do not double-click the script. Right-click `2_Start_Refai.ps1` and select
`Run with PowerShell`. The startup menu now catches errors and keeps them
visible. If Windows still closes the window, use the terminal method above.

### Python Was Not Found

Install 64-bit Python 3.12, close PowerShell, reopen it and run:

```powershell
py -3.12 --version
```

If Windows opens the Microsoft Store instead, Python is not correctly
installed or the Windows app-execution alias is interfering. Installing the
official Python 3.12 package normally resolves this.

### RapidOCR Has No Matching Distribution

If the error mentions `cp313`, `cp314`, Python 3.13, Python 3.14 or
`rapidocr-onnxruntime`, the virtual environment was created with an unsupported
Python version. The updated setup detects this and recreates `.venv` with a
supported Python version. Make sure `py -3.12 --version` works, then run menu
option `1` again.

If manual cleanup is ever required, run this from `Start_Here`:

```powershell
Remove-Item "..\.venv" -Recurse -Force
```

Then start `2_Start_Refai.ps1` again and choose option `1`.

### PyPI Connection Was Reset or Shows `versions: none`

This is normally a temporary internet, firewall, VPN or antivirus problem, not
a missing Python package. The setup automatically uses extended retries. Run
option `1` again. If the problem continues, test:

```powershell
Test-NetConnection pypi.org -Port 443
```

If `TcpTestSucceeded` is `False`, try another trusted network, such as a mobile
hotspot. Do not permanently disable security software merely to install RefAI.

### The Virtual Environment Is Missing

Right-click `2_Start_Refai.ps1`, choose `Run with PowerShell`, and select menu
option `1`.

### JupyterLab Does Not Open

First run menu option `2`. If the installation check reports failures, run
option `1` again. JupyterLab should only be started after installation passes.

### Excel Reports That the File Path Is Longer Than 259 Characters

This can affect outputs created by an older RefAI version with very long run
and file names. Copy the required `.xlsx` file to a short location such as
`C:\RefAI_Output\result.xlsx` and open that copy. New runs use shorter names and
should not encounter this Excel limitation.

### The First Model Run Is Slow

When ArcheoBERTje is enabled, Transformers downloads
`alexbrandsen/ArcheoBERTje` during the first installation check or model run.
OCR and local model processing can also be CPU intensive. Later checks and runs
normally reuse the local model cache. Within one multi-PDF batch, RefAI also
reuses the loaded model and ABR embedding index after the first PDF, so later
files should no longer repeat the full model-initialisation cost.

A VPN, firewall, antivirus tool or unstable connection can interrupt downloads
from PyPI or Hugging Face. This may leave the environment or model cache
incomplete. Do not judge a nearly empty SLM result until menu option `2` passes
and the run output reports `ArcheoBERTje available: True`. If it reports
`False`, or if the download ended with an error, resolve the connection issue,
run option `1` again and repeat the SLM-only run. Use `Logs_04` for technical
diagnosis rather than treating an interrupted run as a valid extraction result.
The hardened pipeline now stops before extraction when ArcheoBERTje is
unavailable, so a model-loading failure cannot silently become an apparently
successful but nearly empty workbook.

The same safeguard applies to menu option `5`: if SLM + LLM was requested but
the Anthropic reviewer is unavailable, RefAI stops and marks the run incomplete
instead of silently returning an SLM-only result under an LLM label.

## API Key Safety

An Anthropic API key is only needed for menu option `5`. The startup script
asks for the key for the current PowerShell process and does not write it into
the package. Never add API keys to the notebook, source code or Git repository.
