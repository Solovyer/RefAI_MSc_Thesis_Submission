# RefAI: Start Here

This is the complete Windows installation and startup guide for RefAI. You do
not need to open the Python code before using the startup menu.

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

Keep the folder structure intact. The `Start_Here`, `Code`, `Pdf`,
`Reference_Docs`, `Templates` and `Output` folders must remain together.

### Step 2: Install Python 3.12

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
   files. Run this after option 1.
3. **Open JupyterLab**: opens the development notebook for interactive use.
4. **Run SLM-only**: processes the seven configured test PDFs locally. This
   does not require an Anthropic API key.
5. **Run SLM + LLM**: runs local extraction followed by Anthropic review. An
   Anthropic API key and internet connection are required.
6. **Open this guide**: opens `1_Start_Here.md`.
0. **Exit**: closes the startup menu.

## Recommended First-Time Sequence

1. Right-click `2_Start_Refai.ps1` and select `Run with PowerShell`.
2. Choose option `1` and wait for installation to finish. This may take time.
3. Choose option `2`. A missing Anthropic key is only a warning for SLM-only
   use; the important result is `0 failure(s)`.
4. Choose option `4` for the first test run without LLM costs.
5. Check the newest dated folder inside `Output`.

For later sessions, right-click `2_Start_Refai.ps1` again and normally choose
option `4` or `5`. Setup does not need to be repeated unless the environment
is missing or damaged.

## JupyterLab Use

Choose menu option `3`. In JupyterLab, open:

```text
Code/Refai_V7.ipynb
```

Select the `RefAI` kernel if Jupyter asks which kernel to use. Execute notebook
cells from top to bottom. The notebook contains the tested seven-PDF batch
configuration.

## Where Results Appear

Every run creates a dated folder inside `Output`. The most important folders
inside a run are:

- `Reviewed_Outputs_01`: clean reviewed CSV and Excel output;
- `Llm_Review_Outputs_02`: optional LLM review output;
- `Raw_Outputs_03`: technical full output;
- `Logs_04`: processing and decision logs;
- `Quality_Control_00`: suspect records and batch-quality summaries.

Use the newest date-time folder when reviewing a new run.

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

### The First Model Run Is Slow

When ArcheoBERTje is enabled, Transformers downloads
`alexbrandsen/ArcheoBERTje` during the first run. OCR and local model processing
can also be CPU intensive. Later runs normally reuse the local model cache.

## API Key Safety

An Anthropic API key is only needed for menu option `5`. The startup script
asks for the key for the current PowerShell process and does not write it into
the package. Never add API keys to the notebook, source code or Git repository.
