# RefAI Start Here

This folder contains the recommended starting point for running RefAI on Windows.
The original notebook and pipeline code are not modified by these startup files.

## What You Need

- Windows 10 or Windows 11.
- PowerShell.
- A 64-bit Python installation. Python 3.12 is recommended.
- Internet access during the first installation and first model download.
- At least 8 GB RAM is recommended. A dedicated GPU is optional.
- An Anthropic API key only if you want to use the optional LLM review.

## Package Layout

The startup files assume that this folder is located here:

```text
RefAI_Deposition_Package/
|-- Start_Here/
|   |-- 1_Start_Here.md
|   |-- 2_Start_Refai.ps1
|   |-- 3_Requirements.txt
|   |-- 4_Setup_Refai.ps1
|   |-- 5_Check_Refai_Installation.ps1
|   |-- 6_Launch_Refai_Jupyter.ps1
|   |-- 7_Run_Refai_Slm_Only.ps1
|   `-- 8_Run_Refai_With_Llm.ps1
|-- Code/
|-- Pdf/
|-- Reference_Docs/
`-- Output/
```

Do not move `Start_Here` outside the RefAI package.

## Numbered Startup Files

Use the files in this order:

1. `1_Start_Here.md`: read this guide first.
2. `2_Start_Refai.ps1`: open the central startup menu.
3. `3_Requirements.txt`: package list used automatically during setup.
4. `4_Setup_Refai.ps1`: create the environment and install dependencies.
5. `5_Check_Refai_Installation.ps1`: verify the installation and package files.
6. `6_Launch_Refai_Jupyter.ps1`: open JupyterLab for notebook use.
7. `7_Run_Refai_Slm_Only.ps1`: run the local SLM-only pipeline.
8. `8_Run_Refai_With_Llm.ps1`: run the local pipeline with Anthropic review.

## How to Open and Run a PowerShell Script

A file ending in `.ps1` is a Windows PowerShell script. These files contain
commands that PowerShell can execute automatically.

To run a script:

1. Open the `Start_Here` folder in Windows File Explorer.
2. Right-click an empty area inside the folder.
3. Select **Open in Terminal**. On some Windows versions this option may be
   called **Open PowerShell window here**.
4. Enter the following commands one at a time and press **Enter** after each
   command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\2_Start_Refai.ps1
```

The first command temporarily permits the startup scripts to run. It applies
only to the current PowerShell window and does not permanently change the
Windows execution policy. The second command starts the RefAI menu.

Do not double-click a `.ps1` file as the main way to start RefAI. Windows may
open the file in a text editor or close the PowerShell window before an error
can be read. If you only want to inspect or edit a script, right-click it and
open it with Notepad or Visual Studio Code.

## Fastest Route

Open PowerShell in this `Start_Here` folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\2_Start_Refai.ps1
```

The menu provides options to:

1. install RefAI;
2. check the installation;
3. open RefAI in JupyterLab;
4. run the SLM-only pipeline;
5. run the SLM + LLM pipeline.

The execution-policy change applies only to the current PowerShell window.

## First Installation

Run:

```powershell
.\4_Setup_Refai.ps1
```

The setup script:

- locates Python;
- creates `.venv` in the package root;
- upgrades `pip`;
- installs `3_Requirements.txt`;
- registers a Jupyter kernel named `RefAI`;
- checks whether required folders and files exist.

The setup does not edit `Code/Refai_V7.ipynb`.

## Check the Installation

Run:

```powershell
.\5_Check_Refai_Installation.ps1
```

This checks:

- the virtual environment;
- required Python imports;
- the notebook and Python script;
- PDF, reference and output folders;
- the seven test PDFs;
- the custom alias workbook;
- the optional Anthropic API key.

Warnings about the API key are normal when you only use SLM-only mode.

## Use RefAI in JupyterLab

Run:

```powershell
.\6_Launch_Refai_Jupyter.ps1
```

JupyterLab opens from the package root. Open:

```text
Code/Refai_V7.ipynb
```

Select the `RefAI` kernel if Jupyter asks which kernel to use. Execute the
notebook cells from top to bottom. The notebook already contains the tested
seven-PDF batch configuration.

## Run Without JupyterLab

For the deterministic local pipeline:

```powershell
.\7_Run_Refai_Slm_Only.ps1
```

For local extraction followed by optional Anthropic review:

```powershell
.\8_Run_Refai_With_Llm.ps1
```

The LLM script securely asks for the key when it is not already set. The key is
stored only in the current PowerShell process and is not written to the package.

## Where Results Appear

Each run creates a dated folder inside `Output`. Important subfolders are:

- `Reviewed_Outputs_01`: clean reviewed CSV and Excel output;
- `Llm_Review_Outputs_02`: optional LLM review output;
- `Raw_Outputs_03`: technical full output;
- `Logs_04`: processing and decision logs;
- `Quality_Control_00`: suspect records and batch-quality summaries.

Use the newest date-time folder when checking a new run.

## First Model Download

When ArcheoBERTje is enabled, Transformers may download:

```text
alexbrandsen/ArcheoBERTje
```

This can take time during the first run. Later runs normally use the local
model cache.

## Common Problems

### PowerShell blocks scripts

Run this in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Python cannot be found

Install a 64-bit Python release and enable `Add Python to PATH`, then reopen
PowerShell.

### The virtual environment is missing

Run:

```powershell
.\4_Setup_Refai.ps1
```

### JupyterLab does not open

Run the setup again, or test:

```powershell
..\.venv\Scripts\python.exe -m jupyter lab
```

### PDFs or reference files are missing

Do not rename or move package folders after setup. Run:

```powershell
.\5_Check_Refai_Installation.ps1
```

### The LLM does not run

The LLM is optional. For LLM mode, obtain an Anthropic API key and use:

```powershell
.\8_Run_Refai_With_Llm.ps1
```

### OCR is slow

OCR and local model processing can be CPU intensive. Test SLM-only first and
avoid starting multiple RefAI runs simultaneously.

## Recommended Workflow

```powershell
.\4_Setup_Refai.ps1
.\5_Check_Refai_Installation.ps1
.\6_Launch_Refai_Jupyter.ps1
```

For later runs, setup normally does not need to be repeated:

```powershell
.\5_Check_Refai_Installation.ps1
.\7_Run_Refai_Slm_Only.ps1
```
