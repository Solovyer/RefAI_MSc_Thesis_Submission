# RefAI Deposition Package

This package contains the deposited version of RefAI, a hybrid extraction and
review pipeline developed for Dutch archaeological news bulletins published by
the KNOB (Koninklijke Nederlandse Oudheidkundige Bond).

The repository root is the folder `RefAI_Deposition_Package`. RefAI resolves
its working paths from this root at runtime, so the package can be cloned or
moved without editing a user-specific absolute path.

The package includes the runnable pipeline, reference documents, selected test
PDFs, evaluation material, example outputs and project documentation. The
selected test set supports reproducibility of the thesis evaluation; the full
historical PDF corpus is not included.

## Start Here

New users should first open the numbered startup guide:

**[Open the RefAI Start Here guide](Start_Here/1_Start_Here.md)**

The `Start_Here` folder is the single recommended entry point for installing
and running RefAI. Its numbered files explain the intended order and cover
PowerShell, dependency installation, installation checks, JupyterLab,
SLM-only processing, optional Anthropic LLM review and troubleshooting.

The shortest startup route is:

1. Open the `Start_Here` folder.
2. Right-click an empty area and select **Open in Terminal**.
3. Run the following commands one at a time:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\2_Start_Refai.ps1
```

The execution-policy change applies only to the current PowerShell window.
Do not start with files in `Code/` unless you deliberately want to inspect or
manually operate the notebook or Python pipeline.

## Main Package Folders

- `Start_Here/`: numbered installation and startup files.
- `Code/`: the RefAI notebook and runnable Python pipeline.
- `Pdf/`: the selected test PDFs used by the configured batch.
- `Reference_Docs/`: ABR, dating, vocabulary and custom rulebook resources.
- `Templates/`: the active output-column template used by the pipeline.
- Literature PDFs are not included in the GitHub upload package. The literature is discussed and cited in the accountability document; any full-text PDFs should only be shared through the Canvas/internal submission package when redistribution is allowed.
- `Output/`: example results and the destination for new runs.
- `Gold_Standard/` and `Evaluation/`: thesis evaluation material.
- `Documentation/`: accountability and user documentation.

## API Key and Privacy

No private Anthropic API key is included. SLM-only processing does not require
an API key. When LLM review is selected, the startup script asks the user to
enter an Anthropic API key for the current terminal session. The key must not
be written into the notebook, source code, documentation or deposited files.

For detailed instructions, always use the numbered guide in `Start_Here`
rather than older commands copied from previous development versions.

## Licence and Citation

See LICENSE for reuse conditions and CITATION.cff for citation metadata. Third-party sources and PDFs remain subject to their own rights and licences.


