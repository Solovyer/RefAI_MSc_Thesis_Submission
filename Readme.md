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

There is one main Windows entry point:

**`Start_Here/2_Start_Refai.ps1`**

After extracting or cloning the package, open `Start_Here`, right-click
`2_Start_Refai.ps1`, and select **Run with PowerShell**. Use this same file
every time you want to install, check or run RefAI. On first use, choose menu
option `1` and then option `2`. For later runs, normally choose option `4`
(SLM-only) or option `5` (SLM + LLM).

Before first use, install **64-bit Python 3.12**. Python 3.13 and 3.14 are not
currently compatible with the OCR dependency. The complete numbered guide is:

**[Open the detailed RefAI Start Here guide](Start_Here/1_Start_Here.md)**

If Windows blocks the right-click method, open a terminal in `Start_Here` and
run these commands once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\*.ps1
.\2_Start_Refai.ps1
```

Do not start with files in `Code/` unless you deliberately want to inspect the
notebook or Python pipeline.

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


