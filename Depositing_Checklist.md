# RefAI Depositing Checklist

## Include

- Runnable pipeline script and notebook.
- The numbered installation and startup files in `Start_Here/`.
- Reference documents and custom alias/rulebook workbooks.
- Selected seven test PDFs used for evaluation.
- Output template used by the current run.
- Example output run with reviewed outputs, LLM review outputs and quality-control files.
- Gold-standard files and evaluation workbooks.
- Accountability document and user guide.

## Exclude

- `API.txt` and all other files containing private credentials.
- Any hardcoded API keys or private tokens.
- Superseded root-level startup scripts or setup guides.
- Full historical PDF corpus unless the repository or deposit explicitly supports large files and the reuse rights are clear.
- Temporary render folders, old debug previews and private local caches.

## Before Upload

- Run a secret scan.
- Confirm that `ANTHROPIC_API_KEY` is only mentioned as a temporary environment variable or secure prompt.
- Confirm that `Start_Here/1_Start_Here.md` and all eight numbered startup files are present.
- Confirm that the selected test PDFs are present.
- Confirm that `Refai_Custom_Aliases.xlsx` is present.
- Confirm that the accountability document and user guide are present.
- Run `Start_Here/5_Check_Refai_Installation.ps1`.
