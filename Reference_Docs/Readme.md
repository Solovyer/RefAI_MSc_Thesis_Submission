# Reference_Docs

This folder contains the ABR, vocabulary, period, place, and custom alias
resources used by RefAI. The runtime Excel resources are stored in
`Excel_Docs` and support traceable normalisation.

The pipeline especially relies on `Excel_Docs/Refai_Custom_Aliases.xlsx`. It stores term mappings, ambiguity rules, ABR governance, and other editable domain decisions outside the procedural Python code where possible.

Do not move or rename runtime Excel resources without also updating the pipeline configuration. Academic and technical publications are not read by the pipeline. For the GitHub upload, literature PDFs are intentionally excluded; the accountability document contains the relevant citations and discussion.

