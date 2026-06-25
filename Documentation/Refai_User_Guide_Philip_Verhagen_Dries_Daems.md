# RefAI User Guide

Prepared for Philip Verhagen and Dries Daems.

RefAI extracts structured archaeological information from selected KNOB Archaeological News PDFs. It should be read as a first-pass extraction and triage tool. The local SLM/rulebook layer produces the baseline output. The optional LLM layer reviews suspicious records as a second reader.

The main output files are in `Reviewed_Outputs_01`. The files in `Llm_Review_Outputs_02` are audit files. The files `Suspect_Records.xlsx` and `Batch_Quality_Summary.xlsx` help identify records that need review.

Important caution: the pipeline is not a fully automatic archaeological interpreter. It helps find useful records and makes uncertainty visible. Ambiguous place names, OCR damage, dating evidence and ABR bucket conflicts still require human judgement.
