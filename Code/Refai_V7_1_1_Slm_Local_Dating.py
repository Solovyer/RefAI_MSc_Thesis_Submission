from __future__ import annotations

# =========================
# DEPOSITION NOTE
# =========================
# This copy is prepared for thesis deposition. It is intentionally conservative:
# it keeps the tested configuration and reference-file workflow, but the project
# root is no longer hardcoded to Tobias' local machine. By default, PROJECT_DIR
# resolves to the parent folder of this script's Code/ directory. If the package
# is moved, either keep the same folder structure or set REFAI_PROJECT_DIR to the
# package root. API keys are never stored here; set ANTHROPIC_API_KEY only in the
# local environment when the optional LLM second-reader layer is used.

import json, math, os, re
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pdfplumber

# =========================
# REFAI IN ONE SENTENCE
# =========================
# RefAI is a hybrid extraction pipeline for Dutch archaeological bulletin texts.
# The local pipeline (rules, lookup tables, OCR and semantic matching) performs
# the reproducible extraction first: reading text, segmenting records, detecting
# places/dating/finds/features and normalising them to ABR terms. The LLM is used
# afterwards as a second reader: not to blindly replace everything, but to review
# suspicious records, identify missing fields and apply only safe corrections.
# This keeps the workflow more transparent than a fully generative approach while
# still benefiting from LLM reasoning on difficult historical text.

try:
    import requests
except Exception:
    requests = None

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

try:
    import fitz
except Exception:
    fitz = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None


# =========================
# SETTINGS
# =========================
# Purpose:
# This block controls which PDF(s), reference files and output folders the
# pipeline uses.
#
# What it does:
# - PROJECT_DIR defaults to the deposition package root; set REFAI_PROJECT_DIR to override it.
# - PDF_NAME is the default single-document run target.
# - RUN_BATCH_PDFS can be switched to True when multiple goldsample PDFs should
#   be processed in one run.
# - BATCH_PDF_NAMES keeps the optional batch list ready, but it is not used unless
#   RUN_BATCH_PDFS is enabled.
# - Chunking, fuzzy thresholds and evidence windows control how strict the local
#   extraction layer should be.
#
# Why this is useful:
# The operational choices are centralised at the top of the script/notebook. A new PDF
# can be processed without changing the internal extraction logic. For deposition,
# keep secrets outside the code and use REFAI_PROJECT_DIR / ANTHROPIC_API_KEY as
# environment variables when needed.

def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def default_project_dir() -> Path:
    """Find the deposition package root in both script and notebook contexts."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "Reference_Docs" / "Excel_Docs").exists() and (candidate / "Pdf").exists():
            return candidate
        if candidate.name == "Code" and (candidate.parent / "Reference_Docs" / "Excel_Docs").exists():
            return candidate.parent
    return cwd


PROJECT_DIR = Path(os.getenv("REFAI_PROJECT_DIR", default_project_dir()))
REFS_DIR = PROJECT_DIR / "Reference_Docs" / "Excel_Docs"
PDF_DIR = PROJECT_DIR / "Pdf"

PDF_NAME = os.getenv("REFAI_PDF_NAME", "Test_2_1979_3_P32_38.pdf")
RUN_BATCH_PDFS = env_bool("REFAI_RUN_BATCH_PDFS", True)
BATCH_PDF_NAMES = [
    "Test_1_1958_P284_287.pdf",
    "Test_2_1979_3_P32_38.pdf",
    "Test_3_1985_1_P32_41.pdf",
    "Test_4_1966_5_P72_75.pdf",
    "Test_5_1963_P345_348.pdf",
    "Test_6_1973_4_P171_184.pdf",
    "Test_7_1992_3_4_P45_48.pdf",
]
OUTPUT_TEMPLATE = str(PROJECT_DIR / "Templates" / "Output_Template_1979.xlsx")
OUTPUT_DIR = str(PROJECT_DIR / "Output")

START_PAGE = None
END_PAGE = None
FLATTEN_LINEBREAKS = False
CHUNK_MAX_CHARS = 2800
CHUNK_OVERLAP = 120
FUZZY_THRESHOLD = 0.93
PLACE_FUZZY_THRESHOLD = 0.95
EVIDENCE_WINDOW_CHARS = 260
WRITE_EXCEL = True

# =========================
# CLAUDE / ANTHROPIC
# =========================
# Purpose:
# This block enables or disables the LLM layer and selects the Anthropic model.
#
# What it does:
# - ENABLE_LLM_STRUCTURAL_SEGMENTATION controls whether Claude may assist with
#   structural segmentation.
# - ENABLE_LLM_REVIEW lets Claude review suspicious records as a second reader.
# - The review settings define when Claude may suggest corrections and when those
#   corrections may be automatically applied to the final fields.
#
# Why this is useful:
# Claude is not used as an uncontrolled black box. The local pipeline first makes
# a reproducible extraction; the LLM then reviews difficult or uncertain cases.
# This makes the result easier to audit and explain. For deposition, never store
# a real API key in the notebook or script. Set ANTHROPIC_API_KEY as an
# environment variable only when you explicitly want to run the LLM layer.

ENABLE_LLM_STRUCTURAL_SEGMENTATION = env_bool("REFAI_ENABLE_LLM_STRUCTURAL_SEGMENTATION", True)
ENABLE_LLM_REVIEW = env_bool("REFAI_ENABLE_LLM_REVIEW", True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_API_VERSION = "2023-06-01"

LLM_SEGMENTATION_MAX_PAGES_PER_CALL = 3
LLM_SEGMENTATION_MAX_TEXT_CHARS = 12000

LLM_REVIEW_CONFIDENCE_THRESHOLD = 0.80
LLM_REVIEW_AUTO_APPLY = True
LLM_REVIEW_ONLY_SUSPICIOUS = True
LLM_REVIEW_MIN_REVIEW_FLAG = 2
LLM_REVIEW_MAX_TEXT_CHARS = 2600
LLM_REVIEW_MAX_CANDIDATES = 40

# =========================
# MODELS / OCR
# =========================
# Purpose:
# This block configures the local models: ArcheoBERTje for semantic matching and
# OCR fallback for pages with weak or missing text layers.
#
# What it does:
# - ArcheoBERTje helps map terms to ABR labels even when they are not exact lookup
#   matches.
# - OCR fallback renders PDF pages as images and rereads them when pdfplumber does
#   not extract enough text.
#
# Why this is useful:
# Historical archaeological bulletins contain OCR noise, old spelling and varied
# terminology. This local layer makes the system less dependent on perfect text
# and less dependent on exact string matching only.

ENABLE_ARCHAEOBERTJE = True
ARCHAEOBERTJE_MODEL_NAME = "alexbrandsen/ArcheoBERTje"
ARCHAEOBERTJE_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
ARCHAEOBERTJE_BATCH_SIZE = 24
ARCHAEOBERTJE_CANDIDATE_THRESHOLD = 0.87
ARCHAEOBERTJE_MAPPING_THRESHOLD = 0.90
ARCHAEOBERTJE_MAX_SPAN_TOKENS = 3
ARCHAEOBERTJE_TOP_K_PER_KIND = 4

ENABLE_OCR_FALLBACK = True
OCR_RENDER_DPI = 240
OCR_MIN_TEXT_CHARS = 20
OCR_TEXT_MIN_CHARS_FOR_PDFPLUMBER = 50
ENABLE_OCR_READABLE_PDF_CACHE = False
KEEP_RUNTIME_CACHE = False

REFAI_VERSION_NAME = "RefAi_v8_1_0_slm_layout_compound_extraction"

DEFAULT_OUTPUT_COLUMNS = [
    "Record_ID", "PDF_File", "Page_Range", "Section_ID", "Chunk_ID",
    "Place_Raw", "Place_Normalised", "Alt_Place_Name", "Province_Normalised",
    "Province_Source", "Place_Resolution_Type", "Dating_Raw", "Dating_Evidence", "Evidence_Quote",
    "Dating_Normalised_Label", "Dating_Normalised_ABR_Code", "Dating_Source",
    "Feature_Context_Raw", "Feature_Context_ABR_Label", "Feature_Context_ABR_Code",
    "Feature_Mapping_Source", "Findings_Raw", "Findings_Normal_ABR_Label",
    "Findings_ABR_Code", "Findings_Mapping_Source", "Material_Raw",
    "Material_Normalised", "Uncertain_Flag", "Segmentation_Confidence",
    "Review_Flag", "Review_Reason",
    "LLM_Segment_Source", "LLM_Segment_Notes",
    "SLM_Record_Score",
    "LLM_Reviewed", "LLM_Needs_Correction", "LLM_Correction_Type",
    "LLM_Reason", "LLM_Confidence", "LLM_Changed_Fields",
    "LLM_Place_Normalised", "LLM_Province_Normalised",
    "LLM_Dating_Label", "LLM_Dating_Code",
    "LLM_Feature_Labels", "LLM_Feature_Codes",
    "LLM_Findings_Labels", "LLM_Findings_Codes",
    "LLM_Material_Labels", "LLM_Material_Codes",
    "LLM_Auto_Applied",
    "LLM_Record_Valid", "LLM_Noise_Record", "LLM_Mixed_Context",
    "LLM_Validation_Flags", "LLM_Merge_Split_Advice", "LLM_Evidence_Check",
    "Final_Place_Normalised", "Final_Province_Normalised",
    "Final_Dating_Normalised_Label", "Final_Dating_Normalised_ABR_Code",
    "Final_Feature_Context_ABR_Label", "Final_Feature_Context_ABR_Code",
    "Final_Findings_Normal_ABR_Label", "Final_Findings_ABR_Code",
    "Final_Material_Normalised", "Final_Material_Codes",
]

CLEAN_REVIEWED_OUTPUT_COLUMNS = [
    "Record_ID", "PDF_File", "Page_Range",
    "Place_Raw", "Place_Normalised", "Alt_Place_Name", "Province_Normalised",
    "Province_Source", "Place_Resolution_Type", "Dating_Raw", "Dating_Evidence", "Evidence_Quote",
    "Dating_Normalised_Label", "Dating_Normalised_ABR_Code", "Dating_Source",
    "Feature_Context_Raw", "Feature_Context_ABR_Label", "Feature_Context_ABR_Code",
    "Feature_Mapping_Source", "Findings_Raw", "Findings_Normal_ABR_Label",
    "Findings_ABR_Code", "Findings_Mapping_Source", "Material_Raw",
    "Material_Normalised", "Uncertain_Flag", "Segmentation_Confidence",
    "Review_Flag", "Review_Num",
]

LLM_REVIEW_OUTPUT_COLUMNS = [
    "Review_Num", "Section_ID", "Chunk_ID", "Place_Resolution_Type", "Review_Reason",
    "LLM_Segment_Source", "LLM_Segment_Notes", "SLM_Record_Score",
    "LLM_Reviewed", "LLM_Needs_Correction", "LLM_Correction_Type",
    "LLM_Reason", "LLM_Confidence", "LLM_Changed_Fields",
    "LLM_Place_Normalised", "LLM_Province_Normalised",
    "LLM_Dating_Label", "LLM_Dating_Code",
    "LLM_Feature_Labels", "LLM_Feature_Codes",
    "LLM_Findings_Labels", "LLM_Findings_Codes",
    "LLM_Material_Labels", "LLM_Material_Codes",
    "LLM_Auto_Applied", "LLM_Record_Valid", "LLM_Noise_Record",
    "LLM_Mixed_Context", "LLM_Validation_Flags", "LLM_Merge_Split_Advice",
    "LLM_Evidence_Check",
    "Final_Place_Normalised", "Final_Province_Normalised",
    "Final_Dating_Normalised_Label", "Final_Dating_Normalised_ABR_Code",
    "Final_Feature_Context_ABR_Label", "Final_Feature_Context_ABR_Code",
    "Final_Findings_Normal_ABR_Label", "Final_Findings_ABR_Code",
    "Final_Material_Normalised", "Final_Material_Codes",
]

# =========================
# STATIC VOCABULARIES AND FILTERS
# =========================
# Purpose:
# These lists act as safety rails for the local extraction pipeline.
#
# What it does:
# - Province names and administrative codes are normalised for consistent place
#   resolution.
# - Reject lists block words that are often falsely interpreted as places, finds
#   or dating evidence.
# - Hint words help classify terms as feature, finding or material.
# - Dating patterns and context windows guide the local dating pass.
#
# Why this is useful:
# This improves document independence. Instead of fixing only one PDF-specific
# error, RefAI gets general protection against headings, directions, personal
# names, common nouns and OCR fragments.

NL_ADMIN1_MAP = {
    "01": "Drenthe", "1": "Drenthe", 1: "Drenthe",
    "02": "Friesland", "2": "Friesland", 2: "Friesland",
    "03": "Gelderland", "3": "Gelderland", 3: "Gelderland",
    "04": "Groningen", "4": "Groningen", 4: "Groningen",
    "05": "Limburg", "5": "Limburg", 5: "Limburg",
    "06": "Noord-Brabant", "6": "Noord-Brabant", 6: "Noord-Brabant",
    "07": "Noord-Holland", "7": "Noord-Holland", 7: "Noord-Holland",
    "09": "Utrecht", "9": "Utrecht", 9: "Utrecht",
    "10": "Zeeland", 10: "Zeeland",
    "11": "Zuid-Holland", 11: "Zuid-Holland",
    "15": "Overijssel", 15: "Overijssel",
    "16": "Flevoland", 16: "Flevoland",
}

DUTCH_PROVINCES = {
    "drenthe", "flevoland", "friesland", "gelderland", "groningen", "limburg",
    "noord-brabant", "noord holland", "noord-holland", "noordholland",
    "overijssel", "utrecht", "zeeland", "zuid-holland", "zuid holland", "zuidholland",
}

PROVINCE_CANONICAL = {
    "drenthe": "Drenthe",
    "flevoland": "Flevoland",
    "friesland": "Friesland",
    "gelderland": "Gelderland",
    "groningen": "Groningen",
    "limburg": "Limburg",
    "noord-brabant": "Noord-Brabant",
    "noord brabant": "Noord-Brabant",
    "noordholland": "Noord-Holland",
    "noord-holland": "Noord-Holland",
    "noord holland": "Noord-Holland",
    "overijssel": "Overijssel",
    "utrecht": "Utrecht",
    "zeeland": "Zeeland",
    "zuid-holland": "Zuid-Holland",
    "zuid holland": "Zuid-Holland",
    "zuidholland": "Zuid-Holland",
}

EXTRA_REGION_HEADINGS = {
    "zuiderzeegebied": "Zuiderzeegebied",
}

STOPWORDS = {
    "de", "het", "een", "eene", "en", "of", "van", "te", "tot", "in", "op", "bij", "voor", "met", "uit",
    "door", "naar", "aan", "om", "als", "dan", "maar", "onbekend", "niet", "toepassing",
    "dit", "dat", "deze", "die", "zijn", "werd", "wordt", "heeft", "hebben", "ook", "nog",
    "meer", "noord", "zuid", "oost", "west", "veen", "eind", "haag",
    "hier", "aldaar", "welke", "waarbij", "waarvan", "waarin", "daarin", "daarvan",
}

PLACE_REJECT_EXACT = {
    "een", "meer", "noord", "zuid", "oost", "west", "veen", "eind", "haag",
    "kerk", "kapel", "toren", "gebouw", "gasthuis", "stadhuis", "ringmuur",
    "kern", "spits", "baksteen", "fundering", "vlechtwerk", "berg", "watertoren",
    "drente", "friesland", "groningen", "gelderland", "overijssel", "utrecht",
    "zeeland", "limburg", "flevoland", "noord-holland", "zuid-holland", "noord-brabant",
    "nederland", "onder", "zijde", "gracht", "teerd", "binnenstad", "stuifzand",
    "lammers", "zand", "middel", "overschot", "einde", "putten", "centrum",
    "janskerk", "hofstede", "klokkenberg", "brunsting", "zuidzijde",
    "herv", "delftse t", "romeinse tijd bevindt",
}

PLACE_REJECT_PREFIXES = (
    "over vrijwel",
    "over het",
    "over de",
    "aan de",
    "in de",
    "op de",
)

PLACE_REJECT_SUBSTRINGS = {
    "hetgehele", "het gehele", "onderzochte", "vrijwel", "richting", "zuidzijde",
}

PLACE_CONTEXT_LOCATION_MARKERS = {
    "te", "bij", "in", "nabij", "onder", "rond", "om", "binnen", "plaats", "stad",
    "dorp", "gemeente", "gem", "provincie", "prov", "locatie", "terrein",
}

PLACE_CONTEXT_OBJECT_WORDS = {
    "bijl", "bijlen", "beslag", "fibula", "fibulae", "gordel", "gordelgarnituren",
    "gracht", "grachten", "greppel", "greppels", "helling", "kerkhof", "kuil", "kuilen",
    "mortel", "paal", "palen", "put", "putten", "spoor", "sporen", "steeg", "streek",
    "steen", "stenen", "waterput", "waterputten", "zand", "zijde",
}

PLACE_CONTEXT_DIRECTION_WORDS = {
    "noorden", "zuiden", "oosten", "westen", "noordzijde", "zuidzijde", "oostzijde", "westzijde",
}

PLACE_CONTEXT_AMBIGUOUS_COMMON_PLACE_NAMES = {
    # Real gazetteer names that frequently appear as ordinary Dutch words in OCR
    # text. They need explicit local support before we accept them as places.
    "begraafplaats", "gave", "hout", "houten", "huizen", "kerkhof", "mortel",
    "noorden", "noordoosthoek", "opgehoogd", "rijen", "streek", "woerd",
    "zuidoost",
}

PLACE_CONTEXT_COMMON_NOUN_PLACE_GUARD = {
    "begraafplaats", "huizen", "noordoosthoek", "woerd", "zuidoost",
    "oranje",
}

HOMONYM_PLACE_TERMS = {
    # Real Dutch place names that can also be ordinary archaeological nouns,
    # features, materials or landscape words. These must be disambiguated from
    # context instead of being accepted by a gazetteer hit alone.
    "huizen", "kerkhof", "kapel", "slot", "burg", "brug", "dam", "dijk",
    "haven", "terp", "schans", "molen", "tempel", "akker", "zand", "veen",
    "ven", "meer", "oever", "waard", "woerd", "woud", "hout", "brink",
    "esch", "donk", "kamp", "laar", "laren", "beek", "noord", "oost",
    "west", "noorden", "einde", "hoek", "haar", "hoorn", "baar", "paal",
    "plaat",
}

HOMONYM_QUARANTINE_TERMS = {
    # Token-conscious subset: only these high-risk terms get the expensive
    # homonym route. Broader homonyms stay handled by the normal place guards.
    "huizen", "kerkhof", "kapel", "slot", "dijk", "haven", "terp",
    "haar", "hoorn", "hout",
}

HOMONYM_FEATURE_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"huisplattegrond(?:en)?|huizen\s+met|(?:twee|drie|vier|vijf|zes|zeven|acht|negen|tien)\s+huizen|"
    r"resten\s+van\s+huizen|bewoning|gebouw(?:en)?|fundering(?:en)?|spoor|sporen|greppel(?:s)?|"
    r"kuil(?:en)?|waterput(?:ten)?|aardewerk|vondst(?:en)?|gevonden|aangetroffen|"
    r"kerkhof|kapel|kerk|burcht|kasteel|brug|dam|dijk|haven|terp|schans|molen|tempel|"
    r"akker|zand|veen|oever|hout|paal|palen|plaat|platen|haar|hoorn"
    r")\b",
    flags=re.I,
)

STANDALONE_SHAPE_FINDINGS = {
    "rechthoekig", "rechthoekige", "vierkant", "vierkante", "ovaal", "ovale",
    "rond", "ronde",
}

STREET_AS_ALT_SUFFIXES = (
    "straat", "weg", "laan", "steeg", "pad", "kade", "dijk", "plein",
)

SITE_PARENT_MAP = {
    # Keep the analytical place as the parent settlement, while preserving the
    # precise site/toponym in Alt_Place_Name.
    "beekmansdal": {"place": "Beekmansdal", "parent": "Nijmegen", "province": "Gelderland"},
    "huis marquette": {"place": "Huis Marquette", "parent": "Heemskerk", "province": "Noord-Holland"},
    "hunerberg": {"place": "Hunerberg", "parent": "Nijmegen", "province": "Gelderland"},
    "kelfkensbos": {"place": "Kelfkensbos", "parent": "Nijmegen", "province": "Gelderland"},
    "trajanusplein": {"place": "Trajanusplein", "parent": "Nijmegen", "province": "Gelderland"},
    "valkhof": {"place": "Valkhof", "parent": "Nijmegen", "province": "Gelderland"},
}

STREET_PARENT_MAP = {
    "erasmusschool": {"place": "Erasmusschool", "parent": "Woerden", "province": "Utrecht"},
    "grotesteeg": {"place": "Grotesteeg", "parent": "Woerden", "province": "Utrecht"},
    "grootesteeg": {"place": "Grootesteeg", "parent": "Woerden", "province": "Utrecht"},
    "havenstraat": {"place": "Havenstraat", "parent": "Woerden", "province": "Utrecht"},
    "iepstraat": {"place": "Iepstraat", "parent": "Nijmegen", "province": "Gelderland"},
    "ubbergse veldweg": {"place": "Ubbergse Veldweg", "parent": "Nijmegen", "province": "Gelderland"},
    "wagenstraat": {"place": "Wagenstraat", "parent": "Woerden", "province": "Utrecht"},
}

LANDSCAPE_WATER_PLACE_TERMS = {
    "linge", "lingedijk", "lingeoever",
}

LANDSCAPE_AS_PLACE_TERMS = {
    "dijk", "duin", "duinen", "heuvel", "heuvels", "kreek", "oever", "strandwal",
    "veen", "veenlaag", "veld", "veldweg", "zand", "zandlaag", "zandplaat",
}

EVENT_WORD_PLACE_REJECT = {
    "brand", "herstel", "onderzoek", "restauratie", "sloop", "werkzaamheden",
}

HISTORICAL_PERSON_PLACE_REJECT = {
    "agrippa", "augustus", "caesar", "claudius", "domitianus", "hadrianus",
    "nero", "neron", "tiberius", "tibr",
}

LANDSCAPE_WATER_CONTEXT_RE = re.compile(
    r"\b(?:westelijke|oostelijke|noordelijke|zuidelijke|langs|aan|bij|op|over|onder|zijde|oever|dijk|rivier|waterloop|stroom)\b",
    flags=re.I,
)

CONTEXT_WORDS_REJECT = {
    "heeft", "hebben", "is", "zijn", "werd", "worden", "nog", "ook", "met", "van", "in", "op",
    "het", "de", "een", "en", "niet", "zeer", "kan", "kunnen", "uit", "voor", "door",
    "bleek", "bestaat", "gevonden", "aangetroffen", "ontdekt", "onderzoek", "waarin",
    "waaruit", "waarvan", "bovendien", "namelijk", "echter",
}

SPAN_STOPWORDS = STOPWORDS | {
    "ca", "circa", "m", "cm", "mm", "nr", "pag", "pagina", "fig", "tabel",
    "bijlage", "onderzoek", "rapport", "locatie", "terrein", "vlak", "gedeelte",
}

EDITORIAL_PATTERNS = [
    r"\br\.o\.b\.,?\s*[A-ZÁÉÍÓÚÄËÏÖÜ][A-Za-zÁÉÍÓÚÄËÏÖÜáéíóúäëïöü\-]+",
    r"\bi\.p\.p\.,?\s*[A-ZÁÉÍÓÚÄËÏÖÜ][A-Za-zÁÉÍÓÚÄËÏÖÜáéíóúäëïöü\-]+",
    r"\brijksmuseum.*?,\s*[A-ZÁÉÍÓÚÄËÏÖÜ][A-Za-zÁÉÍÓÚÄËÏÖÜáéíóúäëïöü\-]+",
]

EDITORIAL_LOCATIONS = {
    "amersfoort", "amsterdam", "leiden", "groningen", "haarlem",
}

BAD_HEADINGS = {
    "afb", "afb.", "fig", "fig.", "opgeblazen", "totslotdankenwijdeheer", "plaat", "kaart",
    "bronzen", "romeinsch", "toelichting", "vervolg", "samenvatting",
}

NOISY_LINE_PATTERNS = [
    r"^afb\.?\s*\d+",
    r"^fig\.?\s*\d+",
    r"^\d+\s*$",
    r"^[IVXLCDM]+\s*$",
    r"^[A-Z]{2,}\s*[A-Z0-9\s\-]{0,20}$",
    r"^omschrijving\b",
    r"^Code\b",
    r"^materiaa?l:?",
]

FINDING_TRIGGERS = [
    r"\baangetroffen\b",
    r"\bgevonden\b",
    r"\bontdekt\b",
    r"\bvastgesteld\b",
    r"\bsporen van\b",
    r"\bresten van\b",
    r"\bbestaat uit\b",
]

FEATURE_HINT_WORDS = {
    "fundering", "funderingen", "greppel", "greppels", "spoor", "sporen", "nederzetting",
    "bewoningsspoor", "bewoningssporen", "haardplaats", "haardplekken", "gebouw",
    "kapel", "kerk", "ringmuur", "steunbeer", "steunberen", "droogoven", "oeverwal",
    "kasteel", "talud", "woning", "plattegrond", "voorburcht", "heuvel", "terp",
    "waterput", "afvalkuil", "kringgreppel", "ringsloot", "paalgat", "paalkuil",
    "muur", "muurresten", "begraving", "begravingen", "sloten", "grepels", "kuilen", "boerderij",
}

FINDING_HINT_WORDS = {
    "aardewerk", "vaatwerk", "kernstenen", "kernsteen", "vlechtwerk", "spits",
    "scherf", "scherven", "bijl", "mes", "ring", "spitspunt", "torens", "toren",
    "haar", "baar", "baksteen", "tufsteen", "steen", "kern", "gewei", "runderschedel",
    "botmateriaal", "houtskool", "kogelpot", "pingsdorf", "paffrath", "andenne", "brons",
    "driepoot", "onderzetter", "band", "plank", "stoel", "boek", "beslag", "hoorn", "hark",
}

MATERIAL_HINT_WORDS = {
    "aardewerk", "baksteen", "tufsteen", "hout", "houtbouw", "steen", "riet",
    "klei", "zavel", "veen", "metaal", "vlechtwerk", "gewei", "botmateriaal",
    "brons", "houtskool", "organisch", "hoorn",
}

PREFERRED_FEATURE_TERMS = {"fundering", "kerk", "kerkhof", "afvalkuil", "waterput", "greppel", "muur"}
PREFERRED_FINDING_TERMS = {"band", "bijl", "mes", "ring", "hark", "hoorn", "scherf", "scherven", "driepoot"}
PREFERRED_MATERIAL_TERMS = {"gewei", "brons", "aardewerk", "baksteen", "tufsteen", "hout", "botmateriaal"}

DATING_ANCHOR_REJECT = {
    "gracht", "greppel", "kuil", "gebouw", "constructie", "terras", "zand", "klei",
    "stuwwal", "cultuurlaag", "vloer", "breed", "afval", "paard",
}

DATE_PATTERNS = [
    r"\blate bronstijd\b",
    r"\bvroege middeleeuwen\b",
    r"\blate middeleeuwen\b",
    r"\bromeinse tijd\b",
    r"\bbronstijd\b",
    r"\bijzertijd\b",
    r"\bmiddeleeuwen\b",
    r"\bnieuwe tijd\b",
]

LOCAL_DATING_CONTEXT_WINDOW_CHARS = 420

LAYOUT_START_HEADING_PATTERNS = [
    r"\barcheologisch\s*nieuws\b",
]

LAYOUT_IGNORE_HEADING_PATTERNS = [
    r"\barcheologisch\s*nieuws\b",
    r"\bmededelingen\s+van\s+de\s+archeologische\s+instellingen\b",
    r"\bredactie\s+rijksdienst\s+voor\s+het\s+oudheidkundig\s+bodemonderzoek\b",
]

LAYOUT_STOP_HEADING_PATTERNS = [
    r"\btentoonstellingen\b",
    r"\bboekbesprekingen?\b",
    r"\baanwinsten\b",
    r"\bbibliografie\b",
    r"\bverenigingsnieuws\b",
    r"\bmuseum\s*[- ]?\s*nieuws\b",
]

CAPTION_LINE_PATTERNS = [
    r"^\s*(afb|afb\.|fig|fig\.|figuur|plaat|foto|kaart)\s*\.?\s*\d+[a-z]?\b",
    r"^\s*(afbeelding|illustratie)\s+\d+[a-z]?\b",
]

BIBLIOGRAPHY_LINE_PATTERNS = [
    r"\bbull\.\s*knob\b",
    r"\bnumaga\b",
    r"\bberichten\s+rob\b",
    r"\bjaarverslag\b",
    r"\btentoonstellingscatalogus\b",
    r"\bwest\s+friesland\s+oud\s+en\s+nieuw\b",
    r"\bde\s+bodemkartering\s+van\s+nederland\b",
    r"\buit\s+de\s+nagelaten\s+geschriften\b",
    r"\b(det\.|determinatie|red\.|eds?\.|pp\.|p\.)\b",
    r"\b(19|20)\d{2}\s*,?\s*\d{1,4}\b",
]

BIBLIOGRAPHY_CONTEXT_PATTERNS = [
    r"\b[A-Z]\.\s?[A-Z]\.\s?[A-Z]?[A-Za-zÀ-ÿ\-]+\b",
    r"\b(?:Amsterdam|Amersfoort|Leiden|'s-Gravenhage|Den Haag)\s*\([A-Z][^)]+\)",
    r"\b(?:tijdschrift|jaarboek|catalogus|publicatie|artikel|rapport)\b",
    r"\b(?:oud\s+en\s+nieuw|bodemkartering|nagelaten\s+geschriften|uitgave)\b",
    r"\b(?:Bull\.|KNOB|Numaga|ROB|I\.P\.P\.|R\.O\.B\.)\b",
    r"\b\d{4}\s*[,:]\s*\d{1,4}\b",
]

BIBLIOGRAPHIC_YEAR_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"bull\.?|knob|numaga|jaarverslag|catalogus|tentoonstellingscatalogus|"
    r"tijdschrift|jaarboek|publicatie|artikel|rapport|literatuur|bibliografie|"
    r"leids\s+jaarboekje|west\s+friesland\s+oud\s+en\s+nieuw|"
    r"amsterdam|amersfoort|leiden|'s-gravenhage|den\s+haag"
    r")\b",
    flags=re.I,
)

PERSON_NAME_PLACE_CONTEXT_PATTERN = (
    r"\b("
    r"(?:[A-Z][A-Za-zÀ-ÿ'’\-]+|[A-Z]\.)\s+(?:[a-zà-ÿ'’\-]+\s+){{0,2}}van\s+{name}"
    r"|(?:door|det(?:erminatie)?\.?|naar|volgens)\s+(?:[A-Z]\.\s*){{1,3}}[A-Z][A-Za-zÀ-ÿ'’\-]*\s+{name}"
    r"|(?:graaf|hertog|bisschop|abt)\s+van\s+{name}"
    r")\b"
)

# Rulebook fallback defaults:
# RefAI now reads ABR governance from Refai_Custom_Aliases.xlsx
# (ambiguous_term_rules, bucket_rules, generic_suppression_rules and
# auto_apply_guard_rules). The sets below remain as conservative fallbacks so
# the pipeline can still run with an older alias workbook.
SENSITIVE_FINDING_CONTEXT_TERMS = {
    "band", "boek", "brok", "haar", "kern", "rechthoekig", "ring", "segment",
    "spits", "type haps", "rijnlands",
}

FINDING_SUPPORT_CONTEXT_RE = re.compile(
    r"\b("
    r"aangetroffen|gevonden|vondst|vondsten|voorwerp|fragment|fragmenten|"
    r"bronzen|brons|ijzeren|ijzer|aardewerk|scherf|scherven|beslag|fibula|"
    r"gordel|gordelgarnituur|bekleding|versiering|object|artefact"
    r")\b",
    flags=re.I,
)

ABR_FEATURE_BUCKET_TERMS = {
    "afvalkuil", "bebouwing erf weg kerkhof", "brug", "cultuurlaag", "dijk",
    "fundering", "gebouw", "gracht", "greppel", "huisplattegrond", "inhumatiegraf",
    "kapel", "kerk", "kerkhof", "kuil", "markt", "nederzetting", "paal",
    "paalkrans", "stad", "steiger", "terras", "urnenveld", "waterput", "weg",
}

ABR_FINDING_BUCKET_TERMS = {
    "aardewerk", "andenne", "badorf", "beslag", "bijl", "bot", "bouwmateriaal",
    "dragendorff", "fibula", "gordelgarnituur", "hoorn", "kogelpot",
    "kerfsnedeversiering", "niederbieber", "paffrath", "pingsdorf", "pot",
    "scherf", "scherven", "slingerkogel", "steengoed", "terra nigra",
    "terra sigillata", "vaatwerk",
}

ABR_CERAMIC_FINDING_TERMS = {
    "aardewerk", "andenne", "badorf", "dragendorff", "kogelpot", "niederbieber",
    "paffrath", "pingsdorf", "ruwwandig", "steengoed", "terra nigra",
    "terra sigillata", "vaatwerk",
}

ABR_MATERIAL_BUCKET_TERMS = {
    "bot menselijk", "bot dierlijk", "brons", "glas", "hout houtskool",
    "ijzer", "keramiek", "klei", "metaal", "organisch", "schelp", "steen",
}

AMBIGUOUS_FINDING_REVIEW_TERMS = {
    "band", "boek", "brok", "haar", "kern", "rechthoekig", "ring", "rijnlands",
    "segment", "spits", "type haps",
}

FINDING_FALSE_CONTEXT_RE = re.compile(
    r"\b("
    r"in\s+verband|verband\s+met|kern\s+van\s+de\s+stad|stadskern|"
    r"rechthoekig(?:e)?\s+(?:stratenplan|omtrek|grond|plattegrond)|"
    r"segment\s+van\s+(?:de\s+)?(?:tekst|stad|dijk|muur)|"
    r"haar\s+karakter|haar\s+ligging|haar\s+vorm|"
    r"boek(?:bespreking|besprekingen)?"
    r")\b",
    flags=re.I,
)

DUTCH_COMPOUND_TERM_PARTS = {
    "aardewerk", "scherf", "scherven", "baksteen", "kloostermop", "kloostermoppen",
    "tufsteen", "bot", "botmateriaal", "hout", "houtskool", "paalgat", "paalkuil",
    "waterput", "afvalkuil", "fundering", "muur", "gracht", "greppel", "fibula",
    "terra sigillata", "pingsdorf", "paffrath", "kogelpot", "gewei", "brons",
    "ijzer", "beslag", "bijl", "mes", "ring",
}

RELATIVE_TIME_DATING_LABELS = {
    "recent", "recente", "laatst", "laatste", "eerst", "eerste", "vroeger",
    "vroegere", "later", "latere", "nieuw", "nieuwe", "oud", "oude",
}

RELATIVE_TIME_FALSE_POSITIVE_PATTERNS = [
    r"\b(meer|minder|zeer|vrij|tamelijk|betrekkelijk)\s+recent\b",
    r"\brecent(?:e)?\s+(?:gegraven|gevonden|aangetroffen|ontdekt|onderzoek|werkzaamheden|restauratie|publicatie|melding|waarneming|verstoring|gat|kuil|sleuf|puin|bouw|opgraving)\b",
    r"\b(?:gegraven|gevonden|aangetroffen|ontdekt|onderzocht|gerestaureerd|gemeld)\s+recent(?:e)?\b",
    r"\b(?:laatst|laatste|onlangs|kort(?:e)?\s+geleden|tijdje\s+terug|enige\s+tijd\s+terug|een\s+tijdje\s+terug)\b",
    r"\b(?:vroeger|vroegere|later|latere|eerst|eerste|nieuw|nieuwe|oud|oude)\s+(?:kerk|gebouw|gat|kuil|sleuf|muur|fase|onderzoek|opgraving|werkzaamheden|bewoning|vondst|vondsten|laag|lagen|puin|bouw)\b",
]

MODERN_EVENT_YEAR_CONTEXT_RE = re.compile(
    r"\b("
    r"brand(?:de|den)?|afgebrand|herstelwerken?|restaurat(?:ie|ies|iewerkzaamheden)?|"
    r"onderzoek(?:en)?|onderzocht|opgraving(?:en)?|opgegraven|campagne|waarneming(?:en)?|"
    r"melding(?:en)?|publicat(?:ie|ies)|gemeld|verricht|uitgevoerd|werkzaamheden|"
    r"kadastrale|kadaster|minuten|kaart(?:en)?|projecteerd|opmeting(?:en)?|"
    r"parallel|bekende|jaar\s+geleden|habets|tekening|nagelaten|geschriften"
    r")\b",
    flags=re.I,
)

PERIOD_SPECIFICITY_RANK = {
    "XXX": 0,
    "XME": 1,
    "ROM": 1,
    "BRONS": 1,
    "IJZ": 1,
    "NT": 1,
    "VME": 2,
    "LME": 2,
    "BRONSV": 2,
    "BRONSM": 2,
    "BRONSL": 2,
    "ROMV": 2,
    "ROMM": 2,
    "ROML": 2,
    "NTA": 2,
    "NTB": 2,
    "NTC": 2,
}

BROAD_GENERIC_TERM_CODES = {
    "AW",   # Aardewerk, ondetermineerbaar
    "KER",  # Keramiek as broad material bucket
    "SXX",  # Steen as broad material bucket
    "NS",   # Stad as broad feature/context bucket
    "XXX",  # Onbekend
    "---",
}


# =========================
# DATA STRUCTURES
# =========================
# Purpose:
# These dataclasses define the standard objects that move through the pipeline.
#
# What it does:
# - RunConfig stores all run settings.
# - Section represents one logical text segment from the PDF.
# - MatchResult is the shared format for dating, feature, finding and material
#   matches.
# - PlaceResolution stores place, province and uncertainty information.
# - LLMReviewResult stores the LLM second-reader response.
# - LayoutLine stores text together with page-position information.
#
# Why this is useful:
# Using shared data structures keeps the pipeline modular. New matchers or
# validators can be added as long as they fill the same fields.

@dataclass
class RunConfig:
    input_pdf: str
    output_dir: str
    output_template: Optional[str] = None
    refs_dir: Optional[str] = None
    custom_alias_file: Optional[str] = None
    gazetteer_file: Optional[str] = None
    period_file: Optional[str] = None
    abr_period_file: Optional[str] = None
    finds_file: Optional[str] = None
    material_file: Optional[str] = None
    complex_file: Optional[str] = None
    geomorphology_file: Optional[str] = None
    land_use_file: Optional[str] = None
    texture_file: Optional[str] = None
    fuzzy_threshold: float = 0.93
    place_fuzzy_threshold: float = 0.95
    chunk_max_chars: int = 2800
    chunk_overlap: int = 120
    evidence_window_chars: int = 260
    write_excel: bool = True
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    flatten_linebreaks: bool = False

    enable_llm_structural_segmentation: bool = False
    llm_segmentation_max_pages_per_call: int = 3
    llm_segmentation_max_text_chars: int = 12000

    enable_llm_review: bool = False
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    anthropic_api_version: str = "2023-06-01"
    llm_review_confidence_threshold: float = 0.80
    llm_review_auto_apply: bool = True
    llm_review_only_suspicious: bool = True
    llm_review_min_review_flag: int = 2
    llm_review_max_text_chars: int = 2600
    llm_review_max_candidates: int = 40

    enable_archaeobertje: bool = True
    archaeobertje_model_name: str = "alexbrandsen/ArcheoBERTje"
    archaeobertje_device: str = "cpu"
    archaeobertje_batch_size: int = 24
    archaeobertje_candidate_threshold: float = 0.87
    archaeobertje_mapping_threshold: float = 0.90
    archaeobertje_max_span_tokens: int = 3
    archaeobertje_top_k_per_kind: int = 4

    enable_ocr_fallback: bool = True
    ocr_render_dpi: int = 240
    ocr_min_text_chars: int = 20
    ocr_text_min_chars_for_pdfplumber: int = 50
    enable_ocr_readable_pdf_cache: bool = False
    keep_runtime_cache: bool = False
    refai_version_name: str = "RefAi_v7_1_4_period_code_year_fallback_claude_ready"


@dataclass
class Section:
    pdf_file: str
    section_id: str
    province_anchor: str
    place_heading_raw: str
    place_heading_normalised: str
    start_page: int
    end_page: int
    text: str
    segmentation_confidence: int
    segmentation_notes: List[str]
    segment_source: str = "heuristic"


@dataclass
class MatchResult:
    raw: str = ""
    label: str = ""
    abr_code: str = ""
    source: str = ""
    evidence: str = ""
    quote: str = ""


@dataclass
class PlaceResolution:
    place_raw: str = ""
    place_normalised: str = ""
    alt_place_name: str = ""
    province_normalised: str = ""
    province_source: str = ""
    resolution_type: str = ""
    fuzzy_score: Optional[float] = None
    ambiguous: bool = False


@dataclass
class LLMReviewResult:
    reviewed: bool = False
    needs_correction: bool = False
    correction_type: str = ""
    reason: str = ""
    confidence: float = 0.0
    changed_fields: str = ""
    corrected_place_normalised: str = ""
    corrected_province_normalised: str = ""
    corrected_dating_label: str = ""
    corrected_dating_code: str = ""
    corrected_feature_labels: str = ""
    corrected_feature_codes: str = ""
    corrected_findings_labels: str = ""
    corrected_findings_codes: str = ""
    corrected_material_labels: str = ""
    corrected_material_codes: str = ""
    auto_applied: int = 0
    record_valid: str = ""
    noise_record: str = ""
    mixed_context: str = ""
    validation_flags: str = ""
    merge_split_advice: str = ""
    evidence_check: str = ""


@dataclass
class LayoutLine:
    page_num: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    label: str = "body"


# =========================
# GENERAL HELPER FUNCTIONS
# =========================
# Purpose:
# This block contains reusable utilities for file handling, normalisation, text
# cleaning and evidence quotes.
#
# What it does:
# - Checks folders, PDFs and reference files.
# - Reads Excel/CSV references in a consistent way.
# - Normalises whitespace, casing, hyphens and multi-value fields.
# - Splits text into sentences/chunks and extracts compact evidence windows.
#
# Why this is useful:
# The rest of the pipeline can work with cleaner strings and stable formats,
# making matching, logging and review more reliable.

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_archaeobertje_dependencies(enabled: bool) -> None:
    if not enabled:
        return
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoTokenizer is None or AutoModel is None:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            "ArcheoBERTje is enabled, but required packages are missing: "
            + ", ".join(missing)
            + "\nInstalleer eerst:\n%pip install torch transformers huggingface_hub safetensors"
        )


def ensure_ocr_dependencies(enabled: bool) -> None:
    if not enabled:
        return
    missing = []
    if fitz is None:
        missing.append("pymupdf")
    if RapidOCR is None:
        missing.append("rapidocr-onnxruntime")
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("pillow")
    if missing:
        raise RuntimeError(
            "OCR fallback is enabled, but required packages are missing: "
            + ", ".join(missing)
            + "\nInstalleer eerst:\n%pip install pymupdf rapidocr-onnxruntime pillow numpy"
        )


def resolve_ref_file(base_dir: Path, filename: str) -> str:
    direct = base_dir / filename
    if direct.exists():
        return str(direct)
    candidates = list(base_dir.rglob(filename))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"Reference file not found: {direct}")


def resolve_optional_file(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    path = Path(path_str)
    return str(path) if path.exists() else None


def resolve_pdf_file(base_dir: Path, filename: str) -> str:
    direct = base_dir / filename
    if direct.exists():
        return str(direct)
    candidates = list(base_dir.rglob(filename))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"Input PDF not found: {direct}")


def auto_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, engine="python")


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_token(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    text = re.sub(r"[^\w\s\-/'().,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_province_token(text: str) -> str:
    return norm_token(text).replace("-", " ")


def canonical_province_name(text: str) -> str:
    key = norm_province_token(text)
    return PROVINCE_CANONICAL.get(key, safe_str(text))


def singularize_dutch(token: str) -> str:
    token = norm_token(token)
    irregular = {
        "scherven": "scherf",
        "kloostermoppen": "kloostermop",
        "moppen": "mop",
        "stenen": "steen",
        "beenderen": "been",
        "graven": "graf",
        "fibulae": "fibula",
    }
    if token in irregular:
        return irregular[token]
    if token.endswith("en") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_token(a), norm_token(b)).ratio()


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        item = norm_ws(item)
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def split_semicolon_values(text: Any) -> List[str]:
    raw = safe_str(text)
    if not raw:
        return []
    parts = re.split(r"[;|]", raw)
    return dedupe_keep_order([norm_ws(p) for p in parts if norm_ws(p)])


def join_semicolon(items: Iterable[str]) -> str:
    return "; ".join(dedupe_keep_order(items))


def normalise_multivalue_text(value: Any) -> str:
    return join_semicolon(split_semicolon_values(value))


def normalise_llm_list(value: Any) -> str:
    if isinstance(value, list):
        return join_semicolon([safe_str(v) for v in value if safe_str(v)])
    return join_semicolon(split_semicolon_values(value))


def split_sentences(text: str) -> List[str]:
    text = norm_ws(text.replace("\n", " "))
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?;:])\s+(?=[A-ZÁÉÍÓÚÄËÏÖÜ0-9])", text)
    return [norm_ws(p) for p in parts if norm_ws(p)]


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = norm_ws(text)
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            pivot = text.rfind(" ", start, end)
            if pivot > start + int(max_chars * 0.6):
                end = pivot
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def extract_evidence_quote(text: str, needle: str, window: int = 260) -> str:
    if not text or not needle:
        return ""
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return text[:window].strip()
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(needle) + window // 2)
    return text[start:end].strip()


def extract_exact_token_evidence_quote(text: str, term: str, window: int = 180) -> str:
    """
    Return a tight evidence window only when the term occurs as its own token.
    This protects sensitive ABR terms such as Ring from OCR/substrings like
    datering, richting, tering or omtuiningen.
    """
    text = safe_str(text)
    term_norm = norm_token(term)
    if not text or not term_norm:
        return ""
    pattern = re.compile(rf"(?<![\w/-]){re.escape(term_norm)}(?![\w/-])", flags=re.I)
    text_norm = norm_token(text)
    match = pattern.search(text_norm)
    if not match:
        return ""
    start = max(0, match.start() - window // 2)
    end = min(len(text_norm), match.end() + window // 2)
    return text_norm[start:end].strip()


def filter_entry_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "row_type" not in df.columns:
        return df.copy()
    series = df["row_type"].astype(str).str.strip().str.lower()
    return df[series.eq("entry")].copy()


def tokenize_simple_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’\-]*", text)


def compact_layout_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_token(text))


def is_known_province_heading(line: str) -> bool:
    line_norm = norm_token(line)
    return (line_norm in DUTCH_PROVINCES and len(line_norm.split()) <= 2) or line_norm in EXTRA_REGION_HEADINGS


def is_layout_stop_heading(line: str) -> bool:
    compact = compact_layout_text(line)
    line_norm = norm_token(line)
    return (
        any(re.search(pat, line_norm, flags=re.I) for pat in LAYOUT_STOP_HEADING_PATTERNS)
        or "tentoonstellingen" in compact
        or compact in {"museumnieuws", "museumnews"}
    )


def is_museum_news_heading(line: str) -> bool:
    return compact_layout_text(line) in {"museumnieuws", "museumnews"}


def is_compact_province_heading(line: str) -> bool:
    compact = compact_layout_text(line)
    province_compacts = {compact_layout_text(p) for p in DUTCH_PROVINCES | set(NL_ADMIN1_MAP.values())}
    region_compacts = {compact_layout_text(p) for p in EXTRA_REGION_HEADINGS}
    return compact in province_compacts | region_compacts


def heading_uppercase_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zÀ-ÿ]", safe_str(text))
    if not letters:
        return 0.0
    uppercase = sum(1 for ch in letters if ch.upper() == ch and ch.lower() != ch)
    return uppercase / max(len(letters), 1)


def is_candidate_section_boundary_heading(line: LayoutLine) -> bool:
    """
    Detect possible non-target section transitions by layout shape, not by one
    hard-coded title. This intentionally stays conservative: province headings,
    running headers and known Archaeological News headings are not candidates.
    """
    text = norm_ws(line.text)
    if not text or len(text) < 8:
        return False
    if is_layout_ignore_heading(text) or is_known_province_heading(text) or is_compact_province_heading(text):
        return False
    if is_caption_line(text) or is_bibliography_line(text):
        return False
    if re.search(r"\b(?:R\.?\s*O\.?\s*B|I\.?\s*P\.?\s*P)\.?\s*,?\s*[A-Z]", text, flags=re.I):
        return False

    rel_y = line.y0 / max(line.page_height, 1.0)
    rel_width = (line.x1 - line.x0) / max(line.page_width, 1.0)
    mid_x = (line.x0 + line.x1) / 2.0
    center_offset = abs(mid_x - (line.page_width / 2.0)) / max(line.page_width, 1.0)
    token_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", text))
    upper_ratio = heading_uppercase_ratio(text)

    return (
        0.16 <= rel_y <= 0.92
        and rel_width >= 0.22
        and center_offset <= 0.18
        and token_count <= 8
        and (upper_ratio >= 0.72 or text == text.title())
    )


def classify_boundary_heading_rule(line: LayoutLine) -> Tuple[str, float, str]:
    """
    Rule-based domain relevance classifier for candidate headings. It returns a
    label, confidence and short reason. The LLM only has to judge candidates
    that are layout-plausible but not obvious.
    """
    text_norm = norm_token(line.text)
    compact = compact_layout_text(line.text)

    if is_layout_stop_heading(line.text):
        return "new_non_archaeological_section", 0.98, "known_stop_heading"
    if is_layout_ignore_heading(line.text):
        return "running_header", 0.95, "known_archaeological_heading"
    if is_known_province_heading(line.text) or is_compact_province_heading(line.text):
        return "archaeological_section", 0.95, "province_heading"

    non_target_terms = {
        "monumentenzorg", "leefmilieu", "tentoonstelling", "tentoonstellingen",
        "boekbespreking", "boekbesprekingen", "aanwinsten", "verenigingsnieuws",
        "museum", "museumnieuws",
    }
    archaeological_terms = {
        "archeologisch", "archeologie", "opgraving", "opgravingen",
        "oudheidkundig", "bodemonderzoek",
    }
    if any(term in compact for term in non_target_terms):
        return "new_non_archaeological_section", 0.92, "non_target_section_keyword"
    if any(term in text_norm for term in archaeological_terms):
        return "archaeological_section", 0.85, "archaeological_keyword"
    return "unknown_section_heading", 0.55, "layout_candidate_needs_judgement"


def is_layout_ignore_heading(line: str) -> bool:
    line_norm = norm_token(line)
    compact = compact_layout_text(line)
    if any(re.search(pat, line_norm, flags=re.I) for pat in LAYOUT_IGNORE_HEADING_PATTERNS):
        return True
    return compact in {"archeologischnieuws", "mededelingenvandearcheologischeinstellingeninnederland"}


def is_caption_line(line: str) -> bool:
    raw = norm_ws(line)
    if not raw:
        return False
    return any(re.search(pat, raw, flags=re.I) for pat in CAPTION_LINE_PATTERNS)


def is_bibliography_line(line: str) -> bool:
    raw = norm_ws(line)
    if not raw:
        return False
    return any(re.search(pat, raw, flags=re.I) for pat in BIBLIOGRAPHY_LINE_PATTERNS)


def is_bibliographic_context(text: str) -> bool:
    """
    Detect chunks that are mostly literature/reference material rather than
    primary archaeological observations. These should not become extraction
    records merely because they mention places, years or object terms.
    """
    text = norm_ws(text)
    if len(text) < 120:
        return False
    sentences = split_sentences(text)
    line_hits = sum(1 for line in re.split(r"[\n.;]", text) if is_bibliography_line(line))
    pattern_hits = sum(1 for pat in BIBLIOGRAPHY_CONTEXT_PATTERNS for _ in re.finditer(pat, text, flags=re.I))
    primary_hits = len(re.findall(
        r"\b(aangetroffen|gevonden|opgegraven|onderzocht|sleuf|proefsleuf|grondspoor|"
        r"waterput|afvalkuil|fundering|muurrest|paalkuil|greppel|gracht|cultuurlaag|"
        r"nederzetting|bewoning|vondst|vondsten)\b",
        text,
        flags=re.I,
    ))
    if (line_hits >= 2 or pattern_hits >= 4) and primary_hits <= 1:
        return True
    if sentences and len(sentences) <= 3 and pattern_hits >= 3 and primary_hits == 0:
        return True
    if pattern_hits >= 6 and primary_hits <= 2:
        return True
    if line_hits >= 1 and pattern_hits >= 3 and primary_hits <= 1:
        return True
    return False


def has_person_name_place_context(raw: str, text: str) -> bool:
    raw_norm = norm_token(raw)
    if not raw_norm or not text:
        return False
    surface = re.escape(norm_ws(raw))
    pattern = PERSON_NAME_PLACE_CONTEXT_PATTERN.format(name=surface)
    if re.search(pattern, text):
        return True
    text_norm = norm_token(text)
    norm_pattern = rf"\b(?:graaf|hertog|bisschop|abt)\s+van\s+{re.escape(raw_norm)}\b"
    return bool(re.search(norm_pattern, text_norm))


def finding_context_supported(match: MatchResult, text: str) -> bool:
    label_norm = norm_token(match.label or match.raw)
    raw_norm = norm_token(match.raw)
    if label_norm in STANDALONE_SHAPE_FINDINGS or raw_norm in STANDALONE_SHAPE_FINDINGS:
        return False
    sensitive = label_norm in SENSITIVE_FINDING_CONTEXT_TERMS or raw_norm in SENSITIVE_FINDING_CONTEXT_TERMS
    if not sensitive:
        return True
    term = match.label or match.raw
    quote = (
        extract_exact_token_evidence_quote(match.quote, term, 180)
        or extract_exact_token_evidence_quote(match.evidence, term, 180)
        or extract_exact_token_evidence_quote(text, term, 180)
    )
    context = norm_ws(quote)
    if not context:
        return False
    if FINDING_FALSE_CONTEXT_RE.search(context):
        return False
    return bool(FINDING_SUPPORT_CONTEXT_RE.search(context))


def abr_bucket_for_match(match: MatchResult, current_bucket: str) -> str:
    values = {norm_token(match.raw), norm_token(match.label), norm_token(match.abr_code)}
    values = {v for v in values if v}
    if values & AMBIGUOUS_FINDING_REVIEW_TERMS:
        return "ambiguous_finding"
    if any(v in ABR_CERAMIC_FINDING_TERMS for v in values):
        return "finding"
    if any(v in ABR_FEATURE_BUCKET_TERMS for v in values):
        return "feature"
    if any(v in ABR_FINDING_BUCKET_TERMS for v in values):
        return "finding"
    if any(v in ABR_MATERIAL_BUCKET_TERMS for v in values):
        return "material"
    return current_bucket


def with_source(match: MatchResult, source: str) -> MatchResult:
    out = MatchResult(
        raw=match.raw,
        label=match.label,
        abr_code=match.abr_code,
        source=source,
        evidence=match.evidence,
        quote=match.quote,
    )
    return out


def quarantined_match(match: MatchResult, source: str) -> MatchResult:
    return MatchResult(
        raw=match.raw or match.label,
        label="",
        abr_code="",
        source=source,
        evidence=match.evidence,
        quote=match.quote,
    )


def is_quarantined_match(match: MatchResult) -> bool:
    source = safe_str(match.source).lower()
    return "quarantine" in source or "bucket_conflict" in source or source.endswith("_review")


def is_protected_heading(line: str) -> bool:
    return is_known_province_heading(line) or is_layout_stop_heading(line)


def line_is_noisy(line: str) -> bool:
    line = norm_ws(line)
    if not line:
        return True
    if is_protected_heading(line):
        return False
    for pat in NOISY_LINE_PATTERNS:
        if re.search(pat, line, flags=re.I):
            return True
    letters = re.findall(r"[A-Za-z]", line)
    digits = re.findall(r"\d", line)
    if len(line) < 4:
        return True
    if digits and len(digits) > len(letters):
        return True
    if re.search(r"(arcbeologique|xxxii|xxxi|xvii|plate|opgeblazen)", line, flags=re.I):
        return True
    return False


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\bNij\s+megen\b", "Nijmegen", text, flags=re.I)
    text = re.sub(r"\bNoordoosipolder\b", "Noordoostpolder", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[|¦]+", " ", text)
    text = re.sub(r"([a-z])([A-ZÁÉÍÓÚÄËÏÖÜ])", r"\1 \2", text)
    text = re.sub(r"([a-z])\.(?=[A-ZÁÉÍÓÚÄËÏÖÜ])", r"\1. ", text)
    return text.strip()


def clean_candidate_phrase(phrase: str) -> str:
    phrase = norm_ws(phrase)
    phrase = re.sub(r"^[,;:\-\*]+|[,;:\-\*]+$", "", phrase).strip()
    tokens = phrase.split()
    while tokens and norm_token(tokens[0]) in SPAN_STOPWORDS | CONTEXT_WORDS_REJECT:
        tokens = tokens[1:]
    while tokens and norm_token(tokens[-1]) in SPAN_STOPWORDS | CONTEXT_WORDS_REJECT:
        tokens = tokens[:-1]
    phrase = " ".join(tokens)
    phrase = re.sub(r"[^\w\s\-/'().,]", " ", phrase)
    return norm_ws(phrase)


def has_editorial_context(text: str) -> bool:
    for pat in EDITORIAL_PATTERNS:
        if re.search(pat, text, flags=re.I):
            return True
    return False


def is_noun_like_phrase(phrase: str) -> bool:
    phrase_norm = norm_token(phrase)
    if not phrase_norm:
        return False
    tokens = phrase_norm.split()
    if len(tokens) > 4:
        return False
    if any(tok in CONTEXT_WORDS_REJECT for tok in tokens):
        return False
    if tokens[0] in SPAN_STOPWORDS or tokens[-1] in SPAN_STOPWORDS:
        return False
    if all(len(tok) <= 3 for tok in tokens):
        return False
    return True


def candidate_spans_from_text(text: str, max_tokens: int = 3) -> List[str]:
    tokens = tokenize_simple_words(text)
    spans = []
    for i in range(len(tokens)):
        for size in range(1, max_tokens + 1):
            if i + size > len(tokens):
                break
            span = clean_candidate_phrase(" ".join(tokens[i:i + size]))
            if is_noun_like_phrase(span):
                spans.append(span)
    return dedupe_keep_order(spans)


def compound_terms_from_text(text: str) -> List[str]:
    text_norm = norm_token(text)
    compact = re.sub(r"[^a-z0-9]+", "", text_norm)
    found = []
    for term in sorted(DUTCH_COMPOUND_TERM_PARTS, key=len, reverse=True):
        term_norm = norm_token(term)
        term_compact = re.sub(r"[^a-z0-9]+", "", term_norm)
        if not term_compact or len(term_compact) < 4:
            continue
        if re.search(rf"\b{re.escape(term_norm)}\b", text_norm) or term_compact in compact:
            found.append(term)
    return dedupe_keep_order(found)


def extract_trigger_phrases(sentence: str) -> List[str]:
    sentence = norm_ws(sentence)
    found = []
    if has_editorial_context(sentence):
        return found
    for trig in FINDING_TRIGGERS:
        m = re.search(trig, sentence, flags=re.I)
        if not m:
            continue
        tail = sentence[m.end():]
        tail = re.split(r"[.;:]", tail, maxsplit=1)[0]
        parts = re.split(r",|\ben\b|\bmaar\b|\bdoch\b", tail)
        for part in parts[:3]:
            cleaned = clean_candidate_phrase(part)
            if not is_noun_like_phrase(cleaned):
                continue
            words = cleaned.split()
            if len(words) > 5:
                cleaned = " ".join(words[:5])
            found.append(cleaned)
    return dedupe_keep_order(found)


def make_output_stem(refai_version_name: str, pdf_path: Path) -> str:
    """Create a readable, Excel-safe filename stem without repeating run metadata."""
    doc_name = re.sub(r"[^\w\-]+", "_", pdf_path.stem).strip("_")
    # The run folder and run_config already preserve timestamp and version details.
    # Keeping filenames compact avoids Excel's practical 259-character path limit.
    return (doc_name or "document")[:80].rstrip("_-")


def make_run_folder_name(refai_version_name: str, run_timestamp: Optional[datetime] = None) -> str:
    """
    Give each notebook execution a short dated folder. The full pipeline version
    remains recorded in run_config, rather than being repeated in every path.
    """
    run_timestamp = run_timestamp or datetime.now()
    return f"Run_{run_timestamp.strftime('%Y%m%d_%H%M%S')}"


# =========================
# ANTHROPIC JSON CLIENT
# =========================
# Purpose:
# This wrapper handles communication with Claude/Anthropic.
#
# What it does:
# - Sends prompts to the Messages API.
# - Expects structured JSON and tries to recover the JSON object if the model
#   returns extra text.
# - Logs parsing problems so API, prompt or response-format issues can be traced.
#
# Why this is useful:
# The LLM layer is isolated from the rest of the extraction code. If the model or
# prompt changes, the entire pipeline does not need to be rewritten.

class AnthropicJSONClient:
    def __init__(self, api_key: str, model: str, api_version: str = "2023-06-01"):
        if requests is None:
            raise RuntimeError("requests is required")
        if not api_key or not model:
            raise RuntimeError("Anthropic requires api_key and model")
        self.api_key = api_key
        self.model = model
        self.api_version = api_version
        self.url = "https://api.anthropic.com/v1/messages"

    def propose(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }
        payload = {
            "model": self.model,
            "max_tokens": 1800,
            "temperature": 0.0,
            "system": "Return valid JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=180)
        if not r.ok:
            raise RuntimeError(f"Anthropic API error {r.status_code}: {r.text[:4000]}")

        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Anthropic returned non-JSON HTTP response: {r.text[:4000]}") from e

        text_blocks = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
        content_text = "\n".join(text_blocks).strip()
        json_text = self._extract_json_payload(content_text)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raw_response = json.dumps(data, ensure_ascii=False)[:4000]
            raise RuntimeError(
                "Anthropic response was not valid JSON. "
                f"Parse error: {e}. "
                f"Model text: {content_text[:4000]!r}. "
                f"Raw API response: {raw_response}"
            ) from e

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        text = safe_str(text)
        if not text:
            raise RuntimeError("Anthropic response contained no text content.")

        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if fence:
            text = fence.group(1).strip()

        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        object_start = stripped.find("{")
        object_end = stripped.rfind("}")
        array_start = stripped.find("[")
        array_end = stripped.rfind("]")

        candidates = []
        if object_start >= 0 and object_end > object_start:
            candidates.append((object_start, object_end + 1))
        if array_start >= 0 and array_end > array_start:
            candidates.append((array_start, array_end + 1))
        if candidates:
            start, end = sorted(candidates, key=lambda x: x[0])[0]
            return stripped[start:end]

        raise RuntimeError(f"Anthropic response did not contain a JSON object or array: {stripped[:4000]!r}")


# =========================
# REFERENCE STORE, CUSTOM ALIASES AND ABR RULEBOOK
# =========================
# Purpose:
# ReferenceStore loads all external knowledge: gazetteer, ABR tables, period
# tables, the custom alias workbook and the project-level ABR rulebook.
#
# What it does:
# - Builds lookup tables for places, periods, features, findings and materials.
# - Reads Refai_Custom_Aliases.xlsx for project-specific synonyms, reject terms,
#   author names, place aliases, dating aliases and ABR governance rules.
# - Reads rulebook sheets for ambiguous terms, preferred ABR buckets, generic
#   suppression and LLM auto-apply safeguards.
# - Makes labels and ABR codes available to both the local extraction layer and
#   the LLM reviewer.
#
# Why this is useful:
# The main reference files remain clean and traceable, while project-specific
# archaeological terminology and domain-policy decisions can still be added
# safely, audited in Excel and changed without rewriting extraction logic.

class ReferenceStore:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.gazetteer = self._load_gazetteer(Path(cfg.gazetteer_file))
        self.period_rules, self.period_code_map = self._load_periods(Path(cfg.period_file), Path(cfg.abr_period_file))
        self.findings_lookup = self._load_findings(Path(cfg.finds_file))
        self.feature_lookup = self._load_features(
            Path(cfg.complex_file),
            Path(cfg.geomorphology_file),
            Path(cfg.land_use_file),
            Path(cfg.texture_file),
        )
        self.material_lookup = self._load_materials(Path(cfg.material_file))
        self.custom_alias_tables = self._load_custom_alias_tables(resolve_optional_file(cfg.custom_alias_file))
        self.custom_reject_terms = self._load_custom_reject_terms()
        self.custom_author_terms = self._load_custom_author_terms()
        self.custom_place_aliases = self._load_custom_place_aliases()
        self.custom_field_type_overrides = self._load_custom_field_type_overrides()
        self.ambiguous_term_rules = self._load_ambiguous_term_rules()
        self.bucket_rules = self._load_bucket_rules()
        self.generic_suppression_rules = self._load_generic_suppression_rules()
        self.auto_apply_guard_rules = self._load_auto_apply_guard_rules()
        self.custom_dating_aliases = self._load_custom_dating_aliases()
        self.period_place_terms = self._build_period_place_terms()
        self._apply_custom_term_aliases()

    @staticmethod
    def _split_alt_names(value: Any) -> List[str]:
        raw = safe_str(value)
        if not raw:
            return []
        return [norm_ws(p) for p in re.split(r"[;,|]", raw) if norm_ws(p)]

    def _load_gazetteer(self, path: Path) -> pd.DataFrame:
        df = auto_read_table(path).copy()
        cols = list(df.columns)
        if cols[:9] == [f"Column{i}" for i in range(1, 10)]:
            df = df.rename(columns={
                "Column1": "geonameid",
                "Column2": "name",
                "Column3": "asciiname",
                "Column4": "alternatenames",
                "Column11": "admin1_code",
            })
        if "name" not in df.columns:
            raise ValueError("Gazetteer must contain a 'name' column.")
        df["name"] = df["name"].map(safe_str)
        df["asciiname"] = df.get("asciiname", "").map(safe_str)
        df["alternatenames"] = df.get("alternatenames", "").map(safe_str)
        df["admin1_code"] = df.get("admin1_code", "").map(lambda x: safe_str(x).replace(".0", ""))
        df["province_guess"] = df["admin1_code"].map(lambda x: NL_ADMIN1_MAP.get(x, ""))
        df["name_norm"] = df["name"].map(norm_token)
        df["asciiname_norm"] = df["asciiname"].map(norm_token)
        df["alt_list"] = df["alternatenames"].map(self._split_alt_names)
        return df

    def _load_periods(self, period_path: Path, abr_period_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        period_df = auto_read_table(period_path).copy()
        abr_df = auto_read_table(abr_period_path).copy()
        period_rules = []
        if "period" in period_df.columns:
            for _, row in period_df.iterrows():
                label = safe_str(row.get("period"))
                if label:
                    period_rules.append({
                        "label": label,
                        "label_norm": norm_token(label),
                        "source": safe_str(row.get("source_institute") or row.get("source")),
                    })
        code_map = {}
        if {"description_main", "abr_code"}.issubset(set(abr_df.columns)):
            for _, row in abr_df.iterrows():
                label = safe_str(row.get("description_main"))
                code = safe_str(row.get("abr_code"))
                if label and code:
                    code_map[norm_token(label)] = code
        if "onbekend" not in code_map:
            code_map["onbekend"] = "XXX"
        return period_rules, code_map

    def _build_lookup(self, df: pd.DataFrame, label_col: str, code_col: str, extra_alias_cols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        extra_alias_cols = extra_alias_cols or []
        lookup: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            label = safe_str(row.get(label_col))
            if not label:
                continue
            code = safe_str(row.get(code_col))
            aliases = {norm_token(label), singularize_dutch(label)}
            for col in extra_alias_cols:
                value = safe_str(row.get(col))
                if value:
                    aliases |= {norm_token(value), singularize_dutch(value)}
            for alias in aliases:
                if alias and alias not in lookup:
                    lookup[alias] = {"label": label, "abr_code": code, "row": row.to_dict()}
        return lookup

    def _load_findings(self, path: Path) -> Dict[str, Dict[str, Any]]:
        df = filter_entry_rows(auto_read_table(path).copy())
        return self._build_lookup(df, "description_main", "abr_code", ["description"] if "description" in df.columns else [])

    def _load_features(self, complex_path: Path, geomorphology_path: Path, land_use_path: Path, texture_path: Path) -> Dict[str, Dict[str, Any]]:
        frames = [filter_entry_rows(auto_read_table(p).copy()) for p in [complex_path, geomorphology_path, land_use_path, texture_path]]
        df = pd.concat(frames, ignore_index=True)
        return self._build_lookup(df, "description_main", "abr_code", ["description"] if "description" in df.columns else [])

    def _load_materials(self, path: Path) -> Dict[str, Dict[str, Any]]:
        df = filter_entry_rows(auto_read_table(path).copy())
        return self._build_lookup(df, "description_main", "abr_code", ["description"] if "description" in df.columns else [])

    def _load_custom_alias_tables(self, path_str: Optional[str]) -> Dict[str, pd.DataFrame]:
        if not path_str:
            return {}
        path = Path(path_str)
        if not path.exists():
            return {}
        tables: Dict[str, pd.DataFrame] = {}
        try:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                tables[norm_token(sheet).replace(" ", "_")] = pd.read_excel(path, sheet_name=sheet).fillna("")
        except Exception:
            return {}
        return tables

    @staticmethod
    def _row_enabled(row: pd.Series) -> bool:
        value = safe_str(row.get("enabled", 1)).lower()
        return value not in {"0", "false", "nee", "no", "n"}

    def _load_custom_reject_terms(self) -> set:
        out = set()
        df = self.custom_alias_tables.get("reject_terms")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            term = norm_token(row.get("term"))
            kind = norm_token(row.get("kind"))
            if term:
                out.add((kind, term))
                out.add(("", term))
        return out

    def _load_custom_author_terms(self) -> set:
        out = set()
        df = self.custom_alias_tables.get("author_names")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            name = norm_token(row.get("author_name"))
            if not name:
                continue
            out.add(name)
            parts = [p for p in re.split(r"\s+", name) if len(p) >= 4]
            for part in parts:
                out.add(part)
        return out

    def _load_custom_place_aliases(self) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        df = self.custom_alias_tables.get("place_aliases")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            alias = norm_token(row.get("alias"))
            canonical = safe_str(row.get("canonical_place"))
            if not alias or not canonical:
                continue
            out[alias] = {
                "place": canonical,
                "province": safe_str(row.get("province")),
                "source": safe_str(row.get("source")) or "custom_place_alias",
            }
        return out

    def _load_custom_field_type_overrides(self) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        df = self.custom_alias_tables.get("field_type_overrides")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            term = norm_token(row.get("term"))
            kind = norm_token(row.get("preferred_kind"))
            if term and kind in {"feature", "finding", "material"}:
                out[term] = {
                    "kind": kind,
                    "canonical_label": safe_str(row.get("canonical_label")),
                    "abr_code": safe_str(row.get("abr_code")),
                }
        return out

    def _load_ambiguous_term_rules(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        df = self.custom_alias_tables.get("ambiguous_term_rules")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            term = norm_token(row.get("term"))
            applies_to = norm_token(row.get("applies_to")) or "finding"
            if not term or applies_to not in {"finding", "feature", "material"}:
                continue
            try:
                window_chars = int(float(safe_str(row.get("context_window_chars")) or 140))
            except Exception:
                window_chars = 140
            out[term] = {
                "term": safe_str(row.get("term")),
                "applies_to": applies_to,
                "category": safe_str(row.get("category")),
                "requires_exact_token": safe_str(row.get("requires_exact_token")).lower() in {"1", "1.0", "true", "yes", "ja"},
                "support_regex": safe_str(row.get("support_regex")),
                "false_context_regex": safe_str(row.get("false_context_regex")),
                "context_window_chars": window_chars,
                "action_without_support": safe_str(row.get("action_without_support")) or "quarantine_review",
                "notes": safe_str(row.get("notes")),
            }
        return out

    def _load_bucket_rules(self) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        df = self.custom_alias_tables.get("bucket_rules")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            term = norm_token(row.get("term"))
            bucket = norm_token(row.get("preferred_bucket"))
            if not term or bucket not in {"feature", "finding", "material", "ambiguous_finding"}:
                continue
            out[term] = {
                "bucket": bucket,
                "canonical_label": safe_str(row.get("canonical_label")),
                "abr_code": safe_str(row.get("abr_code")),
                "priority": safe_str(row.get("priority")),
                "notes": safe_str(row.get("notes")),
            }
        return out

    def _load_generic_suppression_rules(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        df = self.custom_alias_tables.get("generic_suppression_rules")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            field = norm_token(row.get("field"))
            if field not in {"feature", "finding", "material", "findings"}:
                continue
            out.append({
                "field": "finding" if field == "findings" else field,
                "generic_label": safe_str(row.get("generic_label")),
                "generic_label_norm": norm_token(row.get("generic_label")),
                "generic_code": safe_str(row.get("generic_code")).upper(),
                "specific_label_terms": {
                    norm_token(v) for v in split_semicolon_values(row.get("suppress_when_label_contains"))
                },
                "specific_codes": {
                    safe_str(v).upper() for v in split_semicolon_values(row.get("suppress_when_code_in"))
                },
                "notes": safe_str(row.get("notes")),
            })
        return out

    def _load_auto_apply_guard_rules(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        out: Dict[str, Dict[str, Dict[str, str]]] = {}
        df = self.custom_alias_tables.get("auto_apply_guard_rules")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            field = norm_token(row.get("field"))
            term = norm_token(row.get("term"))
            if field == "findings":
                field = "finding"
            if not field or not term:
                continue
            out.setdefault(field, {})[term] = {
                "guard_type": safe_str(row.get("guard_type")),
                "support_regex": safe_str(row.get("support_regex")),
                "false_context_regex": safe_str(row.get("false_context_regex")),
                "action": safe_str(row.get("action")),
                "notes": safe_str(row.get("notes")),
            }
        return out

    def _load_custom_dating_aliases(self) -> List[Dict[str, str]]:
        out = []
        df = self.custom_alias_tables.get("dating_aliases")
        if df is None:
            return out
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            pattern = safe_str(row.get("pattern"))
            label = safe_str(row.get("dating_label"))
            if pattern and label:
                out.append({
                    "pattern": pattern,
                    "dating_label": label,
                    "dating_code": safe_str(row.get("dating_code")),
                    "source": safe_str(row.get("source")) or "custom_dating_alias",
                })
        return out

    def _apply_custom_term_aliases(self) -> None:
        df = self.custom_alias_tables.get("term_aliases")
        if df is None:
            return
        lookups = {
            "feature": self.feature_lookup,
            "finding": self.findings_lookup,
            "material": self.material_lookup,
        }
        for _, row in df.iterrows():
            if not self._row_enabled(row):
                continue
            alias = norm_token(row.get("alias"))
            kind = norm_token(row.get("kind"))
            label = safe_str(row.get("canonical_label"))
            if not alias or kind not in lookups or not label:
                continue
            lookup = lookups[kind]
            payload = lookup.get(norm_token(label)) or lookup.get(singularize_dutch(label))
            if not payload:
                payload = {
                    "label": label,
                    "abr_code": safe_str(row.get("abr_code")),
                    "row": {
                        "description": safe_str(row.get("notes")),
                        "description_main": label,
                        "source": safe_str(row.get("source")) or "custom_alias",
                    },
                }
            for key in {alias, singularize_dutch(alias)}:
                if key:
                    lookup[key] = payload

    def ambiguous_finding_terms(self) -> set:
        terms = {
            term for term, rule in self.ambiguous_term_rules.items()
            if rule.get("applies_to") == "finding"
        }
        return terms or set(AMBIGUOUS_FINDING_REVIEW_TERMS)

    def is_ambiguous_finding_term(self, term: Any) -> bool:
        return norm_token(term) in self.ambiguous_finding_terms()

    def _ambiguous_rule_for_match(self, match: MatchResult) -> Optional[Dict[str, Any]]:
        for value in [match.raw, match.label, match.abr_code]:
            key = norm_token(value)
            if key in self.ambiguous_term_rules and self.ambiguous_term_rules[key].get("applies_to") == "finding":
                return self.ambiguous_term_rules[key]
        return None

    def bucket_for_match(self, match: MatchResult, current_bucket: str) -> str:
        values = {norm_token(match.raw), norm_token(match.label), norm_token(match.abr_code)}
        values = {v for v in values if v}
        if values & self.ambiguous_finding_terms():
            return "ambiguous_finding"
        for value in values:
            rule = self.bucket_rules.get(value)
            if rule:
                return rule.get("bucket") or current_bucket
        return abr_bucket_for_match(match, current_bucket)

    def finding_context_supported(self, match: MatchResult, text: str) -> bool:
        rule = self._ambiguous_rule_for_match(match)
        label_norm = norm_token(match.label or match.raw)
        raw_norm = norm_token(match.raw)
        sensitive = bool(rule) or label_norm in SENSITIVE_FINDING_CONTEXT_TERMS or raw_norm in SENSITIVE_FINDING_CONTEXT_TERMS
        if label_norm in STANDALONE_SHAPE_FINDINGS or raw_norm in STANDALONE_SHAPE_FINDINGS:
            return False
        if not sensitive:
            return True

        term_norm = norm_token((rule or {}).get("term") or match.label or match.raw)
        window_chars = int((rule or {}).get("context_window_chars") or 140)
        term = safe_str((rule or {}).get("term") or match.label or match.raw)
        context = norm_ws(" ".join([
            extract_exact_token_evidence_quote(match.quote, term, max(120, window_chars)),
            extract_exact_token_evidence_quote(match.evidence, term, max(120, window_chars)),
            extract_exact_token_evidence_quote(text, term, max(120, window_chars)),
        ]))
        context_norm = norm_token(context)
        if not context_norm:
            return False

        false_pat = safe_str((rule or {}).get("false_context_regex"))
        if false_pat and re.search(false_pat, context_norm, flags=re.I):
            return False
        if not false_pat and FINDING_FALSE_CONTEXT_RE.search(context_norm):
            return False

        if bool((rule or {}).get("requires_exact_token")) and term_norm:
            exact = re.search(rf"\b{re.escape(term_norm)}\b", context_norm)
            if not exact:
                return False
            start, end = exact.span()
            context_norm = context_norm[max(0, start - window_chars):min(len(context_norm), end + window_chars)]

        support_pat = safe_str((rule or {}).get("support_regex"))
        if support_pat:
            return bool(re.search(support_pat, context_norm, flags=re.I))
        return bool(FINDING_SUPPORT_CONTEXT_RE.search(context_norm))

    def finding_term_supported_in_text(self, term: Any, text: str) -> bool:
        term_norm = norm_token(term)
        if not term_norm:
            return False
        return self.finding_context_supported(
            MatchResult(raw=safe_str(term), label=safe_str(term), source="rulebook_check"),
            text,
        )

    def suppress_generic_matches(self, field: str, matches: List[MatchResult]) -> List[MatchResult]:
        field_norm = "finding" if norm_token(field) == "findings" else norm_token(field)
        rules = [rule for rule in self.generic_suppression_rules if rule.get("field") == field_norm]
        if not rules:
            return matches
        labels_norm = [norm_token(m.label) for m in matches if m.label]
        codes = {safe_str(m.abr_code).upper() for m in matches if m.abr_code}

        def should_drop(match: MatchResult) -> bool:
            label_norm = norm_token(match.label)
            code_norm = safe_str(match.abr_code).upper()
            for rule in rules:
                is_generic = (
                    (rule.get("generic_label_norm") and label_norm == rule.get("generic_label_norm"))
                    or (rule.get("generic_code") and code_norm == rule.get("generic_code"))
                )
                if not is_generic:
                    continue
                specific_label_terms = rule.get("specific_label_terms") or set()
                specific_codes = rule.get("specific_codes") or set()
                has_specific_label = any(
                    specific and any(specific in label for label in labels_norm if label != label_norm)
                    for specific in specific_label_terms
                )
                has_specific_code = bool((codes - {code_norm}) & specific_codes)
                if has_specific_label or has_specific_code:
                    return True
            return False

        return [m for m in matches if not should_drop(m)]

    def auto_apply_guard_terms_supported(self, field: str, labels: str, row_text: str) -> bool:
        field_norm = "finding" if norm_token(field) == "findings" else norm_token(field)
        rules = self.auto_apply_guard_rules.get(field_norm, {})
        if not rules:
            return True
        for label in split_semicolon_values(labels):
            label_norm = norm_token(label)
            if label_norm in rules and not self.finding_term_supported_in_text(label, row_text):
                return False
        return True

    def is_rejected_term(self, term: Any, kind: str = "") -> bool:
        term_norm = norm_token(term)
        kind_norm = norm_token(kind)
        if not term_norm:
            return True
        if (kind_norm, term_norm) in self.custom_reject_terms or ("", term_norm) in self.custom_reject_terms:
            return True
        if term_norm in self.custom_author_terms:
            return True
        return False

    def _build_period_place_terms(self) -> set:
        """
        Build a dynamic deny-list for place names from the period reference files.
        This prevents headings such as "Late IJzertijd" from becoming locations.
        """
        out = {"recent", "protohistorie", "onbekend"}

        def add_variant(value: Any) -> None:
            label = norm_token(value)
            if not label:
                return
            out.add(label)
            tokens = label.split()
            if not tokens:
                return
            adjective_to_suffix = {
                "vroeg": "vroeg", "vroege": "vroeg",
                "midden": "midden",
                "laat": "laat", "late": "laat",
            }
            suffix_to_adjective = {
                "vroeg": "vroege",
                "midden": "midden",
                "laat": "late",
            }
            first_suffix = adjective_to_suffix.get(tokens[0])
            if first_suffix and len(tokens) > 1:
                base = " ".join(tokens[1:])
                out.add(f"{base} {first_suffix}")
                out.add(f"{suffix_to_adjective[first_suffix]} {base}")
            last_suffix = adjective_to_suffix.get(tokens[-1])
            if last_suffix and len(tokens) > 1:
                base = " ".join(tokens[:-1])
                out.add(f"{base} {last_suffix}")
                out.add(f"{suffix_to_adjective[last_suffix]} {base}")

        for rule in self.period_rules:
            add_variant(rule.get("label"))
            add_variant(rule.get("label_norm"))
        for label_norm in self.period_code_map:
            add_variant(label_norm)
        for pattern in DATE_PATTERNS:
            cleaned = re.sub(r"\\b|\(|\)|\?|:", " ", pattern)
            cleaned = re.sub(r"\\s\+|\[[-\\s]\]", " ", cleaned)
            add_variant(cleaned)
        return {term for term in out if term}

    def is_period_term(self, term: Any) -> bool:
        term_norm = norm_token(term)
        if not term_norm:
            return False
        term_norm = re.sub(r"^(mogelijk|vermoedelijk|waarschijnlijk)\s+", "", term_norm)
        term_norm = re.sub(r"\s*\([^)]+\)\s*", " ", term_norm).strip()
        if term_norm in self.period_place_terms:
            return True
        if re.fullmatch(r"(vroege|vroeg|midden|late|laat)\s+(bronstijd|ijzertijd|middeleeuwen|romeinse tijd|nieuwe tijd)", term_norm):
            return True
        if re.fullmatch(r"(bronstijd|ijzertijd|middeleeuwen|romeinse tijd|nieuwe tijd)\s+(vroeg|midden|laat)", term_norm):
            return True
        if re.fullmatch(r"([1-9]|1[0-9]|20)\s*(e|de)\s+eeuw", term_norm):
            return True
        return False

    def is_rejected_match(self, match: MatchResult, kind: str = "") -> bool:
        return (
            self.is_rejected_term(match.raw, kind)
            or (match.label and self.is_rejected_term(match.label, kind))
            or (match.abr_code and self.is_rejected_term(match.abr_code, kind))
        )

    def resolve_custom_place_alias(self, raw: str) -> Optional[PlaceResolution]:
        alias = self.custom_place_aliases.get(norm_token(raw))
        if not alias:
            return None
        return PlaceResolution(
            place_raw=raw,
            place_normalised=alias["place"],
            alt_place_name="",
            province_normalised=alias["province"],
            province_source=alias["source"],
            resolution_type="custom_place_alias",
            ambiguous=False,
        )

    def resolve_site_parent_place(self, raw: str) -> Optional[PlaceResolution]:
        site = SITE_PARENT_MAP.get(norm_token(raw))
        if not site:
            return None
        return PlaceResolution(
            place_raw=raw,
            place_normalised=site["parent"],
            alt_place_name=site["place"],
            province_normalised=site["province"],
            province_source="site_parent_context",
            resolution_type="site_parent",
            ambiguous=False,
        )

    def resolve_street_parent_place(self, raw: str) -> Optional[PlaceResolution]:
        street = STREET_PARENT_MAP.get(norm_token(raw))
        if not street:
            return None
        return PlaceResolution(
            place_raw=raw,
            place_normalised=street["parent"],
            alt_place_name=street["place"],
            province_normalised=street["province"],
            province_source="street_parent_context",
            resolution_type="street_parent",
            ambiguous=False,
        )

    def province_list(self) -> List[str]:
        return sorted(dedupe_keep_order([canonical_province_name(p) for p in DUTCH_PROVINCES]))

    def all_periods(self) -> List[Dict[str, str]]:
        out = []
        seen = set()
        for rule in self.period_rules:
            label = safe_str(rule["label"])
            code = safe_str(self.period_code_map.get(rule["label_norm"], ""))
            if label and label not in seen:
                seen.add(label)
                out.append({"label": label, "code": code})
        if "Onbekend" not in seen:
            out.append({"label": "Onbekend", "code": self.period_code_map.get(norm_token("Onbekend"), "XXX") or "XXX"})
        return out

    @staticmethod
    def _unique_label_code(lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        out = []
        seen = set()
        for payload in lookup.values():
            label = safe_str(payload.get("label"))
            code = safe_str(payload.get("abr_code"))
            if label and label not in seen:
                seen.add(label)
                out.append({"label": label, "code": code})
        out.sort(key=lambda x: x["label"])
        return out

    def all_feature_labels(self) -> List[Dict[str, str]]:
        return self._unique_label_code(self.feature_lookup)

    def all_finding_labels(self) -> List[Dict[str, str]]:
        return self._unique_label_code(self.findings_lookup)

    def all_material_labels(self) -> List[Dict[str, str]]:
        return self._unique_label_code(self.material_lookup)


# =========================
# ARCHEOBERTJE SEMANTIC MATCHING
# =========================
# Purpose:
# This helper uses ArcheoBERTje as a local semantic matcher.
#
# What it does:
# - Embeds ABR labels and aliases.
# - Embeds candidate terms found in the text.
# - Maps terms semantically to likely features, findings or materials when exact
#   or fuzzy matching is not enough.
#
# Why this is useful:
# It helps with old spelling, OCR variants and terms that are conceptually correct
# but not written exactly as they appear in the lookup tables. Because it runs
# locally, it remains reproducible and independent of the LLM.

class ArcheoBERTjeHelper:
    def __init__(self, cfg: RunConfig, refs: ReferenceStore):
        ensure_archaeobertje_dependencies(cfg.enable_archaeobertje)
        self.cfg = cfg
        self.refs = refs
        self.enabled = cfg.enable_archaeobertje
        self.available = False
        self.error_message = ""
        self.device = cfg.archaeobertje_device
        self.model_name = cfg.archaeobertje_model_name
        self.batch_size = cfg.archaeobertje_batch_size
        self.candidate_threshold = cfg.archaeobertje_candidate_threshold
        self.mapping_threshold = cfg.archaeobertje_mapping_threshold
        self.max_span_tokens = cfg.archaeobertje_max_span_tokens
        self.top_k_per_kind = cfg.archaeobertje_top_k_per_kind
        self.tokenizer = None
        self.model = None
        self.index = {}
        self._embed_cache = {}
        self._raw_mapping_cache = {}
        if not self.enabled:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._build_indexes()
            self.available = True
        except Exception as e:
            self.available = False
            self.error_message = str(e)

    def _build_indexes(self) -> None:
        self.index = {
            "feature": self._build_kind_index(self.refs.feature_lookup),
            "finding": self._build_kind_index(self.refs.findings_lookup),
            "material": self._build_kind_index(self.refs.material_lookup),
        }

    def _build_kind_index(self, lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        entries, seen = [], set()
        for alias, payload in lookup.items():
            label = safe_str(payload.get("label"))
            code = safe_str(payload.get("abr_code"))
            row = payload.get("row", {})
            description = safe_str(row.get("description") or row.get("description_main") or "")
            key = (alias, label, code)
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "alias": alias,
                "label": label,
                "abr_code": code,
                "semantic_text": norm_ws(" ".join([label, alias, description])),
            })
        if not entries:
            return {"entries": [], "embeddings": None}
        vectors = self.encode_texts([e["semantic_text"] for e in entries])
        return {"entries": entries, "embeddings": vectors}

    def encode_texts(self, texts: List[str]):
        if not texts:
            return None
        normalized_texts = [norm_ws(t) for t in texts]
        cache_keys = [f"{self.model_name}::{t}" for t in normalized_texts]
        missing = [k for k in cache_keys if k not in self._embed_cache]
        if missing:
            missing_texts = [k.split("::", 1)[1] for k in missing]
            with torch.no_grad():
                for start in range(0, len(missing_texts), self.batch_size):
                    batch = missing_texts[start:start + self.batch_size]
                    encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)
                    hidden = outputs.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    pooled = summed / counts
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()
                    for text_value, vector in zip(batch, pooled):
                        self._embed_cache[f"{self.model_name}::{text_value}"] = vector
        vectors = [self._embed_cache[k] for k in cache_keys]
        return torch.stack(vectors)

    def semantic_match(self, raw_text: str, kind: str) -> Optional[MatchResult]:
        if not self.available:
            return None
        raw_norm = norm_ws(raw_text)
        if not raw_norm or not is_noun_like_phrase(raw_norm):
            return None
        cache_key = f"{kind}::{raw_norm.lower()}"
        if cache_key in self._raw_mapping_cache:
            return self._raw_mapping_cache[cache_key]
        kind_index = self.index.get(kind)
        if not kind_index or not kind_index["entries"]:
            self._raw_mapping_cache[cache_key] = None
            return None
        raw_vec = self.encode_texts([raw_norm])
        sims = torch.matmul(raw_vec, kind_index["embeddings"].T)[0]
        score, idx = torch.max(sims, dim=0)
        score_value = float(score.item())
        entry = kind_index["entries"][int(idx.item())]
        if score_value < self.mapping_threshold:
            self._raw_mapping_cache[cache_key] = None
            return None
        result = MatchResult(
            raw=raw_text,
            label=entry["label"],
            abr_code=entry["abr_code"],
            source="archaeobertje_semantic",
            evidence=f"semantic_score={score_value:.3f}",
            quote="",
        )
        self._raw_mapping_cache[cache_key] = result
        return result


# =========================
# OCR AND PDF EXTRACTION
# =========================
# Purpose:
# This layer reads PDF pages and turns them into usable text.
#
# What it does:
# - PDFExtractor first uses pdfplumber to read the embedded text layer.
# - If that text layer is weak, OCRExtractor uses PyMuPDF and RapidOCR as fallback.
# - Layout information such as columns, headers and page positions is used to
#   improve reading order.
#
# Why this is useful:
# The source PDFs are historical and often contain columns, headers, poor OCR and
# noise. Without this layer, later modules would receive mixed or unreliable text.

class OCRExtractor:
    def __init__(self, dpi: int = 240, boundary_client: Optional[AnthropicJSONClient] = None):
        ensure_ocr_dependencies(True)
        self.dpi = dpi
        self.engine = RapidOCR()
        self.boundary_client = boundary_client
        self.boundary_events: List[Dict[str, Any]] = []

    def extract_pdf_pages(self, pdf_path: Path, start_page: Optional[int] = None, end_page: Optional[int] = None) -> List[Dict[str, Any]]:
        self.boundary_events = []
        pages = []
        doc = fitz.open(str(pdf_path))
        matrix = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        raw_pages = []
        for i in range(len(doc)):
            page_num = i + 1
            if start_page is not None and page_num < start_page:
                continue
            if end_page is not None and page_num > end_page:
                continue
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            result, _ = self.engine(img_np)
            layout_lines = []
            if result:
                for item in result:
                    if len(item) >= 2:
                        text = safe_str(item[1])
                        if not text:
                            continue
                        box = item[0]
                        try:
                            xs = [float(p[0]) for p in box]
                            ys = [float(p[1]) for p in box]
                        except Exception:
                            continue
                        layout_lines.append(LayoutLine(
                            page_num=page_num,
                            text=text,
                            x0=min(xs),
                            y0=min(ys),
                            x1=max(xs),
                            y1=max(ys),
                            page_width=float(pix.width),
                            page_height=float(pix.height),
                        ))
            raw_pages.append({"page_num": page_num, "layout_lines": layout_lines, "source": "ocr"})
        doc.close()
        pages = self._layout_filter_pages(raw_pages)
        return pages

    def _layout_filter_pages(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.boundary_events = []
        start_gate = self._find_document_start_gate(raw_pages)
        stop_gate = self._find_document_stop_gate(raw_pages)
        pages = []
        for raw_page in raw_pages:
            page_num = raw_page["page_num"]
            lines = raw_page["layout_lines"]
            kept = []
            dropped = []
            captions = []
            for line in lines:
                keep, label = self._classify_line(line, start_gate, stop_gate)
                line.label = label
                if keep:
                    kept.append(line)
                else:
                    dropped.append(line)
                    if label == "caption":
                        captions.append(line)
            ordered = self._order_lines_by_column(kept)
            text_lines = [line.text for line in ordered if line.text and not line_is_noisy(line.text)]
            pages.append({
                "page_num": page_num,
                "text": clean_ocr_text("\n".join(text_lines).strip()),
                "source": "ocr_layout",
                "layout_lines": [asdict(line) for line in ordered],
                "layout_dropped": [asdict(line) for line in dropped],
                "layout_stats": {
                    "raw_lines": len(lines),
                    "kept_lines": len(ordered),
                    "dropped_lines": len(dropped),
                    "caption_lines": len(captions),
                    "column_count": self._estimate_column_count(kept),
                    "start_gate": start_gate,
                    "stop_gate": stop_gate,
                    "stop_gate_mode": self._stop_gate_mode(stop_gate) if stop_gate else "",
                    "boundary_events": "; ".join(
                        f"p{e.get('page_num')}:{e.get('heading')}:{e.get('judge')}:{e.get('heading_type')}:{e.get('confidence')}"
                        for e in self.boundary_events
                    )[:900],
                },
            })
        return pages

    def _find_document_start_gate(self, raw_pages: List[Dict[str, Any]]) -> Optional[Tuple[int, float]]:
        for raw_page in raw_pages:
            candidates = []
            for line in raw_page["layout_lines"]:
                if not is_layout_ignore_heading(line.text):
                    continue
                rel_y = line.y0 / max(line.page_height, 1.0)
                rel_width = (line.x1 - line.x0) / max(line.page_width, 1.0)
                if rel_y >= 0.25 or rel_width >= 0.25:
                    candidates.append(line)
            if candidates:
                chosen = sorted(candidates, key=lambda x: x.y0)[0]
                return (raw_page["page_num"], chosen.y1)
        return None

    def _find_document_stop_gate(self, raw_pages: List[Dict[str, Any]]) -> Optional[Tuple[int, float, str, float, float, float]]:
        for raw_page in raw_pages:
            stops = []
            candidates = []
            for line in raw_page["layout_lines"]:
                if not is_layout_stop_heading(line.text):
                    if is_candidate_section_boundary_heading(line):
                        candidates.append(line)
                    continue
                rel_y = line.y0 / max(line.page_height, 1.0)
                # Museum-Nieuws often appears as a running header at the very top.
                # Stop only at the real section heading lower on the page.
                if is_museum_news_heading(line.text) and rel_y < 0.14:
                    continue
                stops.append(line)
            if stops:
                chosen = sorted(stops, key=lambda x: x.y0)[0]
                reason = "museum_news_heading" if is_museum_news_heading(chosen.text) else "layout_stop_heading"
                return (raw_page["page_num"], chosen.y0, reason, chosen.x0, chosen.x1, chosen.page_width)
            for candidate in sorted(candidates, key=lambda x: x.y0):
                decision = self._judge_section_boundary(raw_pages, candidate)
                self.boundary_events.append(decision)
                if decision.get("stop_extraction"):
                    reason = safe_str(decision.get("reason")) or safe_str(decision.get("heading_type")) or "section_boundary"
                    return (raw_page["page_num"], candidate.y0, f"section_boundary:{reason}", candidate.x0, candidate.x1, candidate.page_width)
        return None

    def _judge_section_boundary(self, raw_pages: List[Dict[str, Any]], line: LayoutLine) -> Dict[str, Any]:
        heading_type, confidence, reason = classify_boundary_heading_rule(line)
        decision = {
            "page_num": line.page_num,
            "heading": line.text,
            "heading_type": heading_type,
            "confidence": confidence,
            "reason": reason,
            "stop_extraction": heading_type == "new_non_archaeological_section" and confidence >= 0.90,
            "judge": "rules",
        }
        if decision["stop_extraction"] or heading_type != "unknown_section_heading":
            return decision
        if self.boundary_client is None:
            return decision

        before, after = self._boundary_context(raw_pages, line)
        prompt = f"""
You are validating a section boundary in a Dutch archaeological bulletin OCR.

The extraction target is only the article/rubric "Archeologisch Nieuws".
Decide whether the text AFTER the candidate heading is still part of the
archaeological extraction target, or whether it starts a new non-target section.

Return strict JSON with keys:
stop_extraction
heading_type
confidence
reason

Allowed heading_type values:
archaeological_section
bibliography_within_archaeology
new_non_archaeological_section
running_header
caption
unknown

Candidate heading:
{json.dumps(line.text, ensure_ascii=False)}

Text before heading:
{json.dumps(before[-500:], ensure_ascii=False)}

Text after heading:
{json.dumps(after[:500], ensure_ascii=False)}
""".strip()
        try:
            data = self.boundary_client.propose(prompt)
            llm_type = safe_str(data.get("heading_type")) or "unknown"
            llm_conf = float(data.get("confidence", 0.0) or 0.0)
            stop = bool(data.get("stop_extraction", False))
            # Fail-safe: the LLM may only stop extraction on high confidence.
            decision.update({
                "heading_type": llm_type,
                "confidence": llm_conf,
                "reason": safe_str(data.get("reason"))[:280],
                "stop_extraction": stop and llm_type == "new_non_archaeological_section" and llm_conf >= 0.85,
                "judge": "llm_boundary",
            })
        except Exception as e:
            decision.update({
                "reason": f"llm_boundary_error:{safe_str(e)[:220]}",
                "stop_extraction": False,
                "judge": "llm_boundary_error",
            })
        return decision

    @staticmethod
    def _boundary_context(raw_pages: List[Dict[str, Any]], line: LayoutLine) -> Tuple[str, str]:
        target_page = next((p for p in raw_pages if p["page_num"] == line.page_num), None)
        if not target_page:
            return "", ""
        ordered = sorted(target_page["layout_lines"], key=lambda x: (x.y0, x.x0))
        idx = next((i for i, item in enumerate(ordered) if item is line), -1)
        if idx < 0:
            idx = min(range(len(ordered)), key=lambda i: abs(ordered[i].y0 - line.y0)) if ordered else 0
        before = " ".join(norm_ws(item.text) for item in ordered[max(0, idx - 12):idx])
        after = " ".join(norm_ws(item.text) for item in ordered[idx + 1:idx + 13])
        return before, after

    def _classify_line(
        self,
        line: LayoutLine,
        start_gate: Optional[Tuple[int, float]],
        stop_gate: Optional[Tuple[int, float]],
    ) -> Tuple[bool, str]:
        if start_gate is not None:
            start_page, start_y = start_gate
            if line.page_num < start_page or (line.page_num == start_page and line.y0 <= start_y):
                return False, "before_archaeological_news_region"
        if stop_gate is not None:
            stop_page, stop_y = stop_gate[:2]
            if line.page_num == stop_page and self._line_has_stop_section_context(line.text):
                return False, "after_stop_heading_context"
            if line.page_num > stop_page or (line.page_num == stop_page and self._line_is_after_stop_gate(line, stop_gate)):
                return False, "after_stop_heading"
        rel_y = line.y0 / max(line.page_height, 1.0)
        if rel_y < 0.14 and is_museum_news_heading(line.text):
            return False, "running_header"
        if rel_y < 0.13 and is_layout_ignore_heading(line.text):
            return False, "running_header"
        if is_layout_ignore_heading(line.text):
            return False, "rubric_heading"
        if is_caption_line(line.text):
            return False, "caption"
        if re.match(r"^\*?\s*\d+\s*$", norm_ws(line.text)):
            return False, "page_number"
        if line_is_noisy(line.text) and not is_protected_heading(line.text):
            return False, "noise"
        if is_known_province_heading(line.text):
            return True, "province_heading"
        return True, "body"

    @staticmethod
    def _stop_gate_mode(stop_gate: Optional[Tuple[Any, ...]]) -> str:
        if not stop_gate or len(stop_gate) < 6:
            return "page"
        _, _, _, stop_x0, stop_x1, page_width = stop_gate[:6]
        width_ratio = (float(stop_x1) - float(stop_x0)) / max(float(page_width), 1.0)
        page_span = float(stop_x0) <= float(page_width) * 0.22 and float(stop_x1) >= float(page_width) * 0.78
        if width_ratio >= 0.45 or page_span:
            return "page"
        return "column_zone"

    @staticmethod
    def _line_has_stop_section_context(text: str) -> bool:
        text_norm = norm_token(text)
        if not text_norm:
            return False
        compact = compact_layout_text(text)
        if re.search(
            r"\b(?:monumentenzorg|leefmilieu|deltaplan|wet\s+en\s+regelgeving|beschermde\s+monumenten|rijksmonumenten|evaluatie\s+van\s+de\s+wet|extra\s+middelen|aan\s+de\s+effectuering)\b",
            text_norm,
        ):
            return True
        return any(
            token in compact
            for token in {
                "monumentenzorg",
                "wetenregelgeving",
                "wetenregelgevingopditterrein",
                "regulierebudget",
                "extramiddelen",
                "effectuering",
                "deltaplan",
            }
        )

    def _line_is_after_stop_gate(self, line: LayoutLine, stop_gate: Tuple[Any, ...]) -> bool:
        """
        Stop headings in multi-column scans should not erase earlier columns on
        the same page. If MONUMENTENZORG starts in the middle/right column, keep
        the left archaeological column and drop only the stop-heading zone.
        """
        stop_page, stop_y = stop_gate[:2]
        if line.page_num != stop_page or line.y0 < stop_y:
            return False
        if self._line_has_stop_section_context(line.text):
            return True
        if len(stop_gate) < 6 or self._stop_gate_mode(stop_gate) == "page":
            return True
        _, _, _, stop_x0, stop_x1, page_width = stop_gate[:6]
        line_mid = (line.x0 + line.x1) / 2.0
        page_width = max(float(page_width), 1.0)
        stop_mid = (float(stop_x0) + float(stop_x1)) / 2.0
        if page_width * 0.28 <= stop_mid <= page_width * 0.72:
            stop_column_left = page_width / 3.0
        elif stop_mid > page_width * 0.72:
            stop_column_left = page_width * 2.0 / 3.0
        else:
            stop_column_left = float(stop_x0)
        margin = max(page_width * 0.025, 14.0)
        return line_mid >= max(0.0, stop_column_left - margin)

    def _order_lines_by_column(self, lines: List[LayoutLine]) -> List[LayoutLine]:
        if not lines:
            return []
        column_count = self._estimate_column_count(lines)
        page_width = max((line.page_width for line in lines), default=1.0)

        def key(line: LayoutLine) -> Tuple[int, float, float]:
            mid_x = (line.x0 + line.x1) / 2.0
            col = min(column_count - 1, max(0, int(mid_x / max(page_width, 1.0) * column_count)))
            return (col, line.y0, line.x0)

        ordered = sorted(lines, key=key)
        return ordered

    def _estimate_column_count(self, lines: List[LayoutLine]) -> int:
        if len(lines) < 8:
            return 1
        page_width = max((line.page_width for line in lines), default=1.0)
        mids = sorted(((line.x0 + line.x1) / 2.0) / max(page_width, 1.0) for line in lines)
        left = sum(1 for x in mids if x < 0.34)
        middle = sum(1 for x in mids if 0.34 <= x < 0.67)
        right = sum(1 for x in mids if x >= 0.67)
        if min(left, middle, right) >= max(3, int(len(lines) * 0.12)):
            return 3
        left_half = sum(1 for x in mids if x < 0.50)
        right_half = len(mids) - left_half
        if min(left_half, right_half) >= max(4, int(len(lines) * 0.18)):
            return 2
        return 1


class PDFExtractor:
    header_noise = [re.compile(p, re.I) for p in NOISY_LINE_PATTERNS]

    def __init__(
        self,
        flatten_linebreaks=False,
        start_page=None,
        end_page=None,
        enable_ocr_fallback=True,
        ocr_render_dpi=240,
        ocr_min_text_chars=20,
        ocr_text_min_chars_for_pdfplumber=50,
        enable_ocr_readable_pdf_cache=False,
        keep_runtime_cache=False,
        runtime_cache_dir: Optional[Path] = None,
        enable_llm_boundary_review=False,
        anthropic_api_key: str = "",
        anthropic_model: str = "",
        anthropic_api_version: str = "2023-06-01",
    ):
        self.flatten_linebreaks = flatten_linebreaks
        self.start_page = start_page
        self.end_page = end_page
        self.enable_ocr_fallback = enable_ocr_fallback
        self.ocr_render_dpi = ocr_render_dpi
        self.ocr_min_text_chars = ocr_min_text_chars
        self.ocr_text_min_chars_for_pdfplumber = ocr_text_min_chars_for_pdfplumber
        self.enable_ocr_readable_pdf_cache = enable_ocr_readable_pdf_cache
        self.keep_runtime_cache = keep_runtime_cache
        self.runtime_cache_dir = Path(runtime_cache_dir) if runtime_cache_dir else None
        self.boundary_client = None
        if enable_llm_boundary_review:
            try:
                self.boundary_client = AnthropicJSONClient(
                    api_key=anthropic_api_key,
                    model=anthropic_model,
                    api_version=anthropic_api_version,
                )
            except Exception:
                self.boundary_client = None
        self.ocr = OCRExtractor(dpi=self.ocr_render_dpi, boundary_client=self.boundary_client) if self.enable_ocr_fallback else None
        self.status_rows: List[Dict[str, Any]] = []

    def extract_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        pages = []
        raw_layout_pages = []
        pdfplumber_chars = 0
        pdfplumber_weak_pages = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if self.start_page is not None and i < self.start_page:
                    continue
                if self.end_page is not None and i > self.end_page:
                    continue
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                text_chars = len(norm_ws(text))
                pdfplumber_chars += text_chars
                if text_chars < self.ocr_text_min_chars_for_pdfplumber:
                    pdfplumber_weak_pages += 1
                pages.append({"page_num": i, "text": self.clean_text(text), "source": "pdfplumber"})
                layout_lines = self._extract_pdfplumber_layout_lines(page, i)
                if layout_lines:
                    raw_layout_pages.append({"page_num": i, "layout_lines": layout_lines, "source": "pdfplumber_layout"})
        self.status_rows.append({
            "component": "hybrid_text_extraction",
            "pdf_file": pdf_path.name,
            "pdfplumber_chars": pdfplumber_chars,
            "pdfplumber_weak_pages": pdfplumber_weak_pages,
            "page_count": len(pages),
            "image_only_likely": int(bool(pages) and pdfplumber_chars < self.ocr_text_min_chars_for_pdfplumber),
            "ocr_fallback_enabled": int(self.enable_ocr_fallback),
            "ocr_readable_pdf_cache_enabled": int(self.enable_ocr_readable_pdf_cache),
            "keep_runtime_cache": int(self.keep_runtime_cache),
        })
        if raw_layout_pages and self.ocr is not None:
            layout_pages = self.ocr._layout_filter_pages(raw_layout_pages)
            layout_map = {p["page_num"]: p for p in layout_pages}
            for page in pages:
                layout_page = layout_map.get(page["page_num"])
                if layout_page and self._prefer_layout_text(layout_page["text"], page["text"]):
                    page["text"] = self.clean_text(layout_page["text"])
                    page["source"] = "pdfplumber_layout"
                    page["layout_stats"] = layout_page.get("layout_stats", {})
        if self.enable_ocr_fallback and self.ocr is not None:
            ocr_pages = self.ocr.extract_pdf_pages(pdf_path, self.start_page, self.end_page)
            ocr_map = {p["page_num"]: p for p in ocr_pages}
            for page in pages:
                if len(norm_ws(page["text"])) < self.ocr_min_text_chars and page["page_num"] in ocr_map:
                    ocr_page = ocr_map[page["page_num"]]
                    page["text"] = self.clean_text(ocr_page["text"])
                    page["source"] = safe_str(ocr_page.get("source")) or "ocr"
                    page["layout_stats"] = ocr_page.get("layout_stats", {})
        # Runtime cache is scaffolded for later OCR-readable PDF experiments but
        # is intentionally not created unless an explicit cache feature writes to
        # it. This keeps normal runs clutter-free.
        for page in pages:
            self.status_rows.append({
                "component": "page_extract",
                "page_num": page["page_num"],
                "source": page["source"],
                "text_chars": len(page["text"]),
            })
            if page.get("layout_stats"):
                stats = page["layout_stats"]
                self.status_rows.append({
                    "component": "layout_filter",
                    "page_num": page["page_num"],
                    "source": page["source"],
                    "raw_lines": stats.get("raw_lines", ""),
                    "kept_lines": stats.get("kept_lines", ""),
                    "dropped_lines": stats.get("dropped_lines", ""),
                    "caption_lines": stats.get("caption_lines", ""),
                    "column_count": stats.get("column_count", ""),
                    "start_gate": safe_str(stats.get("start_gate")),
                    "stop_gate": safe_str(stats.get("stop_gate")),
                    "stop_gate_mode": safe_str(stats.get("stop_gate_mode")),
                })
        return pages

    def _extract_pdfplumber_layout_lines(self, page: Any, page_num: int) -> List[LayoutLine]:
        try:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=False) or []
        except Exception:
            return []
        if not words:
            return []
        words = sorted(words, key=lambda w: (float(w.get("top", 0)), float(w.get("x0", 0))))
        grouped = []
        current = []
        current_top = None
        tolerance = 3.5
        for word in words:
            top = float(word.get("top", 0))
            if current and current_top is not None and abs(top - current_top) > tolerance:
                grouped.append(current)
                current = []
            current.append(word)
            current_top = top if current_top is None else (current_top + top) / 2.0
        if current:
            grouped.append(current)

        lines = []
        for group in grouped:
            group = sorted(group, key=lambda w: float(w.get("x0", 0)))
            text = norm_ws(" ".join(safe_str(w.get("text")) for w in group))
            if not text:
                continue
            lines.append(LayoutLine(
                page_num=page_num,
                text=text,
                x0=min(float(w.get("x0", 0)) for w in group),
                y0=min(float(w.get("top", 0)) for w in group),
                x1=max(float(w.get("x1", 0)) for w in group),
                y1=max(float(w.get("bottom", 0)) for w in group),
                page_width=float(page.width or 1),
                page_height=float(page.height or 1),
            ))
        return lines

    @staticmethod
    def _prefer_layout_text(layout_text: str, plain_text: str) -> bool:
        layout_clean = norm_ws(layout_text)
        plain_clean = norm_ws(plain_text)
        if len(layout_clean) < 80:
            return False
        if len(plain_clean) < 80:
            return True
        if is_caption_line(plain_clean[:140]) and not is_caption_line(layout_clean[:140]):
            return True
        # Layout text is preferred when it preserves substantially more content or likely fixes columns.
        return len(layout_clean) >= len(plain_clean) * 0.90

    def clean_text(self, text: str) -> str:
        text = clean_ocr_text(text)
        lines = []
        for line in text.splitlines():
            line = norm_ws(line)
            if not line or line_is_noisy(line):
                continue
            if is_caption_line(line):
                continue
            if any(p.search(line) for p in self.header_noise) and not is_protected_heading(line):
                continue
            lines.append(line)
        cleaned = "\n".join(lines)
        if self.flatten_linebreaks:
            cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
            cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


# =========================
# STRUCTURAL SEGMENTATION
# =========================
# Purpose:
# Segmentation determines where an archaeological record starts and ends.
#
# What it does:
# - HeuristicSectionSegmenter uses headings, place names, provinces and layout to
#   split text into records.
# - LLMStructuralSegmenter can assist with segment boundaries, but is used
#   cautiously so long OCR fragments are not rewritten by the LLM.
#
# Why this is useful:
# Good segmentation is the foundation for everything afterwards. If two sites or
# periods are mixed in one chunk, place, dating and ABR codes can be linked to the
# wrong context.

class HeuristicSectionSegmenter:
    def __init__(self, refs: ReferenceStore):
        self.refs = refs
        self.province_lookup = {norm_token(p): canonical_province_name(p) for p in DUTCH_PROVINCES}

    def is_province_heading(self, line: str) -> Optional[str]:
        line_norm = norm_token(line)
        if line_norm in EXTRA_REGION_HEADINGS:
            return EXTRA_REGION_HEADINGS[line_norm]
        if line_norm in self.province_lookup and len(line_norm.split()) <= 2:
            return self.province_lookup[line_norm]
        return None

    def looks_like_place_heading(self, line: str) -> bool:
        raw = norm_ws(line)
        raw_norm = norm_token(raw)
        if is_caption_line(raw) or is_bibliography_line(raw):
            return False
        if self.refs.is_period_term(raw):
            return False
        if not raw or raw_norm in BAD_HEADINGS or raw_norm in PLACE_REJECT_EXACT:
            return False
        if raw_norm in PLACE_CONTEXT_AMBIGUOUS_COMMON_PLACE_NAMES and raw_norm not in SITE_PARENT_MAP:
            return False
        if not (raw[0].isupper() or raw.startswith(("'", "’"))):
            return False
        if re.search(r"\b(r\.?\s*o\.?\s*b|b\.?\s*a\.?\s*i|r\.?\s*u)\b", raw_norm):
            return False
        if raw_norm.startswith(("tot onze", "het onderzoek", "gelukkig", "zeer waarschijnlijk")):
            return False
        if len(raw_norm) < 4 or len(raw_norm) > 45:
            return False
        if re.search(r"\d", raw):
            return False
        tokens = raw_norm.split()
        place_particles = {"de", "den", "ter", "te", "het", "'s", "s"}
        if any(tok in CONTEXT_WORDS_REJECT and not (idx == 0 and tok in place_particles) for idx, tok in enumerate(tokens)):
            return False
        if len(tokens) > 4:
            return False
        if raw_norm.startswith(("afb", "fig", "plaat")):
            return False
        if not re.match(r"^(?:['’]s[- ]*)?[^\W\d_](?:[^\W\d_]|['’\- ])+$", raw, flags=re.UNICODE):
            return False
        return True

    def _extract_item_start(self, line: str) -> Optional[Tuple[str, str]]:
        line = norm_ws(line)
        if not line:
            return None
        patterns = [
            r"^((?:['’]s[- ]*)?[^\W\d_](?:[^\W\d_]|['’\- ]){2,45})\.\s*(.+)$",
            r"^((?:['’]s[- ]*)?[^\W\d_](?:[^\W\d_]|['’\- ]){2,45}\s*\([^)]+\))\.\s*(.+)$",
            r"^((?:['’]s[- ]*)?[^\W\d_](?:[^\W\d_]|['’\- ]){2,45}),\s*gem\.[^.]+\.\s*(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, line, flags=re.I)
            if not m:
                continue
            place_raw = norm_ws(m.group(1))
            rest = norm_ws(m.group(2))
            place_norm = norm_token(re.sub(r"\s*\([^)]+\)", "", place_raw))
            if not self.looks_like_place_heading(re.sub(r"\s*\([^)]+\)", "", place_raw)):
                return None
            return place_raw, rest
        if self.looks_like_place_heading(line):
            return line, ""
        return None

    def segment(self, pdf_file: str, pages: List[Dict[str, Any]]) -> List[Section]:
        sections = []
        current_province = ""
        current_place_raw = ""
        current_place_norm = ""
        current_lines = []
        current_notes = []
        start_page = None
        end_page = None
        last_page_num = None

        def flush():
            nonlocal current_lines, current_place_raw, current_place_norm, start_page, end_page, current_notes
            text_blob = "\n".join(current_lines).strip()
            if not text_blob:
                return
            confidence = self._score_segmentation(current_province, current_place_raw, current_lines, current_notes)
            sections.append(Section(
                pdf_file=pdf_file,
                section_id=f"SEC_{len(sections)+1:05d}",
                province_anchor=current_province,
                place_heading_raw=current_place_raw,
                place_heading_normalised=current_place_norm,
                start_page=start_page or 1,
                end_page=end_page or (start_page or 1),
                text=text_blob,
                segmentation_confidence=confidence,
                segmentation_notes=list(current_notes),
                segment_source="heuristic",
            ))
            current_lines = []
            current_place_raw = ""
            current_place_norm = ""
            start_page = None
            end_page = None
            current_notes = []

        for page in pages:
            page_num = page["page_num"]
            lines = [norm_ws(x) for x in page["text"].splitlines() if norm_ws(x)]
            for line in lines:
                if is_caption_line(line):
                    current_notes = current_notes + [f"caption_skipped:p{page_num}"]
                    continue
                province = self.is_province_heading(line)
                if province:
                    flush()
                    current_province = province
                    current_notes = [f"province_heading:{province}"]
                    last_page_num = page_num
                    continue
                item_start = self._extract_item_start(line)
                if item_start is not None:
                    flush()
                    place_raw, rest = item_start
                    current_place_raw = place_raw
                    current_place_norm = norm_token(re.sub(r"\s*\([^)]+\)", "", place_raw))
                    start_page = page_num
                    end_page = page_num
                    current_notes = current_notes + [f"place_heading:{place_raw}"]
                    if rest:
                        current_lines.append(rest)
                    last_page_num = page_num
                    continue
                if start_page is None:
                    start_page = page_num
                elif last_page_num is not None and page_num != last_page_num and current_lines:
                    current_notes = current_notes + [f"continued_across_page:{last_page_num}->{page_num}"]
                end_page = page_num
                current_lines.append(line)
                last_page_num = page_num

        flush()
        if not sections:
            all_text = "\n\n".join([p["text"] for p in pages if norm_ws(p["text"])])
            if all_text:
                sections.append(Section(
                    pdf_file=pdf_file,
                    section_id="SEC_00001",
                    province_anchor="",
                    place_heading_raw="",
                    place_heading_normalised="",
                    start_page=pages[0]["page_num"],
                    end_page=pages[-1]["page_num"],
                    text=all_text,
                    segmentation_confidence=1,
                    segmentation_notes=["global_fallback_section"],
                    segment_source="heuristic",
                ))
        return sections

    def _score_segmentation(self, province: str, place: str, lines: List[str], notes: List[str]) -> int:
        score = 0
        text_len = len("\n".join(lines))
        if province:
            score += 1
        else:
            notes.append("missing_province_anchor")
        if place:
            score += 1
        else:
            notes.append("missing_place_heading")
        if text_len >= 120:
            score += 1
        else:
            notes.append("short_section")
        return min(score, 3)


class LLMStructuralSegmenter:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.status_rows: List[Dict[str, Any]] = []
        self.client = None
        if not cfg.enable_llm_structural_segmentation:
            self.status_rows.append({"component": "llm_structural_segmentation", "enabled": False, "message": "disabled"})
            return
        try:
            self.client = AnthropicJSONClient(
                api_key=cfg.anthropic_api_key,
                model=cfg.anthropic_model,
                api_version=cfg.anthropic_api_version,
            )
            self.status_rows.append({
                "component": "llm_structural_segmentation",
                "enabled": True,
                "model": cfg.anthropic_model,
                "message": "ok",
            })
        except Exception as e:
            self.status_rows.append({"component": "llm_structural_segmentation", "enabled": True, "message": str(e)})

    def available(self) -> bool:
        return self.client is not None

    def segment(self, pdf_file: str, pages: List[Dict[str, Any]]) -> List[Section]:
        if not self.available():
            return []
        page_batches = self._make_page_batches(pages, self.cfg.llm_segmentation_max_pages_per_call)
        segments: List[Section] = []
        seg_counter = 1
        for batch in page_batches:
            try:
                prompt = self._build_prompt(batch)
                data = self.client.propose(prompt)
                batch_segments = self._parse_segments(pdf_file, data, batch, seg_counter)
                seg_counter += len(batch_segments)
                segments.extend(batch_segments)
            except Exception as e:
                self.status_rows.append({
                    "component": "llm_structural_segmentation_batch",
                    "pages": f"{batch[0]['page_num']}-{batch[-1]['page_num']}",
                    "message": str(e),
                })
        return segments

    def _make_page_batches(self, pages: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        batches = []
        current = []
        current_chars = 0
        for page in pages:
            text = safe_str(page["text"])
            if current and (len(current) >= batch_size or current_chars + len(text) > self.cfg.llm_segmentation_max_text_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(page)
            current_chars += len(text)
        if current:
            batches.append(current)
        return batches

    def _build_prompt(self, batch: List[Dict[str, Any]]) -> str:
        payload = [{"page_num": p["page_num"], "text": safe_str(p["text"])[:self.cfg.llm_segmentation_max_text_chars]} for p in batch]
        return f"""
You are doing structural segmentation only for a Dutch archaeological bulletin-style OCR text.

Important:
- Do NOT extract archaeological labels.
- Do NOT interpret finds, features, or materials.
- Only split the OCR text into coherent structural record segments.
- Suggest province_candidate and place_candidate only if structurally clear.
- Preserve the original text.
- Escape all JSON strings correctly. Do not put raw line breaks inside JSON string values.
- Encode segment_text as one JSON string with "\\n" escapes for line breaks.
- Do not wrap the response in markdown fences.

Return strict JSON:
{{
  "segments": [
    {{
      "start_page": 1,
      "end_page": 1,
      "province_candidate": "",
      "place_candidate": "",
      "segment_text": "",
      "segmentation_confidence": 1,
      "notes": ""
    }}
  ]
}}

Input pages:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    def _parse_segments(self, pdf_file: str, data: Dict[str, Any], batch: List[Dict[str, Any]], seg_counter_start: int) -> List[Section]:
        segments = []
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            return segments
        page_nums = [p["page_num"] for p in batch]
        min_page = min(page_nums)
        max_page = max(page_nums)
        counter = seg_counter_start
        for item in raw_segments:
            segment_text = norm_ws(safe_str(item.get("segment_text")))
            if not segment_text:
                continue
            start_page = max(min_page, int(item.get("start_page", min_page) or min_page))
            end_page = min(max_page, int(item.get("end_page", start_page) or start_page))
            province_candidate = safe_str(item.get("province_candidate"))
            place_candidate = safe_str(item.get("place_candidate"))
            confidence = max(1, min(3, int(item.get("segmentation_confidence", 1) or 1)))
            notes = split_semicolon_values(item.get("notes"))
            segments.append(Section(
                pdf_file=pdf_file,
                section_id=f"SEC_{counter:05d}",
                province_anchor=province_candidate,
                place_heading_raw=place_candidate,
                place_heading_normalised=norm_token(place_candidate),
                start_page=start_page,
                end_page=end_page,
                text=segment_text,
                segmentation_confidence=confidence,
                segmentation_notes=notes,
                segment_source="llm",
            ))
            counter += 1
        return segments


# =========================
# CANDIDATE EXTRACTION
# =========================
# Purpose:
# CandidateProposer finds raw candidate terms before they are normalised to ABR,
# place or dating values.
#
# What it does:
# - Scans sentences for places, periods, finds, structures and materials.
# - Combines lookup matches, trigger phrases and ArcheoBERTje candidates.
# - Assigns terms to the most likely bucket: feature, finding or material.
#
# Why this is useful:
# This is the first content-extraction step. Collecting candidates broadly and
# mapping them strictly afterwards reduces the chance of missing archaeological
# information.

class CandidateProposer:
    def __init__(self, refs: ReferenceStore, archaeobertje: Optional[ArcheoBERTjeHelper] = None):
        self.refs = refs
        self.archaeobertje = archaeobertje

    def propose(self, section: Section, chunk_text_: str) -> Dict[str, List[str]]:
        out = {"places": [], "features": [], "findings": [], "materials": [], "dating_phrases": []}
        for sentence in split_sentences(chunk_text_):
            sentence_candidates = self._propose_sentence(sentence)
            for key in out:
                out[key] = dedupe_keep_order(out[key] + sentence_candidates.get(key, []))
        out["features"], out["findings"], out["materials"] = self._disambiguate(out["features"], out["findings"], out["materials"])
        return out

    def _propose_sentence(self, sentence: str) -> Dict[str, List[str]]:
        result = {"places": [], "features": [], "findings": [], "materials": [], "dating_phrases": []}
        if has_editorial_context(sentence):
            result["dating_phrases"] += self._find_dating_phrases(sentence)
            return result

        result["features"] += self._match_terms_in_text(sentence, self.refs.feature_lookup)
        result["findings"] += self._match_terms_in_text(sentence, self.refs.findings_lookup)
        result["materials"] += self._match_terms_in_text(sentence, self.refs.material_lookup)
        result["dating_phrases"] += self._find_dating_phrases(sentence)

        for term in compound_terms_from_text(sentence):
            kind = self._guess_kind(norm_token(term))
            if kind == "feature":
                result["features"].append(term)
            elif kind == "finding":
                result["findings"].append(term)
            elif kind == "material":
                result["materials"].append(term)

        lower = sentence.lower()
        for _, row in self.refs.gazetteer.iterrows():
            name = safe_str(row.get("name"))
            name_norm = norm_token(name)
            if (
                not name
                or name_norm in PLACE_REJECT_EXACT
                or len(name_norm) < 5
                or name_norm in EDITORIAL_LOCATIONS
                or self.refs.is_period_term(name)
            ):
                continue
            if re.search(rf"\b{re.escape(name.lower())}\b", lower):
                result["places"].append(name)

        for phrase in extract_trigger_phrases(sentence):
            phrase_norm = norm_token(phrase)
            kind = self._guess_kind(phrase_norm)
            if kind == "material":
                result["materials"].append(phrase)
            elif kind == "feature":
                result["features"].append(phrase)
            elif kind == "finding":
                result["findings"].append(phrase)

        if self.archaeobertje is not None and self.archaeobertje.available:
            for span in candidate_spans_from_text(sentence, max_tokens=3):
                for kind, key in [("feature", "features"), ("finding", "findings"), ("material", "materials")]:
                    semantic = self.archaeobertje.semantic_match(span, kind)
                    if semantic is not None:
                        result[key].append(span)

        for key in result:
            kind = {"features": "feature", "findings": "finding", "materials": "material", "places": "place"}.get(key, "")
            result[key] = dedupe_keep_order([
                clean_candidate_phrase(v)
                for v in result[key]
                if clean_candidate_phrase(v)
                and not self.refs.is_rejected_term(v, kind)
                and not (kind == "place" and self.refs.is_period_term(v))
            ])
        return result

    def _guess_kind(self, phrase_norm: str) -> str:
        tokens = set(phrase_norm.split())
        if phrase_norm in PREFERRED_MATERIAL_TERMS or tokens & MATERIAL_HINT_WORDS:
            return "material"
        if phrase_norm in PREFERRED_FEATURE_TERMS or tokens & FEATURE_HINT_WORDS:
            return "feature"
        if phrase_norm in PREFERRED_FINDING_TERMS or tokens & FINDING_HINT_WORDS:
            return "finding"
        return ""

    def _disambiguate(self, features: List[str], findings: List[str], materials: List[str]) -> Tuple[List[str], List[str], List[str]]:
        f_set = dedupe_keep_order(features)
        fi_set = dedupe_keep_order(findings)
        m_set = dedupe_keep_order(materials)

        def move(term: str, target: str):
            nonlocal f_set, fi_set, m_set
            for bucket in [f_set, fi_set, m_set]:
                if term in bucket:
                    bucket.remove(term)
            if target == "feature":
                f_set.append(term)
            elif target == "finding":
                fi_set.append(term)
            elif target == "material":
                m_set.append(term)

        for term in dedupe_keep_order(f_set + fi_set + m_set):
            n = norm_token(term)
            override = self.refs.custom_field_type_overrides.get(n)
            if override:
                move(term, override["kind"])
                continue
            if n in PREFERRED_FEATURE_TERMS:
                move(term, "feature")
            elif n in PREFERRED_FINDING_TERMS:
                move(term, "finding")
            elif n in PREFERRED_MATERIAL_TERMS:
                move(term, "material")

        return dedupe_keep_order(f_set), dedupe_keep_order(fi_set), dedupe_keep_order(m_set)

    def _match_terms_in_text(self, text: str, lookup: Dict[str, Dict[str, Any]]) -> List[str]:
        out, text_norm = [], norm_token(text)
        compact_text = re.sub(r"[^a-z0-9]+", "", text_norm)
        for alias, payload in lookup.items():
            if not alias or len(alias) < 4:
                continue
            alias_compact = re.sub(r"[^a-z0-9]+", "", alias)
            compound_hit = alias in DUTCH_COMPOUND_TERM_PARTS and alias_compact and alias_compact in compact_text
            if re.search(rf"\b{re.escape(alias)}\b", text_norm) or compound_hit:
                out.append(payload["label"])
        return dedupe_keep_order(out)

    def _find_dating_phrases(self, text: str) -> List[str]:
        results = []
        for pat in DATE_PATTERNS:
            results.extend([m.group(0) for m in re.finditer(pat, text, flags=re.I)])
        return dedupe_keep_order(results)


# =========================
# PLACE RESOLUTION AND CONTEXTUAL PLACE VALIDATOR
# =========================
# Purpose:
# PlaceResolver selects the best place/province and blocks false place names.
#
# What it does:
# - Custom place aliases are applied first.
# - The resolver then tries exact and fuzzy matching against the gazetteer.
# - The contextual validator checks whether a candidate functions as a real
#   location in the text rather than as a direction, object, material, author name,
#   street fragment or OCR artefact.
#
# Why this is useful:
# Place errors damage the entire record because they mislocate all associated
# dating and finds. This validator is especially important for difficult documents
# where words such as Noorden, Helling, Putten or Steeg can be false positives.

class PlaceResolver:
    def __init__(self, refs: ReferenceStore, threshold: float = 0.95):
        self.refs = refs
        self.threshold = threshold

    def resolve(self, raw_candidates: List[str], province_anchor: str, context_text: str = "") -> PlaceResolution:
        raw_candidates = dedupe_keep_order(raw_candidates)
        alt_only_candidate = ""
        common_noun_reject = ""
        homonym_reject = ""
        for raw in raw_candidates:
            parenthetical = self._parenthetical_place(raw, province_anchor)
            if parenthetical is not None:
                return parenthetical
            custom = self.refs.resolve_custom_place_alias(raw)
            if custom is not None:
                return custom
            site_parent = self.refs.resolve_site_parent_place(raw)
            if site_parent is not None:
                return site_parent
            street_parent = self.refs.resolve_street_parent_place(raw)
            if street_parent is not None:
                return street_parent
            landscape = self._landscape_water_place(raw, context_text, province_anchor)
            if landscape is not None:
                return landscape
            homonym_status = self._homonym_place_status(raw, context_text)
            if homonym_status in {"feature_context", "unclear"}:
                if not homonym_reject:
                    homonym_reject = norm_ws(raw)
                continue
            if not alt_only_candidate and self._is_unmapped_street_like(raw):
                alt_only_candidate = norm_ws(raw)
            if not common_noun_reject and self._is_guarded_common_noun_place(raw) and not self._has_explicit_place_context(raw, context_text):
                common_noun_reject = norm_ws(raw)
        candidates = [
            c for c in raw_candidates
            if self._valid_raw_candidate(c) and self._context_supports_place_candidate(c, context_text, province_anchor)
        ]
        if not candidates:
            if alt_only_candidate:
                return PlaceResolution(
                    place_raw=alt_only_candidate,
                    place_normalised="",
                    alt_place_name=alt_only_candidate,
                    province_normalised=province_anchor,
                    province_source="anchor" if province_anchor else "unresolved",
                    resolution_type="unresolved_street",
                    ambiguous=True,
                )
            if common_noun_reject:
                return PlaceResolution(
                    place_raw=common_noun_reject,
                    place_normalised="",
                    alt_place_name="",
                    province_normalised=province_anchor,
                    province_source="anchor" if province_anchor else "unresolved",
                    resolution_type="unresolved_common_noun",
                    ambiguous=True,
                )
            if homonym_reject:
                return PlaceResolution(
                    place_raw=homonym_reject,
                    place_normalised="",
                    alt_place_name="",
                    province_normalised=province_anchor,
                    province_source="anchor" if province_anchor else "unresolved",
                    resolution_type="unresolved_homonym",
                    ambiguous=True,
                )
            return PlaceResolution(
                province_normalised=province_anchor,
                province_source="anchor" if province_anchor else "unresolved",
                resolution_type="unresolved",
            )
        for raw in candidates:
            exact = self._exact(raw, province_anchor)
            if exact is not None:
                if self._is_homonym_place(raw):
                    exact.resolution_type = "exact_homonym_place"
                    exact.ambiguous = True
                return exact
        for raw in candidates:
            fuzzy = self._fuzzy(raw, province_anchor)
            if fuzzy is not None:
                return fuzzy
        return PlaceResolution(
            place_raw=candidates[0],
            province_normalised=province_anchor,
            province_source="anchor" if province_anchor else "unresolved",
            resolution_type="unresolved",
        )

    def _landscape_water_place(self, raw: str, text: str, province_anchor: str = "") -> Optional[PlaceResolution]:
        raw_norm = norm_token(raw)
        if raw_norm not in LANDSCAPE_WATER_PLACE_TERMS:
            return None
        context = norm_token(text)
        if context and (LANDSCAPE_WATER_CONTEXT_RE.search(context) or raw_norm in context):
            return PlaceResolution(
                place_raw=raw,
                place_normalised="",
                alt_place_name=norm_ws(raw),
                province_normalised=province_anchor,
                province_source="landscape_context",
                resolution_type="unresolved_landscape",
                ambiguous=True,
            )
        return None

    def _parenthetical_place(self, raw: str, province_anchor: str = "") -> Optional[PlaceResolution]:
        """
        Normalise place strings like "Den Burg (Texel)" without turning the
        parenthetical context into the main analytical place.
        """
        raw_clean = norm_ws(raw)
        match = re.fullmatch(r"(.+?)\s*\(([^()]{2,60})\)\s*", raw_clean)
        if not match:
            return None
        main = norm_ws(match.group(1))
        alt = norm_ws(match.group(2))
        if not main or not self._valid_raw_candidate(main):
            return None
        exact = self._exact(main, province_anchor)
        if exact is None:
            exact = self._fuzzy(main, province_anchor)
        if exact is None or not exact.place_normalised:
            return None
        exact.place_raw = raw_clean
        exact.alt_place_name = join_semicolon([
            value for value in [exact.alt_place_name, alt]
            if norm_token(value) != norm_token(exact.place_normalised)
        ])
        exact.resolution_type = "parenthetical_place"
        return exact

    def _valid_raw_candidate(self, raw: str) -> bool:
        raw_norm = norm_token(raw)
        if self.refs.is_rejected_term(raw, "place"):
            return False
        if self.refs.is_period_term(raw):
            return False
        if raw_norm in EVENT_WORD_PLACE_REJECT or raw_norm in HISTORICAL_PERSON_PLACE_REJECT:
            return False
        if raw_norm in LANDSCAPE_AS_PLACE_TERMS and raw_norm not in SITE_PARENT_MAP and raw_norm not in STREET_PARENT_MAP:
            return False
        if self._is_unmapped_street_like(raw):
            return False
        if not raw_norm or raw_norm in PLACE_REJECT_EXACT or raw_norm in EDITORIAL_LOCATIONS or raw_norm in BAD_HEADINGS:
            return False
        if (
            raw_norm in PLACE_CONTEXT_AMBIGUOUS_COMMON_PLACE_NAMES
            and raw_norm not in PLACE_CONTEXT_COMMON_NOUN_PLACE_GUARD
            and raw_norm not in SITE_PARENT_MAP
            and raw_norm not in STREET_PARENT_MAP
        ):
            return False
        if (
            raw_norm not in SITE_PARENT_MAP
            and raw_norm not in STREET_PARENT_MAP
            and (
                raw_norm in self.refs.feature_lookup
                or raw_norm in self.refs.material_lookup
                or raw_norm in PLACE_CONTEXT_OBJECT_WORDS
            )
        ):
            return False
        if raw_norm.startswith(PLACE_REJECT_PREFIXES):
            return False
        if any(fragment in raw_norm for fragment in PLACE_REJECT_SUBSTRINGS):
            return False
        if re.search(r"\b(gem|gemeente|sectie|straat|weg|laan|onderzoek|opgraving|rijksmuseum|internaat|stichting)\b", raw_norm):
            return False
        if re.search(r"\b(dr|ir|prof|museum|universiteit|instituut)\b", raw_norm):
            return False
        tokens = raw_norm.split()
        context_tokens = tokens[1:] if tokens and tokens[0] in {"de", "het", "den", "ter", "ten"} else tokens
        if any(tok in CONTEXT_WORDS_REJECT for tok in context_tokens):
            return False
        if len(raw_norm) < 4 or len(raw_norm.split()) > 4:
            return False
        return True

    def _context_supports_place_candidate(self, raw: str, text: str, province_anchor: str = "") -> bool:
        """
        Guard against gazetteer matches on ordinary words. A candidate is kept
        when it behaves like a location in the local text, not merely as a
        direction, material adjective, street fragment, object, or author name.
        """
        if not text:
            return True

        raw_norm = norm_token(raw)
        if not raw_norm:
            return False
        if self.refs.is_period_term(raw):
            return False
        if raw_norm in EVENT_WORD_PLACE_REJECT or raw_norm in HISTORICAL_PERSON_PLACE_REJECT:
            return False
        if raw_norm in self.refs.custom_place_aliases:
            return True
        if raw_norm in SITE_PARENT_MAP or raw_norm in STREET_PARENT_MAP:
            return True
        homonym_status = self._homonym_place_status(raw, text)
        if homonym_status in {"feature_context", "unclear"}:
            return False
        if self._is_guarded_common_noun_place(raw) and not self._has_explicit_place_context(raw, text):
            return False
        if raw_norm in LANDSCAPE_WATER_PLACE_TERMS:
            return False
        if raw_norm in LANDSCAPE_AS_PLACE_TERMS:
            return False
        if self._is_unmapped_street_like(raw):
            return False

        text_norm = norm_token(text)
        if not text_norm:
            return raw_norm not in PLACE_CONTEXT_AMBIGUOUS_COMMON_PLACE_NAMES
        if has_person_name_place_context(raw, text):
            return False

        if raw_norm in PLACE_CONTEXT_DIRECTION_WORDS:
            direction_pattern = rf"\b(ten|in het|aan de|op de|naar het|naar de)\s+{re.escape(raw_norm)}\s+van\b"
            if re.search(direction_pattern, text_norm):
                return False

        strong_patterns = [
            rf"\b(?:te|bij|in|nabij|onder|rond|binnen)\s+(?:de\s+|het\s+|den\s+|ter\s+|ten\s+)?{re.escape(raw_norm)}\b",
            rf"\b(?:gemeente|gem|plaats|stad|dorp|provincie|prov)\.?\s+{re.escape(raw_norm)}\b",
            rf"\b{re.escape(raw_norm)}\s*,?\s*(?:gemeente|gem|provincie|prov)\b",
        ]
        if any(re.search(pattern, text_norm) for pattern in strong_patterns):
            return True

        raw_re = re.compile(rf"(?<![A-Za-zÀ-ÿ]){re.escape(raw)}(?![A-Za-zÀ-ÿ])", flags=re.I)
        matches = list(raw_re.finditer(text))
        if not matches:
            # If the exact OCR surface is absent, do not block heading/custom-ish candidates solely on context.
            if raw_norm in PLACE_CONTEXT_AMBIGUOUS_COMMON_PLACE_NAMES:
                return False
            rows = self._candidate_rows(raw_norm)
            if province_anchor and not rows.empty:
                same_province = rows["province_guess"].map(norm_province_token).eq(norm_province_token(province_anchor)).any()
                return bool(same_province)
            return True

        has_capitalized_surface = False
        has_location_marker_nearby = False
        has_only_object_context = True

        for match in matches:
            surface = match.group(0)
            has_capitalized_surface = has_capitalized_surface or bool(surface and surface[0].isupper())

            before = norm_token(text[max(0, match.start() - 70):match.start()])
            after = norm_token(text[match.end():min(len(text), match.end() + 70)])
            before_tokens = before.split()
            after_tokens = after.split()
            near_tokens = set(before_tokens[-5:] + after_tokens[:5])

            if near_tokens & PLACE_CONTEXT_LOCATION_MARKERS:
                has_location_marker_nearby = True

            object_context = bool(near_tokens & PLACE_CONTEXT_OBJECT_WORDS)
            article_object_context = bool(before_tokens[-2:] and before_tokens[-1:] and before_tokens[-1] in {"de", "het", "een"})
            if not object_context and not article_object_context:
                has_only_object_context = False

        if has_location_marker_nearby:
            return True

        # Lower-case-only matches are usually common nouns/adjectives found by the gazetteer scan.
        if not has_capitalized_surface:
            return False

        if has_only_object_context and raw_norm not in self._well_known_site_names():
            return False

        return True

    @staticmethod
    def _well_known_site_names() -> set:
        return set(SITE_PARENT_MAP) | set(STREET_PARENT_MAP)

    @staticmethod
    def _is_guarded_common_noun_place(raw: str) -> bool:
        return norm_token(raw) in PLACE_CONTEXT_COMMON_NOUN_PLACE_GUARD

    @staticmethod
    def _is_homonym_place(raw: str) -> bool:
        return norm_token(raw) in HOMONYM_QUARANTINE_TERMS

    def _homonym_place_status(self, raw: str, text: str) -> str:
        """
        Classify gazetteer terms that are also ordinary archaeological words.

        This is deliberately conservative: if the local context does not clearly
        introduce the term as a place, Python flags it for LLM review instead of
        silently accepting the gazetteer match.
        """
        raw_norm = norm_token(raw)
        if raw_norm not in HOMONYM_QUARANTINE_TERMS:
            return "not_homonym"
        if not text:
            return "unclear"
        text_norm = norm_token(text)
        if not text_norm:
            return "unclear"
        if raw_norm == "huizen" and re.search(
            r"\b(?:huisplattegrond(?:en)?|(?:twee|drie|vier|vijf|zes|zeven|acht|negen|tien)\s+huizen|huizen\s+met|resten\s+van\s+huizen)\b",
            text_norm,
        ):
            return "feature_context"
        if self._has_strict_homonym_place_context(raw, text):
            return "place_context"
        raw_re = rf"\b{re.escape(raw_norm)}\b"
        matches = list(re.finditer(raw_re, text_norm))
        if not matches:
            return "unclear"
        for match in matches:
            window = text_norm[max(0, match.start() - 120):min(len(text_norm), match.end() + 120)]
            if HOMONYM_FEATURE_CONTEXT_RE.search(window):
                return "feature_context"
        return "unclear"

    @staticmethod
    def _has_strict_homonym_place_context(raw: str, text: str) -> bool:
        """
        Stricter than the generic place context. Bare "in Huizen" is too risky
        for homonyms because OCR/grammar may also produce "in huizen" as a
        common noun. Require explicit settlement wording or a clear toponymic
        construction.
        """
        raw_norm = norm_token(raw)
        text_norm = norm_token(text)
        if not raw_norm or not text_norm:
            return False
        if raw_norm == "huizen":
            huizen_patterns = [
                r"\bte\s+huizen\b",
                r"\b(?:gemeente|gem|plaats|stad|dorp|provincie|prov|opgraving(?:en)?)\.?\s+(?:te\s+)?huizen\b",
                r"\bhuizen\s*,?\s*(?:gemeente|gem|provincie|prov|dorp|stad)\b",
                r"\bhuizen\s*\((?:n\.?h\.?|noord[-\s]?holland|nh)\)",
            ]
            return any(re.search(pattern, text_norm) for pattern in huizen_patterns)
        patterns = [
            rf"\b(?:te|bij|nabij|onder|rond|binnen)\s+(?:de\s+|het\s+|den\s+|ter\s+|ten\s+)?{re.escape(raw_norm)}\b",
            rf"\b(?:gemeente|gem|plaats|stad|dorp|provincie|prov|opgraving(?:en)?)\.?\s+(?:te\s+|bij\s+|in\s+)?{re.escape(raw_norm)}\b",
            rf"\b{re.escape(raw_norm)}\s*,?\s*(?:gemeente|gem|provincie|prov|dorp|stad)\b",
            rf"\btussen\b.{0,80}\b{re.escape(raw_norm)}\b",
            rf"\b{re.escape(raw_norm)}\s*\((?:n\.?h\.?|noord[-\s]?holland|zh|z\.?h\.?|zuid[-\s]?holland)\)",
        ]
        return any(re.search(pattern, text_norm) for pattern in patterns)

    @staticmethod
    def _is_unmapped_street_like(raw: str) -> bool:
        raw_norm = norm_token(raw)
        if not raw_norm or raw_norm in STREET_PARENT_MAP or raw_norm in SITE_PARENT_MAP:
            return False
        tokens = raw_norm.split()
        if any(tok in STREET_AS_ALT_SUFFIXES for tok in tokens):
            return True
        compact = re.sub(r"[^a-z0-9]+", "", raw_norm)
        return any(compact.endswith(suffix) and len(compact) > len(suffix) + 3 for suffix in STREET_AS_ALT_SUFFIXES)

    @staticmethod
    def _has_explicit_place_context(raw: str, text: str) -> bool:
        raw_norm = norm_token(raw)
        text_norm = norm_token(text)
        if not raw_norm or not text_norm:
            return False
        strong_patterns = [
            rf"\b(?:te|bij|in|nabij|onder|rond|binnen)\s+(?:de\s+|het\s+|den\s+|ter\s+|ten\s+)?{re.escape(raw_norm)}\b",
            rf"\b(?:gemeente|gem|plaats|stad|dorp|provincie|prov|opgraving(?:en)?)\.?\s+(?:te\s+|bij\s+|in\s+)?{re.escape(raw_norm)}\b",
            rf"\b{re.escape(raw_norm)}\s*,?\s*(?:gemeente|gem|provincie|prov|dorp|stad)\b",
        ]
        return any(re.search(pattern, text_norm) for pattern in strong_patterns)

    def _candidate_rows(self, raw_norm: str) -> pd.DataFrame:
        df = self.refs.gazetteer
        matches = df[(df["name_norm"] == raw_norm) | (df["asciiname_norm"] == raw_norm)]
        if matches.empty:
            alt_mask = df["alt_list"].map(lambda xs: raw_norm in {norm_token(x) for x in xs})
            matches = df[alt_mask]
        return matches

    def _exact(self, raw: str, province_anchor: str = "") -> Optional[PlaceResolution]:
        raw_norm = norm_token(raw)
        matches = self._candidate_rows(raw_norm)
        if matches.empty:
            return None

        if province_anchor:
            province_matches = matches[matches["province_guess"].map(norm_province_token) == norm_province_token(province_anchor)]
            if not province_matches.empty:
                row = province_matches.iloc[0]
                return PlaceResolution(
                    place_raw=raw,
                    place_normalised=safe_str(row.get("name")),
                    alt_place_name=safe_str(row.get("asciiname")),
                    province_normalised=safe_str(row.get("province_guess") or province_anchor),
                    province_source="exact_province_prioritized",
                    resolution_type="exact_place",
                    ambiguous=False,
                )

        province_values = dedupe_keep_order([safe_str(p) for p in matches["province_guess"].tolist() if safe_str(p)])
        if not province_anchor and len(province_values) > 1:
            row = matches.iloc[0]
            return PlaceResolution(
                place_raw=raw,
                place_normalised=safe_str(row.get("name")),
                alt_place_name=safe_str(row.get("asciiname")),
                province_normalised="",
                province_source="exact_ambiguous_review",
                resolution_type="exact_place",
                ambiguous=True,
            )

        row = matches.iloc[0]
        gazetteer_province = safe_str(row.get("province_guess"))
        ambiguous = bool(province_anchor and gazetteer_province and norm_province_token(gazetteer_province) != norm_province_token(province_anchor))
        return PlaceResolution(
            place_raw=raw,
            place_normalised=safe_str(row.get("name")),
            alt_place_name=safe_str(row.get("asciiname")),
            province_normalised=gazetteer_province or province_anchor,
            province_source="exact_conflict_review" if ambiguous else "exact",
            resolution_type="exact_place",
            ambiguous=ambiguous,
        )

    def _fuzzy(self, raw: str, province_anchor: str = "") -> Optional[PlaceResolution]:
        raw_norm = norm_token(raw)
        df = self.refs.gazetteer.copy()
        if province_anchor:
            province_df = df[df["province_guess"].map(norm_province_token) == norm_province_token(province_anchor)]
            if not province_df.empty:
                df = province_df
        names_norm = df["name_norm"].dropna().astype(str).tolist()
        close = get_close_matches(raw_norm, names_norm, n=1, cutoff=self.threshold)
        if not close:
            return None
        row = df[df["name_norm"] == close[0]].head(1)
        if row.empty:
            return None
        r = row.iloc[0]
        return PlaceResolution(
            place_raw=raw,
            place_normalised=safe_str(r.get("name")),
            alt_place_name=safe_str(r.get("asciiname")),
            province_normalised=safe_str(r.get("province_guess") or province_anchor),
            province_source="fuzzy",
            resolution_type="fuzzy_place",
            fuzzy_score=sequence_ratio(raw_norm, close[0]),
        )


# =========================
# ABR LOOKUP MAPPING
# =========================
# Purpose:
# LookupMapper translates raw text terms into standardised ABR labels and codes.
#
# What it does:
# - Tries exact, singularised, fuzzy and semantic matching.
# - Returns a MatchResult with raw term, label, code, source and evidence quote.
# - Keeps unmapped terms in logs so aliases can be added later without guessing.
#
# Why this is useful:
# The output must remain uniform across documents. ABR mapping turns free text into
# a controlled table that can be compared between PDFs.

class LookupMapper:
    def __init__(self, lookup: Dict[str, Dict[str, Any]], threshold: float = 0.93, archaeobertje: Optional[ArcheoBERTjeHelper] = None):
        self.lookup = lookup
        self.threshold = threshold
        self.aliases = list(lookup.keys())
        self.archaeobertje = archaeobertje

    def map_many(self, raw_terms: List[str], text: str, kind: str = "term") -> List[MatchResult]:
        cleaned_terms = [t for t in dedupe_keep_order(raw_terms) if self._valid_term_candidate(t)]
        return [self.map_one(raw, text, kind) for raw in cleaned_terms]

    def _valid_term_candidate(self, raw: str) -> bool:
        raw_norm = norm_token(raw)
        if not raw_norm:
            return False
        if len(raw_norm) < 4 or len(raw_norm.split()) > 4:
            return False
        if any(tok in CONTEXT_WORDS_REJECT for tok in raw_norm.split()):
            return False
        return True

    def map_one(self, raw: str, text: str, kind: str = "term") -> MatchResult:
        raw_norm = norm_token(raw)
        if raw_norm in self.lookup:
            payload = self.lookup[raw_norm]
            return MatchResult(raw, payload["label"], safe_str(payload["abr_code"]), "exact", raw, extract_evidence_quote(text, raw))
        singular = singularize_dutch(raw_norm)
        if singular in self.lookup:
            payload = self.lookup[singular]
            return MatchResult(raw, payload["label"], safe_str(payload["abr_code"]), "alt", raw, extract_evidence_quote(text, raw))
        close = get_close_matches(raw_norm, self.aliases, n=1, cutoff=self.threshold)
        if close:
            payload = self.lookup[close[0]]
            return MatchResult(raw, payload["label"], safe_str(payload["abr_code"]), "fuzzy", raw, extract_evidence_quote(text, raw))
        if self.archaeobertje is not None and self.archaeobertje.available and kind in {"feature", "finding", "material"}:
            semantic = self.archaeobertje.semantic_match(raw, kind)
            if semantic is not None:
                semantic.quote = extract_evidence_quote(text, raw)
                return semantic
        return MatchResult(raw, "", "", "unmapped", raw, extract_evidence_quote(text, raw))


# =========================
# DATING AND LOCAL DATING PASS
# =========================
# Purpose:
# DatingResolver determines the archaeological period and ABR period code.
#
# What it does:
# - First searches for explicit period labels from the references.
# - Then uses local context around finds, structures and materials to interpret
#   years and centuries.
# - Blocks modern event years such as excavation, restoration or publication years
#   from being interpreted as archaeological dating.
# - Supports Onbekend/XXX when no safe dating is available.
#
# Why this is useful:
# Dating must be strict: no dating is better than treating a modern fieldwork year
# as an archaeological period. At the same time, the local pass can recover useful
# 11th-, 12th- and 13th-century evidence from the immediate archaeological context.

class DatingResolver:
    def __init__(self, refs: ReferenceStore):
        self.refs = refs

    def resolve(self, dating_candidates: List[str], text: str) -> MatchResult:
        if is_bibliographic_context(text):
            return MatchResult(source="bibliographic_context_uncertain")
        text_norm = norm_token(text)
        for rule in sorted(self.refs.period_rules, key=lambda x: len(x["label"]), reverse=True):
            if rule["label_norm"] and self._period_label_matches(rule["label"], rule["label_norm"], text, text_norm):
                label = rule["label"]
                return MatchResult(label, label, self._period_code(label), "exact_period_label", label, extract_evidence_quote(text, label))
        custom = self._resolve_custom_dating_alias(text)
        if custom.label:
            return custom
        for candidate in dating_candidates:
            c_norm = norm_token(candidate)
            for rule in self.refs.period_rules:
                if c_norm == rule["label_norm"] and not self._is_relative_time_false_positive(rule["label"], candidate, text):
                    label = rule["label"]
                    return MatchResult(label, label, self._period_code(label), "candidate_period_label", label, extract_evidence_quote(text, label))
        return MatchResult(source="uncertain")

    def _resolve_custom_dating_alias(self, text: str) -> MatchResult:
        if not text or not self.refs.custom_dating_aliases:
            return MatchResult(source="custom_dating_alias_uncertain")
        for row in self.refs.custom_dating_aliases:
            pattern = safe_str(row.get("pattern"))
            if not pattern:
                continue
            m = re.search(pattern, text, flags=re.I)
            if not m:
                continue
            label = safe_str(row.get("dating_label"))
            code = safe_str(row.get("dating_code")) or self._period_code(label)
            raw = m.group(0)
            return MatchResult(
                raw=raw,
                label=label,
                abr_code=code,
                source=safe_str(row.get("source")) or "custom_dating_alias",
                evidence=raw,
                quote=extract_evidence_quote(text, raw),
            )
        return MatchResult(source="custom_dating_alias_uncertain")

    def resolve_local_context(self, anchor_terms: List[str], text: str) -> MatchResult:
        """
        Second SLM dating pass: only infer dating from text windows around already
        detected artefacts, structures, or materials. This keeps the pipeline
        document-agnostic and avoids turning every loose date into a record.
        """
        windows = self._local_windows(anchor_terms, text)
        if not windows:
            return MatchResult(source="local_context_unavailable")

        for anchor, window in windows:
            if is_bibliographic_context(window):
                continue
            explicit = self._explicit_period_in_text(window, "local_context_period")
            if explicit.label:
                explicit.quote = window
                explicit.evidence = f"{explicit.evidence}; anchor={anchor}" if explicit.evidence else f"anchor={anchor}"
                return explicit

        for anchor, window in windows:
            if is_bibliographic_context(window):
                continue
            inferred = self._infer_century_or_year(window, anchor)
            if inferred.label:
                return inferred

        full_text_year = self._infer_year_from_full_text(anchor_terms, text)
        if full_text_year.label:
            return full_text_year

        return MatchResult(source="local_context_uncertain")

    def _explicit_period_in_text(self, text: str, source: str) -> MatchResult:
        text_norm = norm_token(text)
        for rule in sorted(self.refs.period_rules, key=lambda x: len(x["label"]), reverse=True):
            if rule["label_norm"] and self._period_label_matches(rule["label"], rule["label_norm"], text, text_norm):
                label = rule["label"]
                return MatchResult(
                    raw=label,
                    label=label,
                    abr_code=self._period_code(label),
                    source=source,
                    evidence=label,
                    quote=extract_evidence_quote(text, label),
                )
        return MatchResult(source=source)

    def _period_label_matches(self, label: str, label_norm: str, original_text: str, text_norm: str) -> bool:
        for match in re.finditer(rf"\b{re.escape(label_norm)}\b", text_norm):
            raw = match.group(0)
            if self._is_relative_time_false_positive(label, raw, original_text, match.start(), match.end(), text_norm):
                continue
            return True
        return False

    def _is_relative_time_false_positive(
        self,
        label: str,
        raw: str,
        text: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        text_norm: Optional[str] = None,
    ) -> bool:
        label_norm = norm_token(label)
        raw_norm = norm_token(raw)
        if label_norm not in RELATIVE_TIME_DATING_LABELS and raw_norm not in RELATIVE_TIME_DATING_LABELS:
            return False
        norm_text = text_norm if text_norm is not None else norm_token(text)
        if start_idx is None or end_idx is None:
            idx = norm_text.find(raw_norm)
            start_idx = idx if idx >= 0 else 0
            end_idx = start_idx + len(raw_norm)
        context = norm_text[max(0, start_idx - 80):min(len(norm_text), end_idx + 80)]
        return any(re.search(pattern, context, flags=re.I) for pattern in RELATIVE_TIME_FALSE_POSITIVE_PATTERNS)

    def _local_windows(self, anchor_terms: List[str], text: str) -> List[Tuple[str, str]]:
        text = norm_ws(text)
        if not text:
            return []
        text_lower = text.lower()
        windows = []
        seen = set()
        for anchor in dedupe_keep_order(anchor_terms):
            anchor = clean_candidate_phrase(anchor)
            anchor_norm = norm_token(anchor)
            if not anchor or len(anchor_norm) < 4 or anchor_norm in DATING_ANCHOR_REJECT or self.refs.is_rejected_term(anchor, "dating_anchor"):
                continue
            start_idx = 0
            anchor_lower = anchor.lower()
            while True:
                idx = text_lower.find(anchor_lower, start_idx)
                if idx < 0:
                    break
                start = max(0, idx - LOCAL_DATING_CONTEXT_WINDOW_CHARS // 2)
                end = min(len(text), idx + len(anchor) + LOCAL_DATING_CONTEXT_WINDOW_CHARS // 2)
                key = (start, end)
                if key not in seen:
                    seen.add(key)
                    windows.append((anchor, text[start:end].strip()))
                start_idx = idx + len(anchor_lower)
        return windows

    def _infer_century_or_year(self, window: str, anchor: str) -> MatchResult:
        for match in re.finditer(r"\b([1-9]|1[0-9]|20)\s*(?:e|de)\s+eeuw\b", window, flags=re.I):
            century = int(match.group(1))
            label = self._period_from_century(century)
            if label:
                raw = match.group(0)
                return MatchResult(
                    raw=raw,
                    label=label,
                    abr_code=self._period_code(label),
                    source="local_context_century",
                    evidence=f"{raw}; anchor={anchor}",
                    quote=extract_evidence_quote(window, raw),
                )

        for match in re.finditer(r"\b(1[0-9]{3}|[5-9][0-9]{2})\b", window):
            raw = match.group(1)
            if self._looks_like_postcode(window, match.end()):
                continue
            if not self._year_has_dating_context(window, match.start(), match.end()):
                continue
            label = self._period_from_year(int(raw))
            if label:
                return MatchResult(
                    raw=raw,
                    label=label,
                    abr_code=self._period_code(label),
                    source="local_context_year",
                    evidence=f"{raw}; anchor={anchor}",
                    quote=extract_evidence_quote(window, raw),
                )

        adjective = self._infer_period_adjective(window)
        if adjective.label:
            adjective.evidence = f"{adjective.evidence}; anchor={anchor}"
            return adjective

        return MatchResult(source="local_context_uncertain")

    def _infer_year_from_full_text(self, anchor_terms: List[str], text: str) -> MatchResult:
        text = norm_ws(text)
        if is_bibliographic_context(text):
            return MatchResult(source="bibliographic_context_uncertain")
        anchors = [clean_candidate_phrase(a) for a in dedupe_keep_order(anchor_terms)]
        anchors = [
            a for a in anchors
            if a and len(norm_token(a)) >= 4 and norm_token(a) not in DATING_ANCHOR_REJECT and not self.refs.is_rejected_term(a, "dating_anchor")
        ]
        if not text or not anchors:
            return MatchResult(source="local_context_uncertain")

        for match in re.finditer(r"\b(1[0-9]{3}|[5-9][0-9]{2})\b", text):
            raw = match.group(1)
            if self._looks_like_postcode(text, match.end()):
                continue
            if not self._year_has_dating_context(text, match.start(), match.end()):
                continue
            label = self._period_from_year(int(raw))
            if label:
                return MatchResult(
                    raw=raw,
                    label=label,
                    abr_code=self._period_code(label),
                    source="local_context_historical_year",
                    evidence=f"{raw}; anchor=record_context",
                    quote=extract_evidence_quote(text, raw),
                )

        return MatchResult(source="local_context_uncertain")

    def _infer_period_adjective(self, window: str) -> MatchResult:
        lower = window.lower()
        patterns = [
            (r"\blaat[-\s]?middeleeuws(?:e|en)?\b|\blate middeleeuwen\b", "Late Middeleeuwen"),
            (r"\bvroeg[-\s]?middeleeuws(?:e|en)?\b|\bvroege middeleeuwen\b", "Vroege Middeleeuwen"),
            (r"\bmiddeleeuws(?:e|en)?\b|\bmiddeleeuwen\b", "Middeleeuwen"),
            (r"\bromeinse tijd\b", "Romeinse tijd"),
        ]
        for pattern, label in patterns:
            m = re.search(pattern, lower, flags=re.I)
            if m:
                raw = m.group(0)
                return MatchResult(
                    raw=raw,
                    label=label,
                    abr_code=self._period_code(label),
                    source="local_context_period_adjective",
                    evidence=raw,
                    quote=extract_evidence_quote(window, raw),
                )
        return MatchResult(source="local_context_uncertain")

    def _period_from_century(self, century: int) -> str:
        if 1 <= century <= 4:
            return "Romeinse tijd"
        if 5 <= century <= 10:
            return "Vroege Middeleeuwen"
        if 11 <= century <= 15:
            return "Late Middeleeuwen"
        if 16 <= century <= 20:
            return "Nieuwe tijd"
        return ""

    def _period_from_year(self, year: int) -> str:
        if 12 <= year <= 450:
            return "Romeinse tijd"
        if 450 < year < 1050:
            return "Vroege Middeleeuwen"
        if 1050 <= year <= 1500:
            return "Late Middeleeuwen"
        if 1500 < year <= 1950:
            return "Nieuwe tijd"
        return ""

    def _period_code(self, label: str) -> str:
        label_norm = norm_token(label)
        if label_norm in self.refs.period_code_map:
            return self.refs.period_code_map[label_norm]
        alias_map = {
            "vroege middeleeuwen": "middeleeuwen vroeg",
            "late middeleeuwen": "middeleeuwen laat",
            "vroege bronstijd": "bronstijd vroeg",
            "midden bronstijd": "bronstijd midden",
            "late bronstijd": "bronstijd laat",
            "vroege romeinse tijd": "romeinse tijd vroeg",
            "midden romeinse tijd": "romeinse tijd midden",
            "late romeinse tijd": "romeinse tijd laat",
        }
        alias_norm = alias_map.get(label_norm, "")
        if alias_norm and alias_norm in self.refs.period_code_map:
            return self.refs.period_code_map[alias_norm]
        for rule in self.refs.period_rules:
            if norm_token(rule["label"]) == label_norm:
                return self.refs.period_code_map.get(rule["label_norm"], "")
        return ""

    @staticmethod
    def _looks_like_postcode(text: str, end_idx: int) -> bool:
        return bool(re.match(r"\s*[A-Za-z]{2}\b", text[end_idx:end_idx + 5]))

    @staticmethod
    def _year_has_dating_context(text: str, start_idx: int, end_idx: int) -> bool:
        context = text[max(0, start_idx - 45):min(len(text), end_idx + 45)].lower()
        raw_year = re.sub(r"\D", "", text[start_idx:end_idx])
        year = int(raw_year) if raw_year.isdigit() else 0
        if year >= 1800 and BIBLIOGRAPHIC_YEAR_CONTEXT_RE.search(context):
            return False
        if year >= 1800 and MODERN_EVENT_YEAR_CONTEXT_RE.search(context):
            return False
        return bool(re.search(r"\b(jaar|in|uit|dateer|dater|daterend|dateren|dateert|eeuw|akte|omschreven|ontving|middeleeuw|n\.\s*chr|na\s*chr)", context))


# =========================
# QUALITY SCORE AND REVIEW FLAGS
# =========================
# Purpose:
# These functions decide whether a record is strong enough and whether it needs
# review.
#
# What it does:
# - score_record_strength counts evidence for place, dating, features, findings,
#   materials and evidence quote.
# - derive_uncertainty_and_review converts risks into Uncertain_Flag, Review_Flag
#   and Review_Reason.
#
# Why this is useful:
# Not every tiny text fragment should become a final record. This layer keeps the
# output more semantic and identifies which records need LLM or human review.

def derive_uncertainty_and_review(
    section: Section,
    place: PlaceResolution,
    dating: MatchResult,
    features: List[MatchResult],
    findings: List[MatchResult],
    refs: Optional["ReferenceStore"] = None,
    text: str = "",
) -> Tuple[int, int, str]:
    risk, reasons = 0, []
    if section.segmentation_confidence <= 1:
        risk += 2
        reasons.append("LOW_SEGMENTATION")
    if place.resolution_type in {"exact_homonym_place", "unresolved_homonym"}:
        risk += 2
        reasons.append("HOMONYM_PLACE_REVIEW")
    if place.ambiguous:
        risk += 3
        reasons.append("AMBIGUOUS_PLACE")
    elif place.province_source in {"fuzzy", "exact_conflict_review"}:
        risk += 2
        reasons.append("PLACE_PROVINCE_CONFLICT")
    elif not place.place_normalised:
        risk += 3
        reasons.append("UNRESOLVED_PLACE")
    if not dating.label:
        risk += 2
        reasons.append("NO_DATING_EVIDENCE")
    if any(m.source == "unmapped" for m in features):
        risk += 2
        reasons.append("UNMAPPED_FEATURE")
    if any(m.source == "unmapped" for m in findings):
        risk += 2
        reasons.append("UNMAPPED_FINDING")

    labelled_findings = [m for m in findings if m.label or m.abr_code]
    bucket_for = refs.bucket_for_match if refs is not None else abr_bucket_for_match
    context_supported = refs.finding_context_supported if refs is not None else finding_context_supported
    unresolved_bucket_conflict = any(
        "bucket_conflict" in safe_str(m.source) and (m.label or m.abr_code)
        for m in findings
    ) or any(
        bucket_for(m, "finding") in {"feature", "material"}
        for m in labelled_findings
    )
    ambiguous_finding_still_clean = any(
        bucket_for(m, "finding") == "ambiguous_finding"
        and not context_supported(m, " ".join([m.quote, m.evidence, text]))
        for m in labelled_findings
    )
    ambiguous_finding_quarantined = any(
        "ambiguous_finding_quarantine" in safe_str(m.source)
        for m in findings
    )

    if unresolved_bucket_conflict:
        risk += 2
        reasons.append("ABR_BUCKET_CONFLICT")
    if ambiguous_finding_still_clean:
        risk += 2
        reasons.append("AMBIGUOUS_FINDING_REVIEW")
    if ambiguous_finding_quarantined:
        risk += 1
        reasons.append("AMBIGUOUS_FINDING_QUARANTINED")
    uncertain = 1 if risk > 0 else 0
    review_flag = 0 if risk == 0 else 1 if risk <= 2 else 2 if risk <= 4 else 3 if risk <= 6 else 4
    return uncertain, review_flag, ";".join(dedupe_keep_order(reasons))


def score_record_strength(place: PlaceResolution, dating: MatchResult, features: List[MatchResult], findings: List[MatchResult], materials: List[MatchResult], evidence_quote: str) -> int:
    score = 0
    if place.place_normalised:
        score += 2
    if place.province_source in {"exact_province_prioritized", "exact"}:
        score += 1
    if dating.label:
        score += 2
    score += min(2, sum(1 for m in features if m.label))
    score += min(2, sum(1 for m in findings if m.label))
    score += min(1, sum(1 for m in materials if m.label))
    if len(norm_ws(evidence_quote)) >= 80:
        score += 1
    return score


# =========================
# LLM SECOND READER
# =========================
# Purpose:
# LLMReviewer asks Claude to review suspicious records after local extraction.
#
# What it does:
# - Sends the record, evidence quote, chunk text and allowed labels to Claude.
# - Requests structured JSON with corrections, confidence and reasoning.
# - Applies only safe corrections automatically.
# - Protects good local output from broadening, for example LME is not blindly
#   replaced by XME and specific finds are not replaced by broad categories.
#
# Why this is useful:
# The LLM is strong at semantic checking, but can also overgeneralise. This layer
# uses Claude where it adds value while safeguards protect reliable local output.

class LLMReviewer:
    def __init__(self, cfg: RunConfig, refs: ReferenceStore):
        self.cfg = cfg
        self.refs = refs
        self.status_rows = []
        self.client = None
        if not cfg.enable_llm_review:
            self.status_rows.append({"component": "llm_review", "enabled": False, "message": "disabled"})
            return
        try:
            self.client = AnthropicJSONClient(
                api_key=cfg.anthropic_api_key,
                model=cfg.anthropic_model,
                api_version=cfg.anthropic_api_version,
            )
            self.status_rows.append({
                "component": "llm_review",
                "enabled": True,
                "model": cfg.anthropic_model,
                "message": "ok",
            })
        except Exception as e:
            self.status_rows.append({"component": "llm_review", "enabled": True, "message": str(e)})

    def available(self) -> bool:
        return self.client is not None

    def should_review(self, row: pd.Series) -> bool:
        if not self.available():
            return False
        if not self.cfg.llm_review_only_suspicious:
            return True
        review_flag = int(row.get("Review_Flag") or 0)
        province_source = safe_str(row.get("Province_Source"))
        dating_source = safe_str(row.get("Dating_Source"))
        dating_label = safe_str(row.get("Dating_Normalised_Label"))
        return (
            review_flag >= self.cfg.llm_review_min_review_flag
            or province_source in {"fuzzy", "exact_conflict_review", "unresolved"}
            or dating_source.startswith("local_context_")
            or self._low_segmentation_place_review_candidate(row)
            or not dating_label
        )

    @staticmethod
    def _low_segmentation_place_review_candidate(row: pd.Series) -> bool:
        """
        Keep Haiku usage targeted: review low-segmentation rows only when the
        current place looks like an internal site/building/heading rather than
        a secure settlement.
        """
        review_reason = safe_str(row.get("Review_Reason")).upper()
        if "LOW_SEGMENTATION" not in review_reason:
            return False
        place_text = norm_token(" ".join([
            safe_str(row.get("Place_Raw")),
            safe_str(row.get("Place_Normalised")),
            safe_str(row.get("Alt_Place_Name")),
        ]))
        if not place_text:
            return True
        return bool(re.search(
            r"\b(?:kerk|joriskerk|kapel|hof|stadhuis|museum|school|straat|plein|terrein|kasteel|slot)\b",
            place_text,
        ))

    def review_row(self, row: pd.Series, chunk_text: str) -> LLMReviewResult:
        if not self.should_review(row):
            return LLMReviewResult(reviewed=False)
        prompt = self._build_prompt(row, chunk_text)
        try:
            data = self.client.propose(prompt)
            result = self._parse_result(data)
            result.reviewed = True
            self._mark_unsafe_term_broadening(row, result)
            if (
                result.needs_correction
                and result.confidence >= self.cfg.llm_review_confidence_threshold
                and self.cfg.llm_review_auto_apply
                and self._auto_apply_allowed(row, result)
            ):
                result.auto_applied = 1
            return result
        except Exception as e:
            self.status_rows.append({
                "component": "llm_review_row",
                "record_id": safe_str(row.get("Record_ID")),
                "chunk_id": safe_str(row.get("Chunk_ID")),
                "message": str(e),
            })
            return LLMReviewResult(reviewed=False, needs_correction=False, correction_type="error", reason=str(e), confidence=0.0)

    def _build_prompt(self, row: pd.Series, chunk_text: str) -> str:
        evidence_text = " ".join([
            safe_str(row.get("Evidence_Quote")),
            safe_str(chunk_text),
            safe_str(row.get("Feature_Context_Raw")),
            safe_str(row.get("Findings_Raw")),
            safe_str(row.get("Material_Raw")),
        ])
        payload = {
            "record": {
                "place_raw": safe_str(row.get("Place_Raw")),
                "place_normalised": safe_str(row.get("Place_Normalised")),
                "province_normalised": safe_str(row.get("Province_Normalised")),
                "province_source": safe_str(row.get("Province_Source")),
                "place_resolution_type": safe_str(row.get("Place_Resolution_Type")),
                "dating_label": safe_str(row.get("Dating_Normalised_Label")),
                "feature_labels": safe_str(row.get("Feature_Context_ABR_Label")),
                "findings_labels": safe_str(row.get("Findings_Normal_ABR_Label")),
                "material_labels": safe_str(row.get("Material_Normalised")),
                "feature_mapping_source": safe_str(row.get("Feature_Mapping_Source")),
                "findings_mapping_source": safe_str(row.get("Findings_Mapping_Source")),
                "evidence_quote": safe_str(row.get("Evidence_Quote"))[:self.cfg.llm_review_max_text_chars],
                "chunk_text": safe_str(chunk_text)[:self.cfg.llm_review_max_text_chars],
                "review_flag": int(row.get("Review_Flag") or 0),
                "review_reason": safe_str(row.get("Review_Reason")),
                "homonym_place_candidate": self._homonym_place_candidate_for_review(row),
            },
            "allowed_provinces": self.refs.province_list(),
            "allowed_dating": self.refs.all_periods(),
            "allowed_features": self._prioritized_allowed_labels(
                "feature",
                row,
                evidence_text,
                "Feature_Context_ABR_Label",
                "Feature_Context_ABR_Code",
                self.refs.all_feature_labels(),
            ),
            "allowed_findings": self._prioritized_allowed_labels(
                "finding",
                row,
                evidence_text,
                "Findings_Normal_ABR_Label",
                "Findings_ABR_Code",
                self.refs.all_finding_labels(),
            ),
            "allowed_materials": self._prioritized_allowed_labels(
                "material",
                row,
                evidence_text,
                "Material_Normalised",
                "Final_Material_Codes",
                self.refs.all_material_labels(),
            ),
            "blocked_terms": self._blocked_terms_for_review(),
            "pipeline_conventions": {
                "custom_aliases_are_authoritative": True,
                "keep_specific_supported_slm_codes": True,
                "custom_or_local_dating_with_evidence_should_not_be_broadened": True,
                "homonym_places_require_context": True,
                "bucket_conflicts_require_evidence": True,
                "ambiguous_findings_are_quarantine_until_evidence_supported": True,
                "notes": [
                    "If the SLM has a concrete label/code supported by evidence, prefer retaining it over replacing it with a broader generic code.",
                    "Do not use labels/codes listed in blocked_terms, even if they appear in a broad ABR vocabulary.",
                    "In this project, locally inferred/custom dating rules are part of the thesis convention. You may flag disagreement, but avoid proposing a broader correction when the SLM dating has explicit evidence.",
                    "If place_resolution_type is exact_homonym_place or unresolved_homonym, decide from context whether the candidate is a real settlement/place name or an archaeological feature/common noun.",
                    "If Feature_Mapping_Source or Findings_Mapping_Source contains bucket_conflict or ambiguous_finding_quarantine, do not freely invent a better term. Decide only whether the supplied text explicitly supports retaining, moving, or releasing the quarantined term.",
                    "Ambiguous findings such as Ring, Haar, Spits, Band, Boek, Kern, Segment, Rechthoekig, type Haps and Rijnlands may only be retained as findings when nearby evidence includes words such as gevonden, aangetroffen, bronzen, ijzeren, fragment, voorwerp, vondst, beslag or sieraad.",
                ],
            },
        }
        return f"""
Validate this Dutch archaeological extraction record as a second reader.

You are NOT the primary extractor and you are NOT doing structural segmentation.
The SLM/Python pipeline has already made a record. Your job is to validate it.

Check:
- Is this a valid archaeological record, or likely OCR/noise/context leakage?
- Is the place correct for the evidence?
- Is the province correct for the place and evidence?
- Does the dating belong to this record's place/object/structure?
- Are feature/findings/material labels supported by the evidence?
- Is this record mixed with another nearby record?
- Should adjacent records probably be merged or split? Give advice only; Python decides.
- If a candidate place is also a Dutch common/archaeological word, use context to disambiguate it. For example, "Huizen" can be the municipality near Blaricum, but "drie huizen", "huizen met schuur" or "huisplattegronden" means buildings/features, not a place.
- If a term is marked as bucket_conflict or ambiguous_finding_quarantine, do not answer "what should this be?" freely. Only answer whether the supplied evidence explicitly supports it as feature/context, finding or material.
- If a feature/spoor/context term such as Gracht, Greppel, Waterput, Fundering, Kerkhof, Kerk, Kuil or Huisplattegrond appears under findings, prefer moving it to feature/context or flagging bucket_conflict instead of treating it as an artefact.
- Ceramic terms such as Pingsdorf, Paffrath, kogelpot, terra sigillata, terra nigra, Dragendorff, Niederbieber and steengoed should normally remain finding/ceramic evidence, not material-only labels.

Second-reader dating rules:
- If this is a valid archaeological record but no dating evidence is present in the evidence_quote or chunk_text, you may set corrected_dating_label to "Onbekend" and corrected_dating_code to "XXX".
- Only use "Onbekend" when you are confident there is no explicit or inferable archaeological period/year/century in the supplied text.
- Do not use relative words such as recent, latest, earlier, old, new, or "tijdje terug" as archaeological dating evidence.
- For multiple datings, separate labels and codes with semicolons, not pipes.
- Do not broaden a specific SLM dating into a less specific period. For example: keep Late Middeleeuwen/LME if the supplied evidence supports it; do not replace it with Middeleeuwen/XME unless the original evidence is clearly wrong.

Return strict JSON with keys:
reviewed
needs_correction
correction_type
reason
confidence
changed_fields
corrected_place_normalised
corrected_province_normalised
corrected_dating_label
corrected_dating_code
corrected_feature_labels
corrected_feature_codes
corrected_findings_labels
corrected_findings_codes
corrected_material_labels
corrected_material_codes
record_valid
noise_record
mixed_context
validation_flags
merge_split_advice
evidence_check

Important:
- Use only labels/codes from the allowed lists when proposing corrected labels/codes.
- The existing SLM labels/codes are deliberately included near the top of the allowed lists. Treat them as valid unless the supplied evidence clearly contradicts them.
- Do not replace a specific supported SLM term with a broader generic category. For example: keep Deksel/DEKSEL if "potdeksel" is in the evidence; do not replace it with Aardewerk, ondetermineerbaar/AW.
- Do not use any term listed in blocked_terms. These are known OCR/context/reference traps for this pipeline.
- Late Middeleeuwen, Vroege Middeleeuwen, Late Bronstijd and other medieval/prehistoric period labels may be valid if they appear in allowed_dating.
- If the SLM dating source is custom_dating_alias or local_context_* and the evidence_quote supports it, do not broaden it automatically. You may explain uncertainty in validation_flags instead.
- Province corrections are high risk. Only propose a province correction if the evidence clearly supports it.
- Homonym place corrections are high risk. Only accept a homonym as a place if the text contains explicit settlement/location context such as "te Huizen", "gemeente Huizen", "bij Huizen", "tussen Blaricum en Huizen", or equivalent. If it functions as a feature/common noun, keep place unresolved and suggest feature labels instead.
- ABR bucket corrections are high risk. Only propose corrected feature/findings/material labels when the proposed label/code is in the allowed lists and the evidence_quote or chunk_text explicitly supports that bucket.
- Do not release a quarantined ambiguous finding unless the evidence contains direct nearby support words such as gevonden, aangetroffen, bronzen, ijzeren, fragment, voorwerp, vondst, beslag, sieraad, aardewerk or scherf.
- If you are uncertain, set needs_correction=true but keep confidence below the auto-apply threshold.
- Do not wrap the response in markdown fences.

Payload:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    def _prioritized_allowed_labels(
        self,
        kind: str,
        row: pd.Series,
        evidence_text: str,
        label_col: str,
        code_col: str,
        all_candidates: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        prioritized = []
        prioritized.extend(self._row_label_code_pairs(row, label_col, code_col))
        prioritized.extend(self._custom_alias_pairs_in_text(kind, evidence_text))

        out = []
        seen = set()
        for item in prioritized + all_candidates:
            label = safe_str(item.get("label"))
            code = safe_str(item.get("code"))
            if not label:
                continue
            if self.refs.is_rejected_term(label, kind) or self.refs.is_rejected_term(code, kind):
                continue
            key = (norm_token(label), code.upper())
            if key in seen:
                continue
            seen.add(key)
            out.append({"label": label, "code": code})

        limit = max(self.cfg.llm_review_max_candidates, len(prioritized))
        return out[:limit]

    def _blocked_terms_for_review(self) -> List[Dict[str, str]]:
        out = []
        reject_df = self.refs.custom_alias_tables.get("reject_terms")
        if reject_df is None:
            return out
        for _, row in reject_df.iterrows():
            if not self.refs._row_enabled(row):
                continue
            term = safe_str(row.get("term"))
            if not term:
                continue
            out.append({
                "term": term,
                "kind": safe_str(row.get("kind")),
                "reason": safe_str(row.get("reason")),
            })
        return out

    @staticmethod
    def _homonym_place_candidate_for_review(row: pd.Series) -> str:
        values = [
            safe_str(row.get("Place_Raw")),
            safe_str(row.get("Place_Normalised")),
            safe_str(row.get("Alt_Place_Name")),
        ]
        for value in values:
            if norm_token(value) in HOMONYM_QUARANTINE_TERMS:
                return value
        return ""

    @staticmethod
    def _row_label_code_pairs(row: pd.Series, label_col: str, code_col: str) -> List[Dict[str, str]]:
        labels = split_semicolon_values(row.get(label_col))
        codes = split_semicolon_values(row.get(code_col))
        pairs = []
        for idx, label in enumerate(labels):
            code = codes[idx] if idx < len(codes) else ""
            pairs.append({"label": label, "code": code})
        return pairs

    def _custom_alias_pairs_in_text(self, kind: str, evidence_text: str) -> List[Dict[str, str]]:
        text_norm = norm_token(evidence_text)
        if not text_norm:
            return []
        pairs = []
        term_aliases = self.refs.custom_alias_tables.get("term_aliases")
        if term_aliases is not None:
            for _, alias_row in term_aliases.iterrows():
                if not self.refs._row_enabled(alias_row) or norm_token(alias_row.get("kind")) != kind:
                    continue
                alias = norm_token(alias_row.get("alias"))
                if alias and re.search(rf"\b{re.escape(alias)}\b", text_norm):
                    pairs.append({
                        "label": safe_str(alias_row.get("canonical_label")),
                        "code": safe_str(alias_row.get("abr_code")),
                    })

        for term, payload in self.refs.custom_field_type_overrides.items():
            if payload.get("kind") != kind:
                continue
            if term and re.search(rf"\b{re.escape(term)}\b", text_norm):
                pairs.append({
                    "label": safe_str(payload.get("canonical_label")),
                    "code": safe_str(payload.get("abr_code")),
                })
        return pairs

    def _parse_result(self, data: Dict[str, Any]) -> LLMReviewResult:
        return LLMReviewResult(
            reviewed=bool(data.get("reviewed", True)),
            needs_correction=bool(data.get("needs_correction", False)),
            correction_type=safe_str(data.get("correction_type")),
            reason=safe_str(data.get("reason")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            changed_fields=safe_str(data.get("changed_fields")),
            corrected_place_normalised=safe_str(data.get("corrected_place_normalised")),
            corrected_province_normalised=safe_str(data.get("corrected_province_normalised")),
            corrected_dating_label=normalise_multivalue_text(data.get("corrected_dating_label")),
            corrected_dating_code=normalise_multivalue_text(data.get("corrected_dating_code")),
            corrected_feature_labels=join_semicolon(split_semicolon_values(data.get("corrected_feature_labels"))),
            corrected_feature_codes=join_semicolon(split_semicolon_values(data.get("corrected_feature_codes"))),
            corrected_findings_labels=join_semicolon(split_semicolon_values(data.get("corrected_findings_labels"))),
            corrected_findings_codes=join_semicolon(split_semicolon_values(data.get("corrected_findings_codes"))),
            corrected_material_labels=join_semicolon(split_semicolon_values(data.get("corrected_material_labels"))),
            corrected_material_codes=join_semicolon(split_semicolon_values(data.get("corrected_material_codes"))),
            record_valid=safe_str(data.get("record_valid")),
            noise_record=safe_str(data.get("noise_record")),
            mixed_context=safe_str(data.get("mixed_context")),
            validation_flags=normalise_llm_list(data.get("validation_flags")),
            merge_split_advice=safe_str(data.get("merge_split_advice")),
            evidence_check=safe_str(data.get("evidence_check")),
        )

    def _auto_apply_allowed(self, row: pd.Series, result: LLMReviewResult) -> bool:
        if result.confidence < 0.90:
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_LOW_CONFIDENCE",
                "Haiku auto-apply requires confidence >= 0.90.",
            )
        if safe_str(result.noise_record).lower() == "true" or safe_str(result.mixed_context).lower() == "true":
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_NOISE_OR_MIXED_CONTEXT",
                "LLM marked record as noise or mixed context.",
            )
        if self._multi_period_expansion(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_MULTI_PERIOD_EXPANSION",
                "multi-period dating expansion should be flagged, not auto-applied.",
            )
        if result.corrected_place_normalised and not self._place_correction_allowed(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_UNVERIFIED_PLACE",
                "corrected place is not supported by gazetteer or fixed parent/place maps.",
            )
        if result.corrected_dating_code and not self._dating_correction_allowed(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_DATING_BROADENING",
                "corrected dating is less specific than the SLM dating supported by local evidence.",
            )
        if result.corrected_province_normalised and not self._province_correction_allowed(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_PROVINCE_CONFLICT",
                "corrected province conflicts with gazetteer/place evidence.",
            )
        if self._final_findings_have_unsupported_auto_apply_guard_terms(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_AMBIGUOUS_FINDING_UNSUPPORTED",
                "final findings contain a rulebook-guarded ambiguous term without direct evidence support.",
            )
        if not self._has_any_safe_auto_apply_field(row, result):
            return self._block_auto_apply(
                result,
                "AUTO_APPLY_BLOCKED_NO_SAFE_FIELD",
                "no corrected field passed the auto-apply safety checks.",
            )
        return True

    def auto_apply_field_allowed(self, row: pd.Series, result: LLMReviewResult, field: str) -> bool:
        """
        Field-level safety gate. A safe place/province fix should not be blocked
        merely because the same LLM response proposes a multi-period dating that
        must remain review-only.
        """
        if not (result.needs_correction and self.cfg.llm_review_auto_apply):
            return False
        if result.confidence < 0.90:
            return False
        if safe_str(result.noise_record).lower() == "true" or safe_str(result.mixed_context).lower() == "true":
            return False
        if field == "place":
            return bool(result.corrected_place_normalised) and self._place_correction_allowed(row, result)
        if field == "province":
            return bool(result.corrected_province_normalised) and self._province_correction_allowed(row, result)
        if field == "dating":
            return (
                bool(result.corrected_dating_label or result.corrected_dating_code)
                and not self._multi_period_expansion(row, result)
                and self._dating_correction_allowed(row, result)
            )
        if field in {"feature", "findings", "material"}:
            has_value = {
                "feature": bool(result.corrected_feature_labels or result.corrected_feature_codes),
                "findings": bool(result.corrected_findings_labels or result.corrected_findings_codes),
                "material": bool(result.corrected_material_labels or result.corrected_material_codes),
            }[field]
            if field == "findings" and self._final_findings_have_unsupported_auto_apply_guard_terms(row, result):
                return False
            return has_value and self.term_field_auto_apply_allowed(row, result, field)
        return False

    def _row_rulebook_evidence_text(self, row: pd.Series, result: Optional[LLMReviewResult] = None) -> str:
        parts = [
            safe_str(row.get("Evidence_Quote")),
            safe_str(row.get("Dating_Evidence")),
            safe_str(row.get("Feature_Context_Raw")),
            safe_str(row.get("Findings_Raw")),
            safe_str(row.get("Material_Raw")),
            safe_str(row.get("LLM_Evidence_Check")),
            safe_str(row.get("LLM_Reason")),
        ]
        if result is not None:
            parts.extend([result.evidence_check, result.reason])
        return " ".join(parts)

    def _guarded_findings_supported(self, row: pd.Series, labels: Any, result: Optional[LLMReviewResult] = None) -> bool:
        label_values = split_semicolon_values(labels)
        if not label_values:
            return True
        text = self._row_rulebook_evidence_text(row, result)
        return self.refs.auto_apply_guard_terms_supported("findings", join_semicolon(label_values), text)

    def _final_findings_have_unsupported_auto_apply_guard_terms(self, row: pd.Series, result: LLMReviewResult) -> bool:
        labels = (
            result.corrected_findings_labels
            or row.get("Final_Findings_Normal_ABR_Label")
            or row.get("Findings_Normal_ABR_Label")
        )
        return not self._guarded_findings_supported(row, labels, result)

    def _mark_unsafe_term_broadening(self, row: pd.Series, result: LLMReviewResult) -> None:
        for field in ["feature", "findings", "material"]:
            if not self.term_field_auto_apply_allowed(row, result, field):
                result.validation_flags = join_semicolon(
                    split_semicolon_values(result.validation_flags)
                    + [f"AUTO_APPLY_BLOCKED_{field.upper()}_BROADENING"]
                )
                result.reason = norm_ws(
                    f"{result.reason} AUTO_APPLY_BLOCKED_FIELD: LLM {field} correction would replace a supported specific SLM code with a broader/generic category."
                )

    def term_field_auto_apply_allowed(self, row: pd.Series, result: LLMReviewResult, field: str) -> bool:
        field_map = {
            "feature": (
                "Final_Feature_Context_ABR_Label",
                "Final_Feature_Context_ABR_Code",
                result.corrected_feature_labels,
                result.corrected_feature_codes,
                "Feature_Context_Raw",
            ),
            "findings": (
                "Final_Findings_Normal_ABR_Label",
                "Final_Findings_ABR_Code",
                result.corrected_findings_labels,
                result.corrected_findings_codes,
                "Findings_Raw",
            ),
            "material": (
                "Final_Material_Normalised",
                "Final_Material_Codes",
                result.corrected_material_labels,
                result.corrected_material_codes,
                "Material_Raw",
            ),
        }
        if field not in field_map:
            return True

        label_col, code_col, corrected_labels, corrected_codes, raw_col = field_map[field]
        if (corrected_labels or corrected_codes) and not self._corrected_terms_exist_in_lookup(field, corrected_labels, corrected_codes):
            result.validation_flags = join_semicolon(
                split_semicolon_values(result.validation_flags)
                + [f"AUTO_APPLY_BLOCKED_{field.upper()}_NOT_IN_LOOKUP"]
            )
            return False
        if (corrected_labels or corrected_codes) and not self._corrected_terms_supported_by_evidence(row, field, corrected_labels, corrected_codes):
            result.validation_flags = join_semicolon(
                split_semicolon_values(result.validation_flags)
                + [f"AUTO_APPLY_BLOCKED_{field.upper()}_NO_EXPLICIT_EVIDENCE"]
            )
            return False
        if field == "findings" and not self._guarded_findings_supported(row, corrected_labels or row.get(label_col)):
            result.validation_flags = join_semicolon(
                split_semicolon_values(result.validation_flags)
                + ["AUTO_APPLY_BLOCKED_FINDINGS_RULEBOOK_GUARD"]
            )
            return False

        corrected_code_values = self._term_code_values(corrected_codes)
        original_code_values = self._term_code_values(row.get(code_col))
        if not corrected_code_values or not original_code_values:
            return True

        corrected_set = set(corrected_code_values)
        original_set = set(original_code_values)
        if original_set.issubset(corrected_set):
            return True

        if not corrected_set & BROAD_GENERIC_TERM_CODES:
            return True

        original_terms = (
            split_semicolon_values(row.get(raw_col))
            + split_semicolon_values(row.get(label_col))
            + original_code_values
        )
        if self._row_evidence_supports_terms(row, original_terms):
            return False
        return True

    def _corrected_terms_exist_in_lookup(self, field: str, labels: str, codes: str) -> bool:
        lookup = {
            "feature": self.refs.feature_lookup,
            "findings": self.refs.findings_lookup,
            "material": self.refs.material_lookup,
        }.get(field, {})
        lookup_labels = {norm_token(payload.get("label")) for payload in lookup.values()}
        lookup_codes = {safe_str(payload.get("abr_code")).upper() for payload in lookup.values()}

        for label in split_semicolon_values(labels):
            if norm_token(label) and norm_token(label) not in lookup_labels:
                return False
        for code in split_semicolon_values(codes):
            code_norm = safe_str(code).upper()
            if code_norm and code_norm not in lookup_codes:
                return False
        return True

    def _corrected_terms_supported_by_evidence(self, row: pd.Series, field: str, labels: str, codes: str) -> bool:
        terms = split_semicolon_values(labels)
        if not terms:
            return bool(split_semicolon_values(codes))
        text = " ".join([
            safe_str(row.get("Evidence_Quote")),
            safe_str(row.get("Dating_Evidence")),
            safe_str(row.get("Feature_Context_Raw")),
            safe_str(row.get("Findings_Raw")),
            safe_str(row.get("Material_Raw")),
            safe_str(row.get("LLM_Evidence_Check")),
        ])
        text_norm = norm_token(text)
        if not text_norm:
            return False

        if field == "findings":
            for term in terms:
                if self.refs.is_ambiguous_finding_term(term) and not self._evidence_has_ambiguous_finding_support(row, term):
                    return False

        for term in terms:
            term_norm = norm_token(term)
            if len(term_norm) < 4:
                continue
            if re.search(rf"\b{re.escape(term_norm)}\b", text_norm):
                return True
            if self._lookup_alias_in_evidence(field, term_norm, text_norm):
                return True
        return False

    def _lookup_alias_in_evidence(self, field: str, label_norm: str, text_norm: str) -> bool:
        lookup = {
            "feature": self.refs.feature_lookup,
            "findings": self.refs.findings_lookup,
            "material": self.refs.material_lookup,
        }.get(field, {})
        aliases = [alias for alias, payload in lookup.items() if norm_token(payload.get("label")) == label_norm]
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", text_norm):
                return True
        return False

    def _evidence_has_ambiguous_finding_support(self, row: pd.Series, term: str) -> bool:
        text = " ".join([
            safe_str(row.get("Evidence_Quote")),
            safe_str(row.get("Findings_Raw")),
            safe_str(row.get("LLM_Evidence_Check")),
        ])
        return self.refs.finding_term_supported_in_text(term, text)

    def _has_any_safe_auto_apply_field(self, row: pd.Series, result: LLMReviewResult) -> bool:
        if result.corrected_place_normalised and self._place_correction_allowed(row, result):
            return True
        if result.corrected_province_normalised and self._province_correction_allowed(row, result):
            return True
        if (result.corrected_dating_label or result.corrected_dating_code) and self._dating_correction_allowed(row, result):
            return True
        if (result.corrected_feature_labels or result.corrected_feature_codes) and self.term_field_auto_apply_allowed(row, result, "feature"):
            return True
        if (result.corrected_findings_labels or result.corrected_findings_codes) and self.term_field_auto_apply_allowed(row, result, "findings"):
            return True
        if (result.corrected_material_labels or result.corrected_material_codes) and self.term_field_auto_apply_allowed(row, result, "material"):
            return True
        return False

    @staticmethod
    def _multi_period_expansion(row: pd.Series, result: LLMReviewResult) -> bool:
        corrected_labels = split_semicolon_values(result.corrected_dating_label)
        corrected_codes = split_semicolon_values(result.corrected_dating_code)
        if max(len(corrected_labels), len(corrected_codes)) <= 1:
            return False
        original_labels = split_semicolon_values(
            row.get("Final_Dating_Normalised_Label") or row.get("Dating_Normalised_Label")
        )
        original_codes = split_semicolon_values(
            row.get("Final_Dating_Normalised_ABR_Code") or row.get("Dating_Normalised_ABR_Code")
        )
        return max(len(original_labels), len(original_codes)) <= 1

    def _place_correction_allowed(self, row: pd.Series, result: LLMReviewResult) -> bool:
        corrected_place = safe_str(result.corrected_place_normalised)
        if not corrected_place:
            return True
        corrected_norm = norm_token(corrected_place)
        if corrected_norm in HOMONYM_QUARANTINE_TERMS and not self._row_text_has_homonym_place_context(row, corrected_place):
            return False
        if corrected_norm in {norm_token(payload["parent"]) for payload in SITE_PARENT_MAP.values()}:
            return True
        if corrected_norm in {norm_token(payload["parent"]) for payload in STREET_PARENT_MAP.values()}:
            return True
        if corrected_norm in {norm_token(payload.get("place")) for payload in self.refs.custom_place_aliases.values()}:
            return True
        return not self._gazetteer_rows_for_place(corrected_place).empty

    @staticmethod
    def _row_text_has_homonym_place_context(row: pd.Series, place: str) -> bool:
        place_norm = norm_token(place)
        if not place_norm:
            return False
        text_norm = norm_token(" ".join([
            safe_str(row.get("Evidence_Quote")),
            safe_str(row.get("Review_Reason")),
            safe_str(row.get("LLM_Evidence_Check")),
            safe_str(row.get("LLM_Reason")),
        ]))
        if not text_norm:
            return False
        patterns = [
            rf"\b(?:te|bij|in|nabij|onder|rond|binnen)\s+(?:de\s+|het\s+|den\s+|ter\s+|ten\s+)?{re.escape(place_norm)}\b",
            rf"\b(?:gemeente|gem|plaats|stad|dorp|provincie|prov|opgraving(?:en)?)\.?\s+(?:te\s+|bij\s+|in\s+)?{re.escape(place_norm)}\b",
            rf"\b{re.escape(place_norm)}\s*,?\s*(?:gemeente|gem|provincie|prov|dorp|stad)\b",
            rf"\btussen\b.{0,80}\b{re.escape(place_norm)}\b",
            rf"\b{re.escape(place_norm)}\s*\((?:n\.?h\.?|noord[-\s]?holland|zh|z\.?h\.?|zuid[-\s]?holland)\)",
        ]
        return any(re.search(pattern, text_norm) for pattern in patterns)

    @staticmethod
    def _term_code_values(value: Any) -> List[str]:
        return [safe_str(code).upper() for code in split_semicolon_values(value) if safe_str(code)]

    @staticmethod
    def _row_evidence_supports_terms(row: pd.Series, terms: List[str]) -> bool:
        evidence = norm_token(" ".join([
            safe_str(row.get("Evidence_Quote")),
            safe_str(row.get("Dating_Evidence")),
        ]))
        if not evidence:
            return False
        for term in terms:
            term_norm = norm_token(term)
            if len(term_norm) >= 4 and (
                re.search(rf"\b{re.escape(term_norm)}\b", evidence)
                or term_norm in evidence
            ):
                return True
        return False

    @staticmethod
    def _block_auto_apply(result: LLMReviewResult, flag: str, message: str) -> bool:
        result.reason = norm_ws(f"{result.reason} AUTO_APPLY_BLOCKED: {message}")
        result.validation_flags = join_semicolon(split_semicolon_values(result.validation_flags) + [flag])
        return False

    def _dating_correction_allowed(self, row: pd.Series, result: LLMReviewResult) -> bool:
        corrected_codes = self._period_code_values(result.corrected_dating_code)
        if not corrected_codes:
            return True

        original_codes = self._period_code_values(
            row.get("Final_Dating_Normalised_ABR_Code") or row.get("Dating_Normalised_ABR_Code")
        )
        if not original_codes:
            return True

        corrected_set = set(corrected_codes)
        original_set = set(original_codes)
        if original_set.issubset(corrected_set):
            return True

        original_rank = max(self._period_specificity_rank(code) for code in original_codes)
        corrected_rank = max(self._period_specificity_rank(code) for code in corrected_codes)
        if self._row_has_local_dating_evidence(row) and corrected_rank < original_rank:
            return False

        return True

    @staticmethod
    def _period_code_values(value: Any) -> List[str]:
        return [safe_str(code).upper() for code in split_semicolon_values(value) if safe_str(code)]

    @staticmethod
    def _period_specificity_rank(code: str) -> int:
        return PERIOD_SPECIFICITY_RANK.get(safe_str(code).upper(), 1)

    @staticmethod
    def _row_has_local_dating_evidence(row: pd.Series) -> bool:
        source = safe_str(row.get("Dating_Source"))
        evidence = safe_str(row.get("Dating_Evidence"))
        quote = safe_str(row.get("Evidence_Quote"))
        return bool(source and source != "uncertain" and (evidence or quote))

    def _province_correction_allowed(self, row: pd.Series, result: LLMReviewResult) -> bool:
        corrected_province = safe_str(result.corrected_province_normalised)
        if not corrected_province:
            return True

        place = (
            safe_str(result.corrected_place_normalised)
            or safe_str(row.get("Final_Place_Normalised"))
            or safe_str(row.get("Place_Normalised"))
            or safe_str(row.get("Place_Raw"))
        )
        if not place:
            return False

        rows = self._gazetteer_rows_for_place(place)
        if rows.empty:
            return False

        allowed = rows["province_guess"].map(norm_province_token).tolist()
        return norm_province_token(corrected_province) in allowed

    def _gazetteer_rows_for_place(self, place: str) -> pd.DataFrame:
        place_norm = norm_token(place)
        df = self.refs.gazetteer
        matches = df[(df["name_norm"] == place_norm) | (df["asciiname_norm"] == place_norm)]
        if matches.empty:
            alt_mask = df["alt_list"].map(lambda xs: place_norm in {norm_token(x) for x in xs})
            matches = df[alt_mask]
        return matches


# =========================
# GOLDSTANDARD CHECK AND MAIN PIPELINE
# =========================
# Purpose:
# GoldStandardComparator gives a lightweight comparison with the template or
# goldsample, while ArchaeologyPipeline orchestrates all modules from PDF to output.
#
# What it does:
# - Extraction: Pdf/OCR and layout.
# - Segmentation: split text into archaeological records.
# - Candidate detection: find place, dating, features, findings and materials.
# - Normalisation: gazetteer, ABR lookups, local dating, custom aliases and
#   rulebook-driven ABR validation.
# - Review: determine uncertainty and optionally run the LLM second reader.
# - Output: write CSV/XLSX, reviewed output and logs.
#
# Why this is useful:
# This is the central assembly line of RefAI. Keeping each step separate allows
# targeted improvements without making the full system opaque.

class GoldStandardComparator:
    def __init__(self, template_path: Optional[str]):
        self.template_path = template_path

    def compare(self, output_df: pd.DataFrame) -> pd.DataFrame:
        if not self.template_path:
            return pd.DataFrame()
        try:
            gold = pd.read_excel(self.template_path).copy()
        except Exception:
            return pd.DataFrame()

        summary = []
        summary.append({"metric": "gold_rows", "value": len(gold)})
        summary.append({"metric": "output_rows", "value": len(output_df)})

        for gold_col, out_col, label in [
            ("Place_Normalised", "Final_Place_Normalised", "place_nonempty"),
            ("Province_Normalised", "Final_Province_Normalised", "province_nonempty"),
            ("Dating_Normalised_Label", "Final_Dating_Normalised_Label", "dating_nonempty"),
        ]:
            if gold_col in gold.columns:
                g = gold[gold_col].fillna("").astype(str).str.strip()
                summary.append({"metric": f"gold_{label}", "value": int((g != "").sum())})
            if out_col in output_df.columns:
                o = output_df[out_col].fillna("").astype(str).str.strip()
                summary.append({"metric": f"output_{label}", "value": int((o != "").sum())})

        return pd.DataFrame(summary)


class ArchaeologyPipeline:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.refs = ReferenceStore(cfg)
        self.extractor = PDFExtractor(
            flatten_linebreaks=cfg.flatten_linebreaks,
            start_page=cfg.start_page,
            end_page=cfg.end_page,
            enable_ocr_fallback=cfg.enable_ocr_fallback,
            ocr_render_dpi=cfg.ocr_render_dpi,
            ocr_min_text_chars=cfg.ocr_min_text_chars,
            ocr_text_min_chars_for_pdfplumber=cfg.ocr_text_min_chars_for_pdfplumber,
            enable_ocr_readable_pdf_cache=cfg.enable_ocr_readable_pdf_cache,
            keep_runtime_cache=cfg.keep_runtime_cache,
            runtime_cache_dir=Path(cfg.output_dir) / "05_runtime_cache",
            enable_llm_boundary_review=cfg.enable_llm_structural_segmentation,
            anthropic_api_key=cfg.anthropic_api_key,
            anthropic_model=cfg.anthropic_model,
            anthropic_api_version=cfg.anthropic_api_version,
        )
        self.heuristic_segmenter = HeuristicSectionSegmenter(self.refs)
        self.llm_structural_segmenter = LLMStructuralSegmenter(cfg)
        self.archaeobertje = ArcheoBERTjeHelper(cfg, self.refs) if cfg.enable_archaeobertje else None
        self.proposer = CandidateProposer(self.refs, self.archaeobertje)
        self.place_resolver = PlaceResolver(self.refs, cfg.place_fuzzy_threshold)
        self.feature_mapper = LookupMapper(self.refs.feature_lookup, cfg.fuzzy_threshold, self.archaeobertje)
        self.finding_mapper = LookupMapper(self.refs.findings_lookup, cfg.fuzzy_threshold, self.archaeobertje)
        self.material_mapper = LookupMapper(self.refs.material_lookup, cfg.fuzzy_threshold, self.archaeobertje)
        self.dating_resolver = DatingResolver(self.refs)
        self.llm_reviewer = LLMReviewer(cfg, self.refs)
        self.comparator = GoldStandardComparator(cfg.output_template)
        self.output_columns = self._load_output_columns(cfg.output_template)
        self.review_logs = {
            "ambiguous_places": [],
            "unmapped_terms": [],
            "low_confidence_sections": [],
            "local_dating_status": [],
            "archaeobertje_status": [],
            "extractor_status": [],
            "llm_structural_status": [],
            "llm_review_status": [],
            "llm_review_changes": [],
            "gold_comparison": [],
        }
        self.output_stem = make_output_stem(cfg.refai_version_name, Path(cfg.input_pdf))
        self.chunk_text_map = {}
        self._log_status()

    def _log_status(self) -> None:
        if self.archaeobertje is None:
            self.review_logs["archaeobertje_status"].append({"enabled": False, "available": False, "message": "disabled"})
        else:
            self.review_logs["archaeobertje_status"].append({
                "enabled": self.cfg.enable_archaeobertje,
                "available": self.archaeobertje.available,
                "model_name": self.cfg.archaeobertje_model_name,
                "device": self.cfg.archaeobertje_device,
                "message": self.archaeobertje.error_message or "ok",
            })
        self.review_logs["llm_structural_status"].extend(self.llm_structural_segmenter.status_rows)
        self.review_logs["llm_review_status"].extend(self.llm_reviewer.status_rows)

    def _load_output_columns(self, template_path: Optional[str]) -> List[str]:
        cols = list(DEFAULT_OUTPUT_COLUMNS)
        if not template_path:
            return cols
        try:
            df = auto_read_table(Path(template_path))
            template_cols = [str(c).strip() for c in df.columns if str(c).strip()]
            return dedupe_keep_order(template_cols + [c for c in cols if c not in template_cols])
        except Exception:
            return cols

    def _segment_pages(self, pdf_file: str, pages: List[Dict[str, Any]]) -> List[Section]:
        if self.llm_structural_segmenter.available():
            self.llm_structural_segmenter.status_rows.append({
                "component": "llm_structural_segmentation_policy",
                "enabled": True,
                "message": "reviewer_only_mode_primary_segmentation_uses_heuristic",
            })
        return self.heuristic_segmenter.segment(pdf_file, pages)

    def _should_keep_record(self, score: int) -> bool:
        return score >= 4

    @staticmethod
    def _strong_place_resolution(place: PlaceResolution) -> bool:
        if not place.place_normalised or place.ambiguous:
            return False
        return place.resolution_type in {
            "article_context",
            "custom_place_alias",
            "exact_place",
            "landscape_context",
            "site_parent",
            "street_parent",
        } or place.province_source in {
            "article_context",
            "custom_place_alias",
            "exact",
            "exact_province_prioritized",
            "landscape_context",
            "site_parent_context",
            "street_parent_context",
        }

    @staticmethod
    def _landscape_carry_base(place: PlaceResolution) -> bool:
        """
        Exact place/province conflicts can still be valid archaeological context
        anchors. Use them only for landscape carry, not for general carry.
        """
        return bool(
            place.place_normalised
            and place.province_source in {"exact_conflict_review", "exact_ambiguous_review"}
        )

    @staticmethod
    def _explicit_new_place_context(place: PlaceResolution, text: str) -> bool:
        if not place.place_raw and not place.place_normalised:
            return False
        text_norm = norm_token(text)
        candidates = dedupe_keep_order([place.place_raw, place.place_normalised])
        for candidate in candidates:
            candidate_norm = norm_token(candidate)
            if not candidate_norm:
                continue
            patterns = [
                rf"\b(?:te|bij|in|nabij|onder|rond|binnen|gem\.?|gemeente|plaats|stad|dorp)\s+(?:de\s+|het\s+|den\s+|ter\s+|ten\s+)?{re.escape(candidate_norm)}\b",
                rf"\b{re.escape(candidate_norm)}\s*,?\s*(?:gem\.?|gemeente|provincie|prov)\b",
            ]
            if any(re.search(pattern, text_norm) for pattern in patterns):
                return True
        return False

    @staticmethod
    def _text_mentions_place_context(place: PlaceResolution, text: str) -> bool:
        text_norm = norm_token(text)
        for candidate in dedupe_keep_order([place.place_normalised, place.place_raw, place.alt_place_name]):
            candidate_norm = norm_token(candidate)
            if candidate_norm and re.search(rf"\b{re.escape(candidate_norm)}\b", text_norm):
                return True
        return False

    def _derive_article_context_place(self, section: Section) -> Optional[PlaceResolution]:
        """
        Some OCR-heavy articles have no clean place heading, but the article
        title/intro gives the main site context. Use that as a conservative
        parent place for chunks that otherwise latch onto churches, streets or
        secondary references inside the article body.
        """
        heading_norm = norm_token(section.place_heading_raw)
        internal_heading = bool(re.search(
            r"\b(?:kerk|joriskerk|kapel|hof|stadhuis|plein|straat|school|museum|terrein)\b",
            heading_norm,
        ))
        if section.place_heading_raw and not internal_heading:
            return None
        lead = norm_ws(section.text[:2200])
        lead_norm = norm_token(lead)
        candidates: List[str] = []
        if "hof" in lead_norm and "amersfoort" in lead_norm:
            candidates.append("amersfoort")
        patterns = [
            r"\bopgraving(?:en)?\b.{0,120}\b(?:te|in)\s+([a-z][a-z '\-]{2,45})\b",
            r"\bonderzoek\b.{0,120}\b(?:te|in)\s+([a-z][a-z '\-]{2,45})\b",
            r"\b(?:afd\.?\s*)?archeologie\s*gemeente\s+([a-z][a-z '\-]{2,45})\b",
            r"\bgemeente\s+([a-z][a-z '\-]{2,45})\b",
            r"\bde\s+hof\s+te\s+([a-z][a-z '\-]{2,45})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, lead_norm):
                candidate = norm_ws(match.group(1))
                candidate = re.split(
                    r"\b(?:waar|werd|wordt|zijn|is|met|en|van|voor|door|op|uit|bij)\b",
                    candidate,
                    maxsplit=1,
                )[0].strip(" ,.;:")
                if candidate and len(candidate.split()) <= 3:
                    candidates.append(candidate)
                    tokens = candidate.split()
                    for end in range(1, min(3, len(tokens)) + 1):
                        candidates.append(" ".join(tokens[:end]))
        for candidate in dedupe_keep_order(candidates):
            resolved = self.place_resolver.resolve([candidate], section.province_anchor, lead)
            if self._strong_place_resolution(resolved):
                return PlaceResolution(
                    place_raw=resolved.place_raw or candidate,
                    place_normalised=resolved.place_normalised,
                    alt_place_name="",
                    province_normalised=resolved.province_normalised,
                    province_source="article_context",
                    resolution_type="article_context",
                    fuzzy_score=resolved.fuzzy_score,
                    ambiguous=False,
                )
        return None

    def _apply_article_context_place(
        self,
        place: PlaceResolution,
        text: str,
        article_place: Optional[PlaceResolution],
    ) -> PlaceResolution:
        if article_place is None or not article_place.place_normalised:
            return place
        if self._strong_place_resolution(place) and self._explicit_new_place_context(place, text):
            return place

        current_norm = norm_token(place.place_normalised)
        article_norm = norm_token(article_place.place_normalised)
        mentions_article = bool(article_norm and re.search(rf"\b{re.escape(article_norm)}\b", norm_token(text)))
        internal_detail = bool(re.search(
            r"\b(?:kerk|joriskerk|kapel|hof|stadhuis|plein|straat|school|museum|terrein)\b",
            norm_token(" ".join([place.place_raw, place.place_normalised, place.alt_place_name])),
        ))
        if self._strong_place_resolution(place) and not internal_detail and not mentions_article:
            return place
        weak_or_internal = (
            not place.place_normalised
            or place.resolution_type in {"unresolved", "unresolved_common_noun", "unresolved_landscape", "unresolved_street"}
            or place.province_source in {"anchor", "unresolved", "exact_conflict_review"}
            or not self._explicit_new_place_context(place, text)
        )
        if not weak_or_internal:
            return place

        alt_values = []
        for value in [place.alt_place_name, place.place_raw, place.place_normalised]:
            value_norm = norm_token(value)
            if value and value_norm and value_norm != article_norm:
                alt_values.append(value)

        return PlaceResolution(
            place_raw=place.place_raw or article_place.place_raw,
            place_normalised=article_place.place_normalised,
            alt_place_name=join_semicolon(alt_values),
            province_normalised=article_place.province_normalised,
            province_source="article_context",
            resolution_type="article_context",
            ambiguous=False,
        )

    def _apply_context_place_carry(
        self,
        place: PlaceResolution,
        text: str,
        previous_place: Optional[PlaceResolution],
    ) -> PlaceResolution:
        is_landscape = place.resolution_type == "unresolved_landscape"
        if previous_place is None:
            return place
        previous_is_strong = self._strong_place_resolution(previous_place)
        previous_is_landscape_base = is_landscape and self._landscape_carry_base(previous_place)
        if not previous_is_strong and not previous_is_landscape_base:
            return place
        if self._strong_place_resolution(place) and self._explicit_new_place_context(place, text):
            return place

        text_norm = norm_token(text)
        previous_norm = norm_token(previous_place.place_normalised)
        previous_raw_norm = norm_token(previous_place.place_raw)
        previous_alt_norm = norm_token(previous_place.alt_place_name)
        mentions_previous = bool(
            (previous_norm and re.search(rf"\b{re.escape(previous_norm)}\b", text_norm))
            or (previous_raw_norm and re.search(rf"\b{re.escape(previous_raw_norm)}\b", text_norm))
            or (previous_alt_norm and re.search(rf"\b{re.escape(previous_alt_norm)}\b", text_norm))
        )
        current_weak = (
            not place.place_normalised
            or place.resolution_type in {"unresolved", "unresolved_common_noun", "unresolved_landscape", "unresolved_street"}
            or place.province_source in {"anchor", "unresolved", "landscape_context", "exact_conflict_review"}
        )
        can_carry_landscape = is_landscape and bool(previous_place.place_normalised)
        if not current_weak or not (mentions_previous or can_carry_landscape):
            return place

        alt_values = []
        if is_landscape and place.alt_place_name:
            alt_values.append(place.alt_place_name)
        if place.resolution_type == "unresolved_street" and place.alt_place_name:
            alt_values.append(place.alt_place_name)
        if (
            previous_alt_norm
            and previous_alt_norm != previous_norm
            and re.search(rf"\b{re.escape(previous_alt_norm)}\b", text_norm)
        ):
            alt_values.append(previous_place.alt_place_name)

        return PlaceResolution(
            place_raw=place.place_raw or previous_place.place_raw,
            place_normalised=previous_place.place_normalised,
            alt_place_name=join_semicolon(alt_values),
            province_normalised=previous_place.province_normalised,
            province_source="landscape_context" if is_landscape else "context_carry",
            resolution_type="landscape_context" if is_landscape else "context_carry",
            ambiguous=False,
        )

    def run(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        input_pdf = Path(self.cfg.input_pdf)
        output_dir = Path(self.cfg.output_dir)
        if not input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        ensure_dir(output_dir)
        pages = self.extractor.extract_pages(input_pdf)
        self.review_logs["extractor_status"].extend(self.extractor.status_rows)
        sections = self._segment_pages(input_pdf.name, pages)

        records = []
        record_id = 1
        last_strong_place: Optional[PlaceResolution] = None
        last_landscape_parent_place: Optional[PlaceResolution] = None

        for section in sections:
            if (
                last_strong_place is not None
                and section.province_anchor
                and last_strong_place.province_normalised
                and norm_province_token(section.province_anchor) != norm_province_token(last_strong_place.province_normalised)
            ):
                last_strong_place = None
            if section.segmentation_confidence <= 1:
                self.review_logs["low_confidence_sections"].append({
                    "pdf_file": section.pdf_file,
                    "section_id": section.section_id,
                    "page_range": f"{section.start_page}-{section.end_page}",
                    "segmentation_confidence": section.segmentation_confidence,
                    "notes": ";".join(section.segmentation_notes),
                    "segment_source": section.segment_source,
                })

            article_context_place = self._derive_article_context_place(section)
            chunks = chunk_text(section.text, self.cfg.chunk_max_chars, self.cfg.chunk_overlap)
            for idx, chunk in enumerate(chunks, start=1):
                if is_bibliographic_context(chunk):
                    self.review_logs["low_confidence_sections"].append({
                        "pdf_file": section.pdf_file,
                        "section_id": section.section_id,
                        "page_range": f"{section.start_page}-{section.end_page}",
                        "segmentation_confidence": section.segmentation_confidence,
                        "notes": "bibliographic_context_skipped",
                        "segment_source": section.segment_source,
                    })
                    continue
                chunk_id = f"{section.section_id}_CH{idx:03d}"
                self.chunk_text_map[chunk_id] = chunk

                proposed = self.proposer.propose(section, chunk)
                heading_candidates = [section.place_heading_raw] if section.place_heading_raw else []
                place = self.place_resolver.resolve(heading_candidates + proposed["places"], section.province_anchor, chunk)
                place = self._apply_article_context_place(place, chunk, article_context_place)
                place = self._apply_context_place_carry(place, chunk, last_strong_place)
                if (
                    place.resolution_type == "unresolved_landscape"
                    and last_landscape_parent_place is not None
                    and self._text_mentions_place_context(last_landscape_parent_place, chunk)
                ):
                    place = self._apply_context_place_carry(place, chunk, last_landscape_parent_place)
                if self._strong_place_resolution(place) or self._landscape_carry_base(place):
                    last_strong_place = place
                    last_landscape_parent_place = place
                dating = self.dating_resolver.resolve(proposed["dating_phrases"], chunk)
                feature_matches = self.feature_mapper.map_many(proposed["features"], chunk, "feature")
                finding_matches = self.finding_mapper.map_many(proposed["findings"], chunk, "finding")
                material_matches = self.material_mapper.map_many(proposed["materials"], chunk, "material")
                feature_matches, finding_matches, material_matches = self._clean_term_matches(
                    feature_matches,
                    finding_matches,
                    material_matches,
                    chunk,
                )

                if not dating.label:
                    local_anchor_terms = (
                        proposed["features"]
                        + proposed["findings"]
                        + proposed["materials"]
                        + [m.raw for m in feature_matches + finding_matches + material_matches]
                        + [m.label for m in feature_matches + finding_matches + material_matches if m.label]
                    )
                    local_dating = self.dating_resolver.resolve_local_context(local_anchor_terms, chunk)
                    if local_dating.label:
                        dating = local_dating
                        self.review_logs["local_dating_status"].append({
                            "pdf_file": section.pdf_file,
                            "section_id": section.section_id,
                            "chunk_id": chunk_id,
                            "dating_label": dating.label,
                            "dating_code": dating.abr_code,
                            "dating_source": dating.source,
                            "dating_evidence": dating.evidence,
                            "text_snippet": dating.quote[:500],
                        })

                if place.ambiguous:
                    self.review_logs["ambiguous_places"].append({
                        "pdf_file": section.pdf_file,
                        "section_id": section.section_id,
                        "chunk_id": chunk_id,
                        "place_raw": place.place_raw,
                        "province_anchor": section.province_anchor,
                        "text_snippet": chunk[:500],
                    })

                for match in feature_matches + finding_matches + material_matches:
                    if match.source == "unmapped":
                        self.review_logs["unmapped_terms"].append({
                            "pdf_file": section.pdf_file,
                            "section_id": section.section_id,
                            "chunk_id": chunk_id,
                            "raw": match.raw,
                            "text_snippet": match.quote,
                        })

                evidence_quote = dating.quote or next((m.quote for m in feature_matches + finding_matches + material_matches if m.quote), "") or chunk[: self.cfg.evidence_window_chars]
                record_score = score_record_strength(place, dating, feature_matches, finding_matches, material_matches, evidence_quote)

                if not self._should_keep_record(record_score):
                    continue

                uncertain, review_flag, review_reason = derive_uncertainty_and_review(
                    section,
                    place,
                    dating,
                    feature_matches,
                    finding_matches,
                    self.refs,
                    chunk,
                )
                records.append(self._build_output_row(
                    record_id, section, chunk_id, place, dating,
                    feature_matches, finding_matches, material_matches,
                    uncertain, review_flag, review_reason, record_score, evidence_quote
                ))
                record_id += 1

        output_df = pd.DataFrame(records)
        output_df = self._force_output_columns(output_df)

        if self.llm_reviewer.available():
            output_df = self._run_llm_review(output_df)

        gold_df = self.comparator.compare(output_df)
        if not gold_df.empty:
            self.review_logs["gold_comparison"] = gold_df.to_dict("records")

        self.review_logs["llm_structural_status"] = list(self.llm_structural_segmenter.status_rows)
        self.review_logs["llm_review_status"] = list(self.llm_reviewer.status_rows)

        review_frames = {name: pd.DataFrame(rows) for name, rows in self.review_logs.items()}
        self._write_outputs(output_df, review_frames)
        return output_df, review_frames

    def _clean_term_matches(
        self,
        feature_matches: List[MatchResult],
        finding_matches: List[MatchResult],
        material_matches: List[MatchResult],
        text: str = "",
    ) -> Tuple[List[MatchResult], List[MatchResult], List[MatchResult]]:
        """
        Apply the ABR rulebook after broad SLM extraction.

        The SLM intentionally detects generously. This step then uses
        Refai_Custom_Aliases.xlsx to move terms to the right ABR bucket, quarantine
        ambiguous findings unless the evidence supports them, and suppress broad
        generic labels when a more specific ABR term is already present.
        """
        buckets = {"feature": [], "finding": [], "material": []}

        for match in feature_matches:
            if self.refs.is_rejected_match(match, "feature"):
                continue
            target = self.refs.bucket_for_match(match, "feature")
            if target == "finding":
                buckets["finding"].append(
                    self._remap_or_quarantine(match, "finding", "bucket_moved_from_feature")
                )
            elif target == "material":
                buckets["material"].append(
                    self._remap_or_quarantine(match, "material", "bucket_moved_from_feature")
                )
            else:
                buckets["feature"].append(match)

        for match in finding_matches:
            if self.refs.is_rejected_match(match, "finding"):
                continue
            target = self.refs.bucket_for_match(match, "finding")
            if target == "feature":
                moved = self._remap_or_quarantine(match, "feature", "bucket_moved_from_finding")
                buckets["feature"].append(moved)
                if not moved.label:
                    buckets["finding"].append(quarantined_match(match, "bucket_conflict_finding_as_feature_review"))
            elif target == "material":
                moved = self._remap_or_quarantine(match, "material", "bucket_moved_from_finding")
                buckets["material"].append(moved)
                if not moved.label:
                    buckets["finding"].append(quarantined_match(match, "bucket_conflict_finding_as_material_review"))
            elif target == "ambiguous_finding" and not self.refs.finding_context_supported(match, text):
                buckets["finding"].append(quarantined_match(match, "ambiguous_finding_quarantine_review"))
            else:
                buckets["finding"].append(match)

        for match in material_matches:
            if self.refs.is_rejected_match(match, "material"):
                continue
            target = self.refs.bucket_for_match(match, "material")
            if target == "finding":
                buckets["finding"].append(
                    self._remap_or_quarantine(match, "finding", "bucket_moved_from_material")
                )
            elif target == "feature":
                buckets["feature"].append(
                    self._remap_or_quarantine(match, "feature", "bucket_moved_from_material")
                )
            else:
                buckets["material"].append(match)

        mapped_raw_terms = {
            norm_token(m.raw)
            for matches in buckets.values()
            for m in matches
            if m.source != "unmapped" and m.label and norm_token(m.raw)
        }
        mapped_labels = {
            norm_token(m.label)
            for matches in buckets.values()
            for m in matches
            if m.source != "unmapped" and m.label
        }

        def keep(match: MatchResult) -> bool:
            if match.source != "unmapped":
                return True
            raw_norm = norm_token(match.raw)
            label_norm = norm_token(match.label)
            return raw_norm not in mapped_raw_terms and label_norm not in mapped_labels

        cleaned_features = self.refs.suppress_generic_matches(
            "feature",
            [m for m in buckets["feature"] if keep(m)],
        )
        cleaned_findings = self.refs.suppress_generic_matches(
            "finding",
            [m for m in buckets["finding"] if keep(m)],
        )
        cleaned_materials = self.refs.suppress_generic_matches(
            "material",
            [m for m in buckets["material"] if keep(m)],
        )

        # Final safety pass: no ambiguous finding may reach final labels/codes
        # unless the local evidence window supports it. Raw/source are retained
        # through quarantine so the LLM or human reviewer can still inspect it.
        cleaned_findings = [
            self._quarantine_unsupported_ambiguous_finding(match, text)
            for match in cleaned_findings
        ]
        cleaned_findings = self._dedupe_quarantined_ambiguous_findings(cleaned_findings, text)

        return (cleaned_features, cleaned_findings, cleaned_materials)

    def _quarantine_unsupported_ambiguous_finding(self, match: MatchResult, text: str) -> MatchResult:
        if is_quarantined_match(match):
            return match
        if self.refs.bucket_for_match(match, "finding") != "ambiguous_finding":
            return match
        if self.refs.finding_context_supported(match, text):
            return with_source(match, "ambiguous_finding_supported")
        return quarantined_match(match, "ambiguous_finding_quarantine_review")

    def _ambiguous_finding_key(self, match: MatchResult) -> str:
        if self.refs.bucket_for_match(match, "finding") != "ambiguous_finding":
            return ""
        rule = self.refs._ambiguous_rule_for_match(match)
        return norm_token((rule or {}).get("term") or match.label or match.raw or match.abr_code)

    def _dedupe_quarantined_ambiguous_findings(self, matches: List[MatchResult], text: str) -> List[MatchResult]:
        out: List[MatchResult] = []
        grouped: Dict[str, List[MatchResult]] = {}
        order: List[str] = []

        for match in matches:
            key = self._ambiguous_finding_key(match)
            if not key:
                out.append(match)
                continue
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(match)

        for key in order:
            group = grouped[key]
            supported = [
                match for match in group
                if not is_quarantined_match(match) and self.refs.finding_context_supported(match, text)
            ]
            quarantined = [match for match in group if is_quarantined_match(match)]
            if supported:
                out.extend(supported[:1])
            elif quarantined:
                out.append(quarantined[0])
            else:
                out.append(quarantined_match(group[0], "ambiguous_finding_quarantine_review"))

        return out

    def _remap_or_quarantine(self, match: MatchResult, target_bucket: str, source: str) -> MatchResult:
        lookup = {
            "feature": self.refs.feature_lookup,
            "finding": self.refs.findings_lookup,
            "material": self.refs.material_lookup,
        }.get(target_bucket, {})
        for value in [match.raw, match.label]:
            key = norm_token(value)
            if key in lookup:
                payload = lookup[key]
                return MatchResult(
                    raw=match.raw,
                    label=payload["label"],
                    abr_code=safe_str(payload["abr_code"]),
                    source=source,
                    evidence=match.evidence,
                    quote=match.quote,
                )
        return quarantined_match(match, f"{source}_unmapped_review")

    def _build_output_row(self, record_id, section, chunk_id, place, dating, features, findings, materials, uncertain, review_flag, review_reason, record_score, evidence_quote) -> Dict[str, Any]:
        feature_raw = join_semicolon([m.raw for m in features])
        feature_labels = join_semicolon([m.label for m in features if m.label])
        feature_codes = join_semicolon([m.abr_code for m in features if m.abr_code])
        feature_sources = join_semicolon([m.source for m in features if m.source])

        findings_raw = join_semicolon([m.raw for m in findings])
        findings_labels = join_semicolon([m.label for m in findings if m.label])
        findings_codes = join_semicolon([m.abr_code for m in findings if m.abr_code])
        findings_sources = join_semicolon([m.source for m in findings if m.source])

        materials_raw = join_semicolon([m.raw for m in materials])
        materials_labels = join_semicolon([m.label for m in materials if m.label])
        materials_codes = join_semicolon([m.abr_code for m in materials if m.abr_code])

        source_text = " ".join([feature_sources, findings_sources, join_semicolon([m.source for m in materials if m.source])]).lower()
        review_reasons = split_semicolon_values(review_reason)
        if "ambiguous_finding_quarantine" in source_text:
            review_reasons.append("AMBIGUOUS_FINDING_QUARANTINED")
            uncertain = 1
            review_flag = max(int(review_flag or 0), 2)
        if "bucket_conflict" in source_text:
            review_reasons.append("ABR_BUCKET_CONFLICT")
            uncertain = 1
            review_flag = max(int(review_flag or 0), 2)
        if "_unmapped_review" in source_text:
            review_reasons.append("ABR_REMAP_UNMAPPED_REVIEW")
            uncertain = 1
            review_flag = max(int(review_flag or 0), 1)
        review_reason = join_semicolon(review_reasons)

        return {
            "Record_ID": record_id,
            "PDF_File": section.pdf_file,
            "Page_Range": f"{section.start_page}-{section.end_page}",
            "Section_ID": section.section_id,
            "Chunk_ID": chunk_id,
            "Place_Raw": place.place_raw,
            "Place_Normalised": place.place_normalised,
            "Alt_Place_Name": place.alt_place_name,
            "Province_Normalised": place.province_normalised or section.province_anchor,
            "Province_Source": place.province_source or ("anchor" if section.province_anchor else ""),
            "Place_Resolution_Type": place.resolution_type or ("unresolved" if not place.place_normalised else ""),
            "Dating_Raw": dating.raw,
            "Dating_Evidence": dating.evidence,
            "Evidence_Quote": evidence_quote,
            "Dating_Normalised_Label": dating.label,
            "Dating_Normalised_ABR_Code": dating.abr_code,
            "Dating_Source": dating.source,
            "Feature_Context_Raw": feature_raw,
            "Feature_Context_ABR_Label": feature_labels,
            "Feature_Context_ABR_Code": feature_codes,
            "Feature_Mapping_Source": feature_sources,
            "Findings_Raw": findings_raw,
            "Findings_Normal_ABR_Label": findings_labels,
            "Findings_ABR_Code": findings_codes,
            "Findings_Mapping_Source": findings_sources,
            "Material_Raw": materials_raw,
            "Material_Normalised": materials_labels,
            "Uncertain_Flag": uncertain,
            "Segmentation_Confidence": section.segmentation_confidence,
            "Review_Flag": review_flag,
            "Review_Reason": review_reason,
            "LLM_Segment_Source": section.segment_source,
            "LLM_Segment_Notes": join_semicolon(section.segmentation_notes),
            "SLM_Record_Score": record_score,

            "LLM_Reviewed": 0,
            "LLM_Needs_Correction": 0,
            "LLM_Correction_Type": "",
            "LLM_Reason": "",
            "LLM_Confidence": "",
            "LLM_Changed_Fields": "",
            "LLM_Place_Normalised": "",
            "LLM_Province_Normalised": "",
            "LLM_Dating_Label": "",
            "LLM_Dating_Code": "",
            "LLM_Feature_Labels": "",
            "LLM_Feature_Codes": "",
            "LLM_Findings_Labels": "",
            "LLM_Findings_Codes": "",
            "LLM_Material_Labels": "",
            "LLM_Material_Codes": "",
            "LLM_Auto_Applied": 0,
            "LLM_Record_Valid": "",
            "LLM_Noise_Record": "",
            "LLM_Mixed_Context": "",
            "LLM_Validation_Flags": "",
            "LLM_Merge_Split_Advice": "",
            "LLM_Evidence_Check": "",

            "Final_Place_Normalised": place.place_normalised,
            "Final_Province_Normalised": place.province_normalised or section.province_anchor,
            "Final_Dating_Normalised_Label": dating.label,
            "Final_Dating_Normalised_ABR_Code": dating.abr_code,
            "Final_Feature_Context_ABR_Label": feature_labels,
            "Final_Feature_Context_ABR_Code": feature_codes,
            "Final_Findings_Normal_ABR_Label": findings_labels,
            "Final_Findings_ABR_Code": findings_codes,
            "Final_Material_Normalised": materials_labels,
            "Final_Material_Codes": materials_codes,
        }

    def _run_llm_review(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            row = row.copy()
            chunk_text = self.chunk_text_map.get(safe_str(row.get("Chunk_ID")), "")
            review = self.llm_reviewer.review_row(row, chunk_text)

            row["LLM_Reviewed"] = int(review.reviewed)
            row["LLM_Needs_Correction"] = int(review.needs_correction)
            row["LLM_Correction_Type"] = review.correction_type
            row["LLM_Reason"] = review.reason
            row["LLM_Confidence"] = review.confidence
            row["LLM_Changed_Fields"] = review.changed_fields
            row["LLM_Place_Normalised"] = review.corrected_place_normalised
            row["LLM_Province_Normalised"] = review.corrected_province_normalised
            row["LLM_Dating_Label"] = review.corrected_dating_label
            row["LLM_Dating_Code"] = review.corrected_dating_code
            row["LLM_Feature_Labels"] = review.corrected_feature_labels
            row["LLM_Feature_Codes"] = review.corrected_feature_codes
            row["LLM_Findings_Labels"] = review.corrected_findings_labels
            row["LLM_Findings_Codes"] = review.corrected_findings_codes
            row["LLM_Material_Labels"] = review.corrected_material_labels
            row["LLM_Material_Codes"] = review.corrected_material_codes
            row["LLM_Auto_Applied"] = review.auto_applied
            row["LLM_Record_Valid"] = review.record_valid
            row["LLM_Noise_Record"] = review.noise_record
            row["LLM_Mixed_Context"] = review.mixed_context
            row["LLM_Validation_Flags"] = review.validation_flags
            row["LLM_Merge_Split_Advice"] = review.merge_split_advice
            row["LLM_Evidence_Check"] = review.evidence_check

            apply_place = self.llm_reviewer.auto_apply_field_allowed(row, review, "place")
            apply_province = self.llm_reviewer.auto_apply_field_allowed(row, review, "province")
            apply_dating = self.llm_reviewer.auto_apply_field_allowed(row, review, "dating")
            apply_feature_terms = self.llm_reviewer.auto_apply_field_allowed(row, review, "feature")
            apply_finding_terms = self.llm_reviewer.auto_apply_field_allowed(row, review, "findings")
            apply_material_terms = self.llm_reviewer.auto_apply_field_allowed(row, review, "material")
            field_level_applied = any([
                apply_place,
                apply_province,
                apply_dating,
                apply_feature_terms,
                apply_finding_terms,
                apply_material_terms,
            ])
            if field_level_applied:
                row["LLM_Auto_Applied"] = 1
                if apply_place and review.corrected_place_normalised:
                    row["Final_Place_Normalised"] = review.corrected_place_normalised
                if apply_province and review.corrected_province_normalised:
                    row["Final_Province_Normalised"] = review.corrected_province_normalised
                if apply_dating and review.corrected_dating_label:
                    row["Final_Dating_Normalised_Label"] = review.corrected_dating_label
                if apply_dating and review.corrected_dating_code:
                    row["Final_Dating_Normalised_ABR_Code"] = review.corrected_dating_code
                if apply_feature_terms and review.corrected_feature_labels:
                    row["Final_Feature_Context_ABR_Label"] = review.corrected_feature_labels
                if apply_feature_terms and review.corrected_feature_codes:
                    row["Final_Feature_Context_ABR_Code"] = review.corrected_feature_codes
                if apply_finding_terms and review.corrected_findings_labels:
                    row["Final_Findings_Normal_ABR_Label"] = review.corrected_findings_labels
                if apply_finding_terms and review.corrected_findings_codes:
                    row["Final_Findings_ABR_Code"] = review.corrected_findings_codes
                if apply_material_terms and review.corrected_material_labels:
                    row["Final_Material_Normalised"] = review.corrected_material_labels
                if apply_material_terms and review.corrected_material_codes:
                    row["Final_Material_Codes"] = review.corrected_material_codes

            rows.append(row)
        return self._force_output_columns(pd.DataFrame(rows))

    def _force_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df = pd.DataFrame(columns=self.output_columns)
        for col in self.output_columns:
            if col not in df.columns:
                df[col] = ""
        return df[self.output_columns]

    def _build_clean_reviewed_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public-facing reviewed output: compact and readable. The safest final
        values are folded back into the normal extraction columns, while the
        detailed LLM/audit trail is written separately.
        """
        clean = df.copy()
        final_to_public = {
            "Final_Place_Normalised": "Place_Normalised",
            "Final_Province_Normalised": "Province_Normalised",
            "Final_Dating_Normalised_Label": "Dating_Normalised_Label",
            "Final_Dating_Normalised_ABR_Code": "Dating_Normalised_ABR_Code",
            "Final_Feature_Context_ABR_Label": "Feature_Context_ABR_Label",
            "Final_Feature_Context_ABR_Code": "Feature_Context_ABR_Code",
            "Final_Findings_Normal_ABR_Label": "Findings_Normal_ABR_Label",
            "Final_Findings_ABR_Code": "Findings_ABR_Code",
            "Final_Material_Normalised": "Material_Normalised",
            "Final_Material_Codes": "Material_Codes",
        }
        for final_col, public_col in final_to_public.items():
            if final_col in clean.columns:
                values = clean[final_col].fillna("").astype(str).str.strip()
                if public_col not in clean.columns:
                    clean[public_col] = ""
                clean.loc[values.ne(""), public_col] = values[values.ne("")]

        if "Review_Num" not in clean.columns:
            clean["Review_Num"] = clean.get("Record_ID", pd.Series(range(1, len(clean) + 1), index=clean.index))
        else:
            review_num = clean["Review_Num"].fillna("").astype(str).str.strip()
            fallback = clean.get("Record_ID", pd.Series(range(1, len(clean) + 1), index=clean.index))
            clean.loc[review_num.eq(""), "Review_Num"] = fallback[review_num.eq("")]

        for col in CLEAN_REVIEWED_OUTPUT_COLUMNS:
            if col not in clean.columns:
                clean[col] = ""
        return clean[CLEAN_REVIEWED_OUTPUT_COLUMNS]

    def _build_llm_review_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Technical review output: all LLM decisions, review flags, segmentation
        notes and Final_* values live here instead of cluttering the main file.
        """
        review = df.copy()
        review_num = review.get("Record_ID", pd.Series(range(1, len(review) + 1), index=review.index))
        if "Review_Num" in review.columns:
            existing = review["Review_Num"].fillna("").astype(str).str.strip()
            review.loc[existing.eq(""), "Review_Num"] = review_num[existing.eq("")]
        else:
            review.insert(0, "Review_Num", review_num)
        for col in LLM_REVIEW_OUTPUT_COLUMNS:
            if col not in review.columns:
                review[col] = ""
        return review[LLM_REVIEW_OUTPUT_COLUMNS]

    def _write_outputs(self, output_df: pd.DataFrame, review_frames: Dict[str, pd.DataFrame]) -> None:
        out_dir = Path(self.cfg.output_dir)
        ensure_dir(out_dir)
        base = self.output_stem
        reviewed_dir = out_dir / "Reviewed_Outputs_01"
        llm_dir = out_dir / "Llm_Review_Outputs_02"
        raw_dir = out_dir / "Raw_Outputs_03"
        logs_dir = out_dir / "Logs_04"
        for directory in [reviewed_dir, llm_dir, raw_dir, logs_dir]:
            ensure_dir(directory)
        clean_reviewed_df = self._build_clean_reviewed_output(output_df)
        llm_review_df = self._build_llm_review_output(output_df)

        raw_csv = self._write_csv_safely(output_df, raw_dir / f"{base}.csv")
        reviewed_csv = self._write_csv_safely(clean_reviewed_df, reviewed_dir / f"reviewed_{base}.csv")
        llm_csv = self._write_csv_safely(llm_review_df, llm_dir / f"LLM_output_reviewed_{base}.csv")

        if self.cfg.write_excel:
            raw_xlsx = self._write_excel_safely(output_df, raw_dir / f"{base}.xlsx")
            reviewed_xlsx = self._write_excel_safely(clean_reviewed_df, reviewed_dir / f"reviewed_{base}.xlsx")
            llm_xlsx = self._write_excel_safely(llm_review_df, llm_dir / f"LLM_output_reviewed_{base}.xlsx")
        else:
            raw_xlsx = None
            reviewed_xlsx = None
            llm_xlsx = None

        for name, frame in review_frames.items():
            self._write_csv_safely(frame, logs_dir / f"{name}_{base}.csv")

        with (logs_dir / f"run_config_{base}.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)
        with (out_dir / "Readme_Run_Folder.txt").open("w", encoding="utf-8") as f:
            f.write(
                "RefAI run folder\n\n"
                "Reviewed_Outputs_01 = clean files for thesis/manual review\n"
                "Llm_Review_Outputs_02 = technical LLM/validation audit files\n"
                "Raw_Outputs_03 = full raw pipeline exports with all columns\n"
                "Logs_04 = diagnostics, status logs and run configuration\n"
                "Quality_Control_00 = batch-level suspect records and summary, created after the run loop\n"
            )
        self.output_paths = {
            "run_dir": str(out_dir),
            "reviewed_csv": str(reviewed_csv),
            "reviewed_xlsx": str(reviewed_xlsx) if reviewed_xlsx else "",
            "llm_reviewed_csv": str(llm_csv),
            "llm_reviewed_xlsx": str(llm_xlsx) if llm_xlsx else "",
            "raw_csv": str(raw_csv),
            "raw_xlsx": str(raw_xlsx) if raw_xlsx else "",
        }

    def _write_csv_safely(self, df: pd.DataFrame, path: Path) -> Path:
        try:
            df.to_csv(path, index=False)
            return path
        except PermissionError:
            fallback = self._locked_fallback_path(path)
            df.to_csv(fallback, index=False)
            return fallback

    def _write_excel_safely(self, df: pd.DataFrame, path: Path) -> Path:
        try:
            df.to_excel(path, index=False)
            return path
        except PermissionError:
            fallback = self._locked_fallback_path(path)
            df.to_excel(fallback, index=False)
            return fallback

    @staticmethod
    def _locked_fallback_path(path: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_locked_{stamp}{path.suffix}")


# =========================
# BATCH QUALITY OUTPUTS
# =========================
# Purpose:
# These functions make batch results easier to inspect manually.
#
# What it does:
# - Writes the important reviewed XLSX files to final_outputs.
# - Creates Batch_Quality_Summary.xlsx with counts for suspect records, missing
#   dating, LLM corrections and unresolved places.
# - Creates Suspect_Records.xlsx containing only rows that need attention.
#
# Why this is useful:
# Multi-PDF runs quickly create many log files. This layer gives the reviewer two
# overview files instead of forcing them to open dozens of CSVs manually.

def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str).str.strip()


def _write_dataframe_excel_safely(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        fallback = ArchaeologyPipeline._locked_fallback_path(path)
        df.to_excel(fallback, index=False)
        return fallback


def build_batch_quality_outputs(batch_results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if batch_results and safe_str(batch_results[0].get("run_output_dir")):
        final_dir = Path(batch_results[0]["run_output_dir"]) / "Quality_Control_00"
    else:
        final_dir = Path(OUTPUT_DIR) / "Quality_Control_00"
    ensure_dir(final_dir)

    summary_rows = []
    suspect_frames = []
    suspect_columns = [
        "Source_PDF", "Record_ID", "Page_Range", "Section_ID", "Chunk_ID",
        "Place_Raw", "Place_Normalised", "Province_Normalised",
        "Place_Resolution_Type",
        "Final_Place_Normalised", "Final_Province_Normalised",
        "Final_Dating_Normalised_Label", "Final_Dating_Normalised_ABR_Code",
        "Final_Feature_Context_ABR_Label", "Final_Feature_Context_ABR_Code",
        "Final_Findings_Normal_ABR_Label", "Final_Findings_ABR_Code",
        "Final_Material_Normalised", "Final_Material_Codes",
        "Review_Flag", "Review_Reason", "LLM_Reviewed", "LLM_Needs_Correction",
        "LLM_Auto_Applied", "LLM_Record_Valid", "LLM_Noise_Record",
        "LLM_Mixed_Context", "LLM_Validation_Flags", "LLM_Reason",
        "Evidence_Quote",
    ]

    for result in batch_results:
        reviewed_path = Path(result.get("reviewed_xlsx", ""))
        llm_reviewed_path = Path(result.get("llm_reviewed_xlsx", ""))
        if not reviewed_path.exists():
            continue
        clean_df = pd.read_excel(reviewed_path)
        df = pd.read_excel(llm_reviewed_path) if llm_reviewed_path.exists() else clean_df.copy()
        source_pdf = safe_str(result.get("pdf_name")) or reviewed_path.name

        final_place = _text_series(df, "Final_Place_Normalised")
        final_province = _text_series(df, "Final_Province_Normalised")
        if final_place.eq("").all():
            final_place = _text_series(clean_df, "Place_Normalised")
        if final_province.eq("").all():
            final_province = _text_series(clean_df, "Province_Normalised")
        place_before = _text_series(clean_df, "Place_Normalised")
        province_before = _text_series(clean_df, "Province_Normalised")
        final_dating = _text_series(df, "Final_Dating_Normalised_Label")
        if final_dating.eq("").all():
            final_dating = _text_series(clean_df, "Dating_Normalised_Label")
        review_flag = _numeric_series(clean_df, "Review_Flag")
        needs_correction = _numeric_series(df, "LLM_Needs_Correction")
        auto_applied = _numeric_series(df, "LLM_Auto_Applied")
        validation_flags = _text_series(df, "LLM_Validation_Flags").str.lower()

        place_changed = final_place.ne("") & ((place_before != final_place) | (province_before != final_province))
        place_resolution_type = _text_series(clean_df, "Place_Resolution_Type")
        unresolved_place = (
            _text_series(clean_df, "Place_Normalised").eq("")
            | _text_series(clean_df, "Province_Source").isin({"unresolved", "exact_ambiguous_review"})
            | place_resolution_type.isin({"unresolved", "unresolved_landscape"})
        )
        no_final_dating = final_dating.eq("")
        suspect_mask = (
            review_flag.ge(2)
            | needs_correction.ge(1)
            | no_final_dating
            | place_changed
            | unresolved_place
            | validation_flags.str.contains("mixed|noise|place", regex=True, na=False)
        )

        summary_rows.append({
            "Source_PDF": source_pdf,
            "Reviewed_XLSX": str(reviewed_path),
            "LLM_Review_XLSX": str(llm_reviewed_path) if llm_reviewed_path.exists() else "",
            "Rows": len(clean_df),
            "Suspect_Rows": int(suspect_mask.sum()),
            "Review_Flag_GTE_2": int(review_flag.ge(2).sum()),
            "Review_Flag_GTE_3": int(review_flag.ge(3).sum()),
            "LLM_Needs_Correction": int(needs_correction.ge(1).sum()),
            "LLM_Auto_Applied": int(auto_applied.ge(1).sum()),
            "No_Final_Dating": int(no_final_dating.sum()),
            "Unresolved_Or_Ambiguous_Place": int(unresolved_place.sum()),
            "Place_Or_Province_Changed": int(place_changed.sum()),
            "Final_Dating_Filled": int(final_dating.ne("").sum()),
            "Final_Features_Filled": int((_text_series(df, "Final_Feature_Context_ABR_Label") if "Final_Feature_Context_ABR_Label" in df.columns else _text_series(clean_df, "Feature_Context_ABR_Label")).ne("").sum()),
            "Final_Findings_Filled": int((_text_series(df, "Final_Findings_Normal_ABR_Label") if "Final_Findings_Normal_ABR_Label" in df.columns else _text_series(clean_df, "Findings_Normal_ABR_Label")).ne("").sum()),
            "Final_Materials_Filled": int((_text_series(df, "Final_Material_Normalised") if "Final_Material_Normalised" in df.columns else _text_series(clean_df, "Material_Normalised")).ne("").sum()),
        })

        suspect_source = clean_df.copy()
        for col in df.columns:
            if col not in suspect_source.columns:
                suspect_source[col] = df[col].values if len(df) == len(suspect_source) else ""
        if "Record_ID" not in suspect_source.columns and "Review_Num" in suspect_source.columns:
            suspect_source["Record_ID"] = suspect_source["Review_Num"]
        suspect = suspect_source.loc[suspect_mask].copy()
        if not suspect.empty:
            suspect.insert(0, "Source_PDF", source_pdf)
            for col in suspect_columns:
                if col not in suspect.columns:
                    suspect[col] = ""
            suspect_frames.append(suspect[suspect_columns])

    summary_df = pd.DataFrame(summary_rows)
    suspect_df = pd.concat(suspect_frames, ignore_index=True) if suspect_frames else pd.DataFrame(columns=suspect_columns)

    _write_dataframe_excel_safely(summary_df, final_dir / "Batch_Quality_Summary.xlsx")
    _write_dataframe_excel_safely(suspect_df, final_dir / "Suspect_Records.xlsx")
    return summary_df, suspect_df


# =========================
# RUN CONFIG AND EXECUTION
# =========================
# Purpose:
# This bottom block builds a RunConfig per PDF and starts the pipeline.
#
# What it does:
# - build_run_config fills all paths and settings for one PDF.
# - run_one_pdf executes the full pipeline and prints key information.
# - The run loop processes either the single PDF_NAME or, when RUN_BATCH_PDFS is
#   True, all files in BATCH_PDF_NAMES.
#
# Why this is useful:
# The same notebook can run one document for development or a full goldsample set
# for evaluation. This avoids changing internal code when switching documents.

RUN_STARTED_AT = datetime.now()
RUN_FOLDER_NAME = make_run_folder_name(REFAI_VERSION_NAME, RUN_STARTED_AT)
RUN_OUTPUT_DIR = str(Path(OUTPUT_DIR) / RUN_FOLDER_NAME)


def build_run_config(pdf_name: str) -> RunConfig:
    input_pdf = resolve_pdf_file(PDF_DIR, pdf_name)
    return RunConfig(
        input_pdf=input_pdf,
        output_dir=RUN_OUTPUT_DIR,
        output_template=resolve_optional_file(OUTPUT_TEMPLATE),
        refs_dir=str(REFS_DIR),
        custom_alias_file=resolve_optional_file(str(REFS_DIR / "Refai_Custom_Aliases.xlsx")),
        gazetteer_file=resolve_ref_file(REFS_DIR, "Vocab_Nl.xlsx"),
        period_file=resolve_ref_file(REFS_DIR, "Period_Nl.xlsx"),
        abr_period_file=resolve_ref_file(REFS_DIR, "Abr_Period.xlsx"),
        finds_file=resolve_ref_file(REFS_DIR, "Abr_Finds_Ground_Traces.xlsx"),
        material_file=resolve_ref_file(REFS_DIR, "Abr_Material.xlsx"),
        complex_file=resolve_ref_file(REFS_DIR, "Abr_Complex.xlsx"),
        geomorphology_file=resolve_ref_file(REFS_DIR, "Abr_Geomorphology.xlsx"),
        land_use_file=resolve_ref_file(REFS_DIR, "Abr_Land_Use.xlsx"),
        texture_file=resolve_ref_file(REFS_DIR, "Abr_Textural_Characteristics_Of_The_Find_Layer.xlsx"),
        fuzzy_threshold=FUZZY_THRESHOLD,
        place_fuzzy_threshold=PLACE_FUZZY_THRESHOLD,
        chunk_max_chars=CHUNK_MAX_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        evidence_window_chars=EVIDENCE_WINDOW_CHARS,
        write_excel=WRITE_EXCEL,
        start_page=START_PAGE,
        end_page=END_PAGE,
        flatten_linebreaks=FLATTEN_LINEBREAKS,

        enable_llm_structural_segmentation=ENABLE_LLM_STRUCTURAL_SEGMENTATION,
        llm_segmentation_max_pages_per_call=LLM_SEGMENTATION_MAX_PAGES_PER_CALL,
        llm_segmentation_max_text_chars=LLM_SEGMENTATION_MAX_TEXT_CHARS,

        enable_llm_review=ENABLE_LLM_REVIEW,
        anthropic_api_key=ANTHROPIC_API_KEY,
        anthropic_model=ANTHROPIC_MODEL,
        anthropic_api_version=ANTHROPIC_API_VERSION,
        llm_review_confidence_threshold=LLM_REVIEW_CONFIDENCE_THRESHOLD,
        llm_review_auto_apply=LLM_REVIEW_AUTO_APPLY,
        llm_review_only_suspicious=LLM_REVIEW_ONLY_SUSPICIOUS,
        llm_review_min_review_flag=LLM_REVIEW_MIN_REVIEW_FLAG,
        llm_review_max_text_chars=LLM_REVIEW_MAX_TEXT_CHARS,
        llm_review_max_candidates=LLM_REVIEW_MAX_CANDIDATES,

        enable_archaeobertje=ENABLE_ARCHAEOBERTJE,
        archaeobertje_model_name=ARCHAEOBERTJE_MODEL_NAME,
        archaeobertje_device=ARCHAEOBERTJE_DEVICE,
        archaeobertje_batch_size=ARCHAEOBERTJE_BATCH_SIZE,
        archaeobertje_candidate_threshold=ARCHAEOBERTJE_CANDIDATE_THRESHOLD,
        archaeobertje_mapping_threshold=ARCHAEOBERTJE_MAPPING_THRESHOLD,
        archaeobertje_max_span_tokens=ARCHAEOBERTJE_MAX_SPAN_TOKENS,
        archaeobertje_top_k_per_kind=ARCHAEOBERTJE_TOP_K_PER_KIND,

        enable_ocr_fallback=ENABLE_OCR_FALLBACK,
        ocr_render_dpi=OCR_RENDER_DPI,
        ocr_min_text_chars=OCR_MIN_TEXT_CHARS,
        ocr_text_min_chars_for_pdfplumber=OCR_TEXT_MIN_CHARS_FOR_PDFPLUMBER,
        enable_ocr_readable_pdf_cache=ENABLE_OCR_READABLE_PDF_CACHE,
        keep_runtime_cache=KEEP_RUNTIME_CACHE,
        refai_version_name=REFAI_VERSION_NAME,
    )


def run_one_pdf(pdf_name: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], RunConfig]:
    cfg = build_run_config(pdf_name)

    print("\n" + "=" * 80)
    print("Input PDF:", cfg.input_pdf)
    print("OCR fallback enabled:", cfg.enable_ocr_fallback)
    print("ArcheoBERTje enabled:", cfg.enable_archaeobertje)
    print("LLM structural segmentation enabled:", cfg.enable_llm_structural_segmentation)
    print("LLM review enabled:", cfg.enable_llm_review)
    print("Anthropic model:", cfg.anthropic_model)
    print("Output stem:", make_output_stem(cfg.refai_version_name, Path(cfg.input_pdf)))

    pipeline = ArchaeologyPipeline(cfg)

    if pipeline.archaeobertje is not None:
        print("ArcheoBERTje available:", pipeline.archaeobertje.available)
        if not pipeline.archaeobertje.available:
            print("ArcheoBERTje load error:", pipeline.archaeobertje.error_message)

    print("LLM structural segmenter available:", pipeline.llm_structural_segmenter.available())
    print("LLM reviewer available:", pipeline.llm_reviewer.available())

    output_df, review_frames = pipeline.run()

    print(f"Main output rows: {len(output_df)}")
    for name, frame in review_frames.items():
        print(f"{name}: {len(frame)}")
    print(f"Output directory: {cfg.output_dir}")
    print("Reviewed outputs:", getattr(pipeline, "output_paths", {}).get("reviewed_xlsx", ""))
    print("LLM review outputs:", getattr(pipeline, "output_paths", {}).get("llm_reviewed_xlsx", ""))
    return output_df, review_frames, cfg


pdf_names_to_run = BATCH_PDF_NAMES if RUN_BATCH_PDFS else [PDF_NAME]
batch_results = []

for pdf_name in pdf_names_to_run:
    output_df, review_frames, cfg = run_one_pdf(pdf_name)
    output_stem = make_output_stem(cfg.refai_version_name, Path(cfg.input_pdf))
    run_dir = Path(cfg.output_dir)
    batch_results.append({
        "pdf_name": pdf_name,
        "input_pdf": cfg.input_pdf,
        "output_stem": output_stem,
        "reviewed_xlsx": str(run_dir / "Reviewed_Outputs_01" / f"reviewed_{output_stem}.xlsx"),
        "llm_reviewed_xlsx": str(run_dir / "Llm_Review_Outputs_02" / f"LLM_output_reviewed_{output_stem}.xlsx"),
        "rows": len(output_df),
        "output_dir": cfg.output_dir,
        "run_output_dir": cfg.output_dir,
    })

batch_summary_df = pd.DataFrame(batch_results)


def show_result_table(value: Any) -> None:
    """Render tables in Jupyter and fall back to readable text in PowerShell."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display as ipython_display
            ipython_display(value)
            return
    except Exception:
        pass
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    else:
        print(value)


print("\nBatch summary")
show_result_table(batch_summary_df)

quality_summary_df, suspect_records_df = build_batch_quality_outputs(batch_results)
print("\nBatch quality summary")
show_result_table(quality_summary_df)
print(f"Suspect records: {len(suspect_records_df)}")

if batch_results:
    show_result_table(output_df.head(20))
    for key in ["extractor_status", "low_confidence_sections", "gold_comparison", "unmapped_terms"]:
        if key in review_frames and not review_frames[key].empty:
            print(f"\nPreview last run: {key}")
            show_result_table(review_frames[key].head(20))
