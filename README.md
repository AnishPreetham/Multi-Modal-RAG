# Evidence AI

**Offline-runtime multimodal Retrieval-Augmented Generation over documents, images and audio.**

Ask one question and get a grounded answer assembled from PDFs, Word documents,
spreadsheets, screenshots and voice recordings at the same time — with a citation
back to the exact page, timestamp or image behind every claim, and an explicit
refusal when the evidence isn't there.

Every model runs locally. After a one-time online preparation step, the
application needs no internet at all.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Data flow](#data-flow)
- [Technology stack](#technology-stack)
- [Hardware requirements](#hardware-requirements)
- [Setup](#setup)
- [Running it](#running-it)
- [Ingestion](#ingestion)
- [Querying — four modes](#querying--four-modes)
- [Citations](#citations)
- [Relationships](#relationships)
- [Abstention](#abstention)
- [API](#api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Offline operation](#offline-operation)
- [Persistence](#persistence)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What it does

| Capability | How |
| --- | --- |
| Multimodal ingestion | PDF, DOC, DOCX, TXT, CSV, PNG/JPG/WEBP, WAV/MP3/M4A |
| Local OCR | Tesseract 5 |
| Local image understanding | Qwen2.5-VL:3B through Ollama — real image input |
| Local speech-to-text | faster-whisper, timestamped segments |
| Semantic retrieval | all-MiniLM-L6-v2 + FAISS (exact cosine) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Relationship-aware retrieval | deterministic links + query-time semantic links |
| Grounded generation | Qwen2.5-VL:3B, evidence-only prompt |
| Citation transparency | built in code, never emitted by the model |
| Abstention | refuses before the LLM is ever called |
| Reproducible evaluation | fixed dataset, versioned reports |
| Offline runtime | verified with an outbound-socket guard |

---

## Architecture

```
                          EVIDENCE AI
                               |
              +----------------+----------------+
              |                                 |
          INGESTION                          QUERY
              |                                 |
     PDF DOC DOCX TXT CSV              text / image / audio / document
     images   audio                              |
              |                                  v
              v                        query representation
     documents.py  images.py  audio.py           |
              |                                  v
              +--------> ContentItem[] <---------+
                               |
                        all-MiniLM-L6-v2 (CPU)
                               |
                    +----------+----------+
                    |                     |
                  FAISS                SQLite
             (384-d cosine)     (items, sources, links)
                    |                     |
                    +----------+----------+
                               |
                          retrieval (top_k=10)
                               |
                    cross-encoder rerank (top 5)
                               |
                       ABSTENTION GATE
                               |
                  relationship expansion (capped)
                               |
                       final evidence set
                               |
                    Qwen2.5-VL:3B (Ollama, local)
                               |
              +----------------+----------------+
              |                                 |
     programmatic citations              relationships
              |                                 |
              +----------------+----------------+
                               |
                         QueryResponse
                               |
                    +----------+----------+
                    |                     |
               Streamlit UI          FastAPI
```

**One core, two front ends.** `app.py` and `api.py` both call `rag_pipeline.py`
and nothing else, so the UI and the API cannot drift apart.

---

## Data flow

Every modality is reduced to a **text representation**, and that representation
is what gets embedded. This is called **representation-based multimodal
retrieval**.

```
PDF / DOCX / TXT / CSV  ->  extracted, chunked text        ->  MiniLM  ->  FAISS
IMAGE                   ->  Qwen description + OCR text    ->  MiniLM  ->  FAISS
AUDIO                   ->  timestamped transcript window  ->  MiniLM  ->  FAISS
```

> **Being precise about this:** MiniLM is a *text* encoder. It never sees pixels
> or waveforms. An image is searchable because Qwen describes it and Tesseract
> reads it; audio is searchable because Whisper transcribes it. This is **not** a
> native joint image-text embedding space, and the project does not claim to be
> one. The original image and audio files are preserved and remain reachable for
> display, playback, and direct visual inspection at generation time.

---

## Technology stack

| Layer | Component | Runs on |
| --- | --- | --- |
| LLM + vision | Ollama `qwen2.5vl:3b` | GPU (partial) + CPU |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU |
| Speech-to-text | `faster-whisper` base, int8 | CPU |
| OCR | Tesseract 5.5 | CPU |
| Vector index | FAISS `IndexFlatIP` | CPU |
| Metadata | SQLite | — |
| PDF | PyMuPDF | — |
| DOCX | python-docx | — |
| DOC | LibreOffice headless → DOCX | — |
| CSV | pandas | — |
| API | FastAPI + Uvicorn | — |
| UI | Streamlit | — |

**Why PyTorch is pinned to CPU:** on a 4 GB GPU, Ollama already uses ~2.8 GB for
Qwen. Loading MiniLM, the cross-encoder and Whisper onto CUDA would evict Qwen
layers and make every query dramatically slower. Embedding a chunk on this CPU
takes single-digit milliseconds, so the GPU is reserved entirely for the LLM.

---

## Hardware requirements

Verified on the target machine:

| | |
| --- | --- |
| CPU | Intel i5-9300H, 4 cores / 8 threads |
| RAM | 16 GB (**~6 GB free is enough**) |
| GPU | GTX 1650, 4 GB VRAM |
| Disk | ~8 GB for models and dependencies |
| OS | Windows 11 |
| Python | 3.13.7 |

**Measured performance on this hardware:**

| Operation | Time |
| --- | --- |
| Text query, warm | 6–10 s |
| Query with direct image inspection | 12–25 s |
| First query after idle (model reload) | +14 s |
| Image ingestion (OCR + description) | ~7 s |
| Audio transcription | ~1–3× realtime |
| Full 5-file demo corpus ingestion | ~34 s |

`ollama ps` reports the model split **57% CPU / 43% GPU** — a 3B vision model
does not fully fit in 4 GB, so generation runs at 8–12 tokens/second. This is
expected, not a misconfiguration.

> **Run ingestion and querying sequentially.** With ~6 GB of free RAM, a Whisper
> transcription running at the same time as a Qwen generation will thrash.

---

## Setup

### One command

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates the venv, installs CPU PyTorch and all dependencies, verifies every
import, checks Tesseract / LibreOffice / Ollama, pulls `qwen2.5vl:3b`, downloads
and caches MiniLM + CrossEncoder + Whisper, and generates the demo data.

### Manual steps

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\verify_env.py
```

### External tools

```powershell
winget install UB-Mannheim.TesseractOCR                # OCR (required for images)
winget install Gyan.FFmpeg                             # broader audio format support
winget install TheDocumentFoundation.LibreOffice       # ONLY for legacy .doc
```

Ollama comes from <https://ollama.com/download>, then:

```powershell
ollama pull qwen2.5vl:3b
```

### Cache models locally

```powershell
.venv\Scripts\python.exe scripts\prepare_models.py
```

Models are cached at:

| Model | Location |
| --- | --- |
| `qwen2.5vl:3b` | `C:\Users\<you>\.ollama\models` |
| MiniLM, CrossEncoder, Whisper | `C:\Users\<you>\.cache\huggingface\hub` |
| Tesseract | `C:\Program Files\Tesseract-OCR` |

> On Windows without Developer Mode, HuggingFace cannot create symlinks
> (`WinError 1314`). `config.py` sets `HF_HUB_DISABLE_SYMLINKS=1` before any
> HuggingFace import, so this is handled automatically — no admin rights needed.

**After this succeeds you can disconnect from the internet.**

---

## Running it

```powershell
.\start.ps1                 # API on :8000 and UI on :8501
.\start.ps1 -ApiOnly        # API only
.\start.ps1 -UiOnly         # UI only
```

`start.ps1` starts Ollama if it isn't running, verifies the model is present,
warns about a missing index, and then launches.

Manually:

```powershell
.venv\Scripts\python.exe -m uvicorn api:app --port 8000
.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Ingestion

Generate the demo corpus:

```powershell
.venv\Scripts\python.exe create_demo_data.py
```

Ingest everything under `data/`:

```powershell
.venv\Scripts\python.exe scripts\ingest_demo.py
```

Or use the **Ingest** tab in the UI, or `POST /ingest`.

| Format | Extracted as | Location preserved |
| --- | --- | --- |
| PDF | text per page, OCR fallback for scanned pages | `page_number` |
| DOCX | paragraphs + tables, grouped by heading | `section` |
| DOC | LibreOffice → DOCX → as above | `section` |
| TXT | chunked text | — |
| CSV | one `column: value` sentence per row | `row_index` |
| Images | Qwen description + Tesseract OCR | `image_path` |
| Audio | windowed timestamped transcript | `timestamp_start/end` |

Chunking defaults to 800 characters with 100 overlap, breaking at sentence
boundaries where possible.

**`source_id` is a content hash.** Re-ingesting the same bytes produces the same
IDs and adds no duplicate vectors, which is what makes evaluation reproducible.

---

## Querying — four modes

### Text → anything

> *"What evidence do we have about Project Alpha's 2024 development?"*

Returns evidence from PDF, image and audio in one ranked list.

### Image → anything

Upload a screenshot. It is OCR'd and described by Qwen, and that representation
becomes the query.

> *"Find documents and audio recordings related to this screenshot."*

### Audio → anything

Upload a recording. Whisper transcribes it and the transcript becomes the query.

> *"What information in the reports corresponds to this recording?"*

### Document → anything

Upload a PDF or DOCX, then either ask about it or find related material.

> *"What were the main findings in this document?"*
> *"Find screenshots and recordings related to this document."*

> **Self-exclusion:** when you query *with* a file that is already indexed, its
> own chunks are excluded from the results. Otherwise a document's nearest
> neighbours are trivially itself, and "find related material" returns nothing
> useful.

**Discovery vs. factual questions** are detected and prompted differently. A
factual question must be answerable from evidence or it abstains; a discovery
request ("find things related to this") is answered by describing the retrieved
evidence, because there the retrieved set *is* the answer.

---

## Citations

Citations are constructed in `citations.py` from the final evidence set. **The
model never writes citation markers** — the prompt forbids it. A fabricated
citation is structurally impossible: every `Citation` is built from a real
`ContentItem` that was actually retrieved.

Each answer sentence is embedded and matched against the evidence, so citations
reflect what actually supports the text.

```
[1] project_report.pdf - Page 2
[2] project_notes.docx - Q4 Review Meeting
[3] project_dashboard.png
[4] project_meeting.wav - 00:06-00:19
[5] project_metrics.csv - Row 4
```

DOCX cites a **section, never a page**: `python-docx` cannot know page
boundaries, because pagination is computed by the renderer and not stored in the
file. Inventing one would be fabrication.

---

## Relationships

**Deterministic — computed at ingestion, stored in SQLite:**

| Type | Rule |
| --- | --- |
| `SAME_SOURCE` | items from the same file |
| `SAME_PAGE` | items on the same PDF page |
| `TRANSCRIPT_OF` | adjacent audio windows |
| `OCR_OF` | text derived from an image |

**Semantic — computed at query time, never persisted:**

| Type | Rule |
| --- | --- |
| `RELATED_TO` | cosine ≥ `RELATED_TO_THRESHOLD` among *retrieved* evidence only |

`RELATED_TO` compares only the handful of already-retrieved items — a few dozen
pairs, never a scan of the database. Same-source pairs are skipped because
`SAME_SOURCE` already covers them, which leaves the genuinely interesting
cross-modal links.

Expanded evidence is ranked strictly below the weakest direct hit, so a related
item can never outrank real evidence.

---

## Abstention

The abstention gate sits **before the LLM call**. When the best rerank score
falls below `ABSTENTION_THRESHOLD`, the model is never asked, so it cannot
improvise:

```
I could not find sufficient evidence in the available sources to answer this question.
```

with `abstained = true` and `error_code = INSUFFICIENT_EVIDENCE`. The retrieved
evidence is still returned so you can see *why* it abstained. A second check
catches the model emitting `INSUFFICIENT_EVIDENCE` itself.

---

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | runtime + index status |
| `POST /ingest` | upload and index files |
| `POST /query` | text query |
| `POST /query_file` | image / audio / document query |
| `GET /sources` | list indexed sources |
| `GET /source/{id}` | source detail and its items |
| `GET /source/{id}/file` | original file (constrained to `data/`) |

Interactive docs at `http://127.0.0.1:8000/docs`.

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"What was budget utilisation in 2024?\"}'
```

---

## Configuration

Copy `.env.example` to `.env`. Key values:

| Setting | Default | Why |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | |
| `OLLAMA_NUM_CTX` | `8192` | one image costs ~1030 tokens; 4096 is too small |
| `OLLAMA_KEEP_ALIVE` | `30m` | avoids a measured 14 s reload mid-demo |
| `OLLAMA_TEMPERATURE` | `0.0` | minimises generation variance |
| `TOP_K` | `10` | FAISS candidates |
| `RERANK_TOP_K` | `5` | after cross-encoder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | |
| `RELATED_TO_THRESHOLD` | `0.70` | |
| `CITATION_THRESHOLD` | `0.35` | sentence→evidence attribution |
| `ABSTENTION_THRESHOLD` | `-6.0` | cross-encoder logit floor |
| `MAX_CONTEXT_ITEMS` | `5` | |
| `MAX_INSPECT_IMAGES` | `1` | context budget |
| `TORCH_DEVICE` | `cpu` | GPU reserved for Ollama |

---

## Evaluation

```powershell
.venv\Scripts\python.exe evaluation.py
```

Fixed dataset at `evaluation/dataset/queries.json` (version 1.0) with six
queries: PDF, DOCX, image, audio, cross-modal, and one deliberately
unanswerable.

Reports land in `evaluation/reports/` as JSON + CSV + Markdown, each stamped
with dataset version, all four model names, the full configuration, the git
commit and a timestamp.

**Measured results** (dataset v1.0, demo corpus):

| Metric | Value |
| --- | --- |
| Retrieval precision | 0.40 |
| Retrieval recall | 0.96 |
| MRR | 0.75 |
| Citation accuracy | 0.59 |
| Answer correctness | 1.00 |
| Groundedness | 1.00 |
| Hallucination rate | 0.00 |
| Abstention accuracy | 1.00 |
| Mean latency | 20.8 s |

> **Reading precision honestly.** Precision looks low because the metric counts
> any retrieved file not in `expected_sources` as a miss — but the demo corpus is
> *deliberately* overlapping, so retrieving `project_metrics.csv` alongside
> `project_report.pdf` is genuinely relevant and still scores as a false
> positive. Recall, MRR, groundedness and abstention are the meaningful signals
> here. The metric was left un-gamed rather than tuned to flatter the system.

**Determinism.** Retrieval, reranking, relationships and citations are exactly
reproducible. Generation is not: Qwen is sampled and Ollama gives no
cross-version seed guarantee. Temperature is pinned to 0.0, so answer-level
metrics are *approximately* reproducible. Groundedness is scored by keyword and
number overlap, never by an LLM judge — a 3B model grading its own output would
be neither reliable nor honest.

---

## Offline operation

```powershell
.venv\Scripts\python.exe scripts\verify_offline.py
```

This does more than assert. It **monkey-patches `socket.connect` to block every
non-loopback address** for the whole run, then loads all four models, reloads the
index, runs a full cross-modal query and an abstention check. If anything
silently reaches for the network it raises instead of quietly succeeding while
you still have Wi-Fi.

Verified result on this machine:

```
8/8 checks passed
No outbound network connections were attempted.
The only sockets used were loopback to Ollama.
OFFLINE RUNTIME: VERIFIED
```

**Loopback is not the internet.** Ollama runs at `127.0.0.1:11434` and Streamlit
talks to FastAPI over localhost. Those are local processes. The README says this
plainly rather than claiming zero sockets are opened.

For a physical test, disable your network adapter and run `start.ps1` — both the
guard and the real network will then be down.

---

## Persistence

FAISS writes to `index/faiss.index`, SQLite to `index/store.db`, on every ingest.

```powershell
.venv\Scripts\python.exe scripts\ingest_demo.py   # ingest
# close everything
.\start.ps1                                       # restart
```

The index reloads from disk. `test_index_survives_restart` and
`test_store_insert_search_and_reload` cover this in CI.

---

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v          # everything
.venv\Scripts\python.exe -m pytest tests\ -m "not live_ollama"   # no LLM needed
```

**51 tests, all passing.** Nothing is mocked except two deliberate failure
injections (Ollama down, vision failure). Tests skip rather than fail when a
dependency is genuinely absent, so the suite never reports a pass it didn't earn.

Coverage: schemas and ID determinism, chunking, PDF/DOCX/TXT/CSV extraction and
page/section preservation, embedding dimension and normalisation, FAISS
insert/search/reload, idempotent re-ingestion, retrieval ranking and source
exclusion, rerank ordering and score preservation, relationship thresholds,
citation construction and rendering, real OCR, real Whisper timestamps, real
Qwen vision, LLM abstention, no-citation-markers, API endpoints, and four
end-to-end query modes.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `OLLAMA_NOT_AVAILABLE` | daemon down | `ollama serve` |
| `QWEN_MODEL_NOT_AVAILABLE` | model missing | `ollama pull qwen2.5vl:3b` |
| `OCR_NOT_AVAILABLE` | Tesseract missing | `winget install UB-Mannheim.TesseractOCR` |
| `CONVERSION_UNAVAILABLE` | LibreOffice missing | install it, or convert `.doc` → `.docx` manually |
| `WinError 1314` on model download | Windows symlink privilege | handled automatically; else enable Developer Mode |
| First query takes ~40 s | model was unloaded | expected; raise `OLLAMA_KEEP_ALIVE` |
| Very slow generation | model split to CPU | expected on 4 GB VRAM |
| Empty results | nothing indexed | run `scripts\ingest_demo.py` |
| Everything abstains | threshold too strict | raise `ABSTENTION_THRESHOLD` |

---

## Limitations

Stated plainly rather than hidden.

1. **Not a joint embedding space.** Images and audio are retrieved through text
   proxies (description + OCR, transcript). A purely visual similarity with no
   describable content will not retrieve well.
2. **`confidence` is not calibrated.** It is a sigmoid of the best cross-encoder
   score — a relative ordering signal, useful for comparing results within one
   query, not a probability that the answer is correct.
3. **Citations are sentence-to-evidence, not claim-level.** A sentence is matched
   to its best-supporting evidence by embedding similarity. That is an
   approximation, not proof of entailment.
4. **Generation is not bit-reproducible.** See [Evaluation](#evaluation).
5. **A 3B model is a small model.** It occasionally reformats numbers; the prompt
   explicitly forbids this and the evaluation checks for invented numbers, but
   the risk is not zero.
6. **DOCX has no page numbers.** By design — see [Citations](#citations).
7. **`.doc` needs LibreOffice.** Without it, `.doc` ingestion fails loudly with
   `CONVERSION_UNAVAILABLE` rather than silently producing nothing.
8. **No speaker diarization.** Whisper does not identify speakers and this
   project does not invent them.
9. **English only** as configured (Tesseract `eng`, Whisper default).
10. **Single-user, local scale.** `IndexFlatIP` is exact search, ideal up to
    ~10⁵ vectors. Beyond that, switch to an approximate index.

---

## Project layout

```
EvidenceAI/
├── app.py                  Streamlit UI
├── api.py                  FastAPI service
├── rag_pipeline.py         the single orchestrator
├── config.py               all configuration
├── schemas.py              shared data contracts
│
├── documents.py            PDF / DOC / DOCX / TXT / CSV
├── images.py               OCR + Qwen vision
├── audio.py                faster-whisper
│
├── embeddings.py           MiniLM (CPU, cached)
├── vector_store.py         FAISS + SQLite
├── retriever.py            semantic search
├── reranker.py             cross-encoder
├── relationships.py        deterministic + semantic links
├── llm.py                  Ollama / Qwen2.5-VL
├── citations.py            programmatic citations
├── evaluation.py           reproducible evaluation
├── create_demo_data.py     synthetic demo corpus
│
├── scripts/
│   ├── verify_env.py       import verification
│   ├── prepare_models.py   online model caching
│   ├── ingest_demo.py      bulk ingestion
│   ├── run_demo.py         the seven demo scenarios
│   └── verify_offline.py   offline proof with socket guard
│
├── data/{documents,images,audio}/
├── index/                  FAISS + SQLite (generated)
├── evaluation/{dataset,expected_answers,expected_sources,reports}/
└── tests/{test_core.py,test_integration.py}
```

---

## Demo script

```powershell
.venv\Scripts\python.exe create_demo_data.py
.venv\Scripts\python.exe scripts\ingest_demo.py
.venv\Scripts\python.exe scripts\run_demo.py
```

Seven scenarios: cross-modal text query, image→documents, audio→documents,
document Q&A, document→media discovery, abstention, and direct image inspection.

---

*Demo data is synthetic and clearly labelled. "Project Alpha" is fictional.*
