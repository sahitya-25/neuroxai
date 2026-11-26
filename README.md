# NeuroXAI — Explainable AI for Cerebral Microbleed Analysis

Local web application for the NeuroXAI research project: **upload a T2* MRI → Swin UNETR segmentation → candidate microbleeds →
false-positive-reduction stage → burden assessment → explanations → PDF/CSV/NIfTI report**, with a study history.

- **Frontend:** React 18 + Vite, Recharts, custom CSS (dark dashboard after the reference design), canvas 3D view.
- **Backend:** Python FastAPI, PyTorch + MONAI (your trained Swin UNETR), NiBabel, NumPy/SciPy, SQLite.
- Runs entirely on your computer. No cloud service, no paid API.

> Research prototype. Not a medical device. Outputs are not diagnoses or treatment recommendations.

![Dashboard (simulated MNI152 brain, demo mode)](docs/screenshots/dashboard_demo_brain.jpg)
*Screenshot: the included simulated scan (real MNI152 brain anatomy, synthetic microbleeds) analysed in demo mode —
not patient data and not the trained model. With your model installed the same screens show the real Swin UNETR output on VALDO scans.*

---

## 0. One folder for the whole project

| Folder | What is in it | Where it comes from |
|---|---|---|
| `backend/`, `frontend/` | the web app (FastAPI + React) | included |
| `research/` | the training notebook, all stage findings, earlier pipeline cells, **`RESULTS.md` (numbers to quote)** | included |
| `assets/mni152/` + `data/samples/DEMO-MNI152-simulated/` | public MNI152 brain → simulated demo scan (real anatomy) | included |
| `model_artifacts/` | **trained Swin UNETR v1** weights, config, model card | Kaggle → `scripts/export_kaggle_project.py` |
| `data/samples/sub-xxx/` | 9 validation + 10 test VALDO scans with the Kaggle result per scan | same export |
| `data/splits/`, `data/processed/kaggle_cache/` | split.json, inventories, the preprocessed cache of all scans | same export |
| `results/` | training log + curves, QC figures, v1 predicted masks, per-scan metrics, Stage 7–9 outputs, v2 logs | same export |
| `data/valdo/` | the raw VALDO Task 2 dataset (all 72 scans, opened directly in the viewer) | your copy, or `fetch_from_kaggle.ps1 -Dataset` |

The trained model and the processed files exist only in your Kaggle sessions, so they cannot be shipped inside this ZIP.
**Section 3** brings them over with one Kaggle cell + one PowerShell command, into exactly these folders.

---

## 1. What works immediately, and what needs your model or more validation

| Feature | Works out of the box | Needs your checkpoint | Needs further validation |
|---|---|---|---|
| Upload + validation of NIfTI files, pseudonymous study IDs, history (SQLite) | ✅ | | |
| Preprocessing identical to the training notebook (RAS, brain crop, normalisation) — tested bit-identical | ✅ | | |
| MRI viewer: slices, min-IP slab, zoom/pan/reset, prev/next, label overlay | ✅ (T2* view) | mask/overlay/probability views | |
| Swin UNETR segmentation + candidate table (voxel + RAS mm coordinates, volume, probability) | | ✅ | |
| Comparison with an uploaded expert mask (Dice, found/missed/false) | | ✅ | |
| False-positive reduction | interface + status | | ✅ no validated FPR model exists yet (docs/FPR.md) |
| Burden bands (None/Single/Multiple/Many), volume, size & depth distribution | | ✅ | ✅ bands are **provisional** (docs/SEVERITY.md) |
| Brain-region distribution | shown as *unavailable* | | ✅ needs atlas registration |
| Explainability: Seg-Grad-CAM, occlusion, SmoothGrad per candidate, opacity, slice selection | | ✅ | interpretation caveats shown |
| 3D view of the scan's brain surface + candidate positions | ✅ after preparation | candidates | |
| PDF report, CSV, NIfTI mask/probability downloads | | ✅ | |
| Model evaluation tables (validation + held-out test) | | ✅ `model_card.json` from the export | |
| **Demo mode** (`NEUROXAI_DEMO_MODE=1`): intensity heuristic instead of the model, stamped **DEMO** everywhere | ✅ | | not a model — only for trying the UI |

The interface never invents numbers: confidence is shown as *mean candidate probability (not calibrated)*, FPR shows a
reduction only if a held-out evaluation is supplied, and unavailable measurements are labelled as such.

---

## 2. Quick start (Windows, VS Code)

**Prerequisites:** Python 3.10–3.12 (python.org, tick *Add to PATH*), Node.js 20 or 22 LTS (nodejs.org), VS Code.

1. **Extract** `NeuroXAI.zip` (right-click → *Extract All…*) to e.g. `C:\Users\<you>\Projects\NeuroXAI`.
2. **Open** the extracted `NeuroXAI` folder in VS Code (*File → Open Folder…*). Open a terminal: *Terminal → New Terminal* (PowerShell).
3. **Set up** (once, ~5–10 min, downloads PyTorch CPU, MONAI and the frontend packages):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
   NVIDIA GPU? use `.\setup.ps1 -Cuda` (or `-TorchIndex <url from pytorch.org>`).
4. **Add your model and data** (section 3). To try the interface first, set `NEUROXAI_DEMO_MODE=1` in `.env`
   and open the included *Simulated demo* scan (real brain anatomy, synthetic microbleeds, labelled DEMO everywhere).
5. **Start the backend** (terminal 1):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_backend.ps1
   ```
6. **Start the frontend** (terminal 2, click **+** in the terminal panel):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_frontend.ps1
   ```
7. **Open** http://localhost:5173 . API docs: http://127.0.0.1:8000/docs . Health check: http://127.0.0.1:8000/api/health

Optional single-URL mode: `.\build_frontend.ps1` once, then only `.\run_backend.ps1` and open http://127.0.0.1:8000 .

---

## 3. Bring everything from Kaggle (model, processed files, results, scans)

> **Your trained model is `runs/swinunetr_t2s_20261016_061117/best.pt`** in the notebook-trail output — not
> `pretrained/model_swinvit.pt`. That file is MONAI's self-supervised *encoder* used as the starting point for training: it has
> no decoder and cannot find microbleeds. The app refuses it with a clear message.

**Fastest route — you already have the notebook output folder (`neuroxai/` with `cache/`, `runs/`, `split.json`):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_from_kaggle.ps1 -NotebookOutput "D:\path\to\neuroxai"
# or directly from Kaggle (API token needed):
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_from_kaggle.ps1 -Kernel "amreenfathima165/notebook-trail" -Dataset "amreenfathima165/neuroxai-valdo-task2-clean"
```
This runs `scripts/import_notebook_output.py`: backs up `best.pt`, converts it to a plain state dict, loads it strictly into the
app's network, copies split/inventories/logs/QC/cache, builds the model card (validation recomputed from Kaggle's saved
probabilities: expect mean Dice 0.3723) and, with `--evaluate test`, re-runs the held-out test scans on your computer
(expect 0.3019). The `-Dataset` part puts the raw VALDO scans in `data/valdo/` so every scan opens in the viewer.

**Alternative — one export cell on Kaggle** (also re-evaluates on the GPU and packs the 19 val/test scans):

1. **On Kaggle** — open the notebook session that holds your training output (or attach the *notebook trail* output and the
   VALDO dataset as inputs), turn the GPU on, paste all of [`scripts/export_kaggle_project.py`](scripts/export_kaggle_project.py)
   into a new cell and run it (≈ 10–20 min). It re-evaluates the model, checks that this app's preprocessing equals the
   training cache, and writes `/kaggle/working/neuroxai_project_bundle.zip`. Check the printout:
   *VAL mean Dice ≈ 0.372, TEST ≈ 0.302* and *Preprocessing equivalence check … PASS*.
   Then **Save Version** (so the output is kept) — or download the zip from the Output panel.
2. **On your computer** (in the NeuroXAI folder):
   ```powershell
   # downloaded the zip by hand:
   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_from_kaggle.ps1 -BundleZip "$HOME\Downloads\neuroxai_project_bundle.zip"
   # or straight from Kaggle with your API token (kaggle.com → Settings → API → Create New Token):
   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_from_kaggle.ps1 -Kernel "<username>/<notebook-slug>" -Dataset "amreenfathima165/neuroxai-valdo-task2-clean"
   ```
   It unpacks into `model_artifacts/`, `data/` and `results/` (existing files are kept as `*.bak`), optionally downloads the raw
   VALDO scans into `data/valdo/`, and runs the verifier.
3. **Verify** any time: `.\.venv\Scripts\python.exe scripts\verify_install.py` — lists what is installed and runs the trained
   model on a held-out test scan; the Dice must match the Kaggle value within 0.01.
4. Restart the backend. The header pill must read **Model ready**.

**Real brain images in the viewer.** Every scan you open — exported samples, any of the 72 raw VALDO scans, or your own upload —
is shown as the actual T2* slices with the model's candidates and the expert label drawn on top. On the Upload page, *Scans ready
to open* groups them as **Held-out test / Validation / Training / Simulated demo**; training scans carry a warning because the
model has seen them. Already have the VALDO folder elsewhere? Set `NEUROXAI_VALDO_DIR=D:\path\to\Task2` in `.env`.

Shortcut without the export: put Kaggle's `best.pt` in `model_artifacts/` (no model card, samples or results then).

**About the reported Dice.** The app shows the metrics recomputed from the exported checkpoint on the project split (`model_card.json`).
For the v1 run the project notes record validation pooled Dice ≈ 0.55 and held-out test mean Dice ≈ 0.30; patch-level pseudo-Dice
values seen during training (≈ 0.6) are not comparable with scan-level Dice. Use the model-card numbers in reports.

### Fixed model configuration (`backend/app/core/config.py`)
T2* only (1 channel) · MONAI SwinUNETR feature size 48 · patch 64×64×32 · sliding window overlap 0.25, gaussian · sigmoid × brain mask ·
threshold 0.5 · preprocessing = notebook Cell 5. These are constants on purpose: changing them runs an unvalidated pipeline.
Deployment settings (paths, device, upload limit, CORS, Grad-CAM layer, demo mode) live in `.env` (see `.env.example`).

---

## 4. Using the app

1. **Upload MRI** — drop a skull-stripped T2* NIfTI (`*_desc-masked_T2S.nii.gz`) and optionally its CMB mask, or open a scan from *Scans ready to open*.
   The file is validated (extension, NIfTI header, 3D, size, mask grid) and prepared; the viewer works from here on.
2. **Start analysis** — progress is shown per stage (model loading, sliding windows, candidates, FPR, severity, saving).
3. **Dashboard** — metric cards, MRI viewer (T2*, predicted mask, overlay, probability, min-IP; zoom, pan, centre on candidate),
   detection table, severity, 3D view, Grad-CAM preview, size/depth distribution.
4. **Analysis Results** — full candidate table (sort, filter, CSV), comparison with the label, threshold sensitivity, model evaluation, downloads.
5. **Severity Assessment** — band, uncertainty, explanation, distributions, provisional-category notice.
6. **Explainability** — pick a candidate and a method → original, mask, heatmap, overlay with opacity and slice controls, plus image evidence.
7. **Patient History** — reopen or delete previous studies.

Keyboard in the viewer: ↑/↓ slice, `+`/`-` zoom, `0` reset; mouse wheel = slice, Ctrl+wheel = zoom, drag = pan.

---

## 5. Where data is stored (all local)
| Path | Content |
|---|---|
| `data/neuroxai.db` | SQLite index: pseudonymous study ID, timestamps, status, summary metrics, optional note |
| `data/uploads/<study>/` | uploaded NIfTI files (`t2s.nii.gz`, `mask.nii.gz`) |
| `data/processed/<study>/` | prepared arrays, results JSON, candidates CSV, output NIfTIs, explanation maps |
| `reports/<study>.pdf` | generated reports |
| `data/valdo/` | raw VALDO scans (read-only; a scan is copied into `data/uploads/` only when opened) |

Original file names, patient names or dates are not stored. Logs contain study IDs only. Delete a study from Patient History to remove all its files.

---

## 6. Project structure
```
NeuroXAI/
  backend/
    main.py                     FastAPI app (CORS, error format, optional static frontend)
    requirements.txt            web + imaging deps;  requirements-ml.txt: MONAI (torch via setup.ps1)
    config/severity.json        provisional burden bands (editable)
    app/
      api/routes.py             all REST endpoints (docs/API.md)
      core/config.py            model constants + .env settings
      schemas/models.py         request/response models
      models/swin_unetr.py      network definition matching the checkpoint
      services/
        preprocess.py           notebook Cell-5 preprocessing
        inference.py            checkpoint adapter (CPU/CUDA, status, demo heuristic)
        candidates.py           connected components, matching, measurements, image-evidence cues
        fpr.py                  false-positive-reduction interface + evaluation utility
        severity.py             burden assessment module
        explain.py, xai_methods.py   Seg-Grad-CAM, occlusion, SmoothGrad
        pipeline.py             background preparation/analysis/explanation jobs, 3D data
        render.py, report.py    PNG rendering, PDF report
        storage.py              SQLite + study folders
      utils/nifti.py            upload validation
    tests/                      pytest: health, validation, severity, FPR, candidates, preprocessing, full API flow
  frontend/                     React + Vite (src/pages, components, services/api.js, hooks, styles, utils)
        valdo.py                raw VALDO folder discovery (Stage-1 rules) + train/val/test tags
  research/                     RESULTS.md, training notebook, stage findings, earlier pipeline cells
  assets/mni152/                public MNI152 template (source of the simulated demo scan)
  model_artifacts/              trained checkpoint, config, model card (from the Kaggle export)
  data/ samples/ splits/ processed/ valdo/ uploads/
  results/                      Kaggle outputs: training logs, QC, predicted masks, Stage 7-9 (from the export)
  reports/
  scripts/                      export_kaggle_project.py (Kaggle cell), fetch_from_kaggle.ps1, verify_install.py,
                                build_kaggle_export.py, make_demo_brain.py
  docs/                         API.md, SEVERITY.md, FPR.md, TROUBLESHOOTING.md
  setup.ps1  run_backend.ps1  run_frontend.ps1  run_tests.ps1  build_frontend.ps1  .env.example
```

## 7. Tests
```powershell
powershell -ExecutionPolicy Bypass -File .\run_tests.ps1
```
32 tests, no checkpoint or GPU needed (synthetic phantom, demo mode): health endpoint, upload validation and rejection,
VALDO folder discovery (nested files, `._` junk, split tags),
severity bands and config validation, FPR interface, candidate metrics (reproduces the documented Stage-4 validation table),
preprocessing bit-identical to the notebook, and the full upload → prepare → analyse → slices → CSV/NIfTI/PDF → history → delete flow.

## 8. Troubleshooting
See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) (PowerShell policy, Python/venv, PyTorch DLLs, CUDA, MONAI version mismatch, Node/npm, ports, data issues).

## 9. Limitations
- Held-out test set: 10 scans with 9 labelled microbleeds → every metric has very wide uncertainty.
- The model saw only skull-stripped VALDO T2* from three sites; other data is out of distribution.
- Burden bands are provisional; no anatomical region mapping; no validated false-positive reduction.
- Attribution maps show output sensitivity, not proof; image-evidence cues are heuristics.
