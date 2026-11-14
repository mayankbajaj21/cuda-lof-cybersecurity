# Windows Setup Guide — CUDA LOF Cybersecurity Project

This is a complete, step-by-step guide to set this project up from a clean Windows
machine, build the CUDA kernels, run every milestone (M1–M6), and launch the
Streamlit dashboard (M7) to view the results. Every command below was actually
run and verified during development on a Windows 11 machine with an NVIDIA RTX
4050 Laptop GPU — follow it in order.

> **Dataset note**: this guide assumes you either have the real CICIDS2017 CSVs
> or will use the synthetic placeholder fallback (see §6). Both paths are
> described.

---

## 0. What you need before starting

| Requirement | Why | Minimum version |
| :--- | :--- | :--- |
| Windows 10/11 64-bit | Target OS | 10 22H2 / 11 23H2+ |
| NVIDIA GPU | To build/run the CUDA kernels (optional — CPU-only works without it) | Any CUDA-capable GPU, compute capability 6.0+ |
| ~10 GB free disk | CUDA Toolkit + VS Build Tools + Python packages + dataset files | — |
| Admin rights | Installing the toolkit/compiler | — |

If you don't have an NVIDIA GPU, skip §3–§4 (CUDA build) — the CPU reference
implementation, dashboard, and all CPU-only milestones still work fully.

---

## 1. Install Python

1. Download Python 3.11–3.13 (64-bit) from https://www.python.org/downloads/
2. Run the installer. **Check "Add python.exe to PATH"** on the first screen.
3. Verify in a **new** terminal (PowerShell):
   ```powershell
   python --version
   pip --version
   ```

---

## 2. Get the project and install Python dependencies

1. Clone or copy the project to a local path, e.g. `C:\dev\cuda-lof-cybersecurity`.
2. Open PowerShell in that folder:
   ```powershell
   cd C:\dev\cuda-lof-cybersecurity
   ```
3. (Recommended) create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked by execution policy, run once as admin:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   This installs: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pytest`,
   `matplotlib`, `psutil`, `streamlit`, `plotly`.

At this point, everything **except** the CUDA kernels will work. If you don't
have an NVIDIA GPU or don't want to build CUDA code, skip to **§5**.

---

## 3. Install the CUDA build toolchain (GPU path only)

CUDA kernels are compiled with `nvcc`, which itself needs a Microsoft C++
compiler (`cl.exe`) as its "host compiler" on Windows. Both must be installed.

### 3.1 Install Visual Studio 2022 Build Tools (the MSVC host compiler)

1. Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   (Or full Visual Studio 2022 Community, which also works — either includes
   the same C++ toolset.)
2. Run the installer. On the workload screen, check:
   **"Desktop development with C++"**
3. Install (this takes a while — several GB).
4. Confirm it's there:
   ```powershell
   Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022"
   ```
   You should see a `Community` or `BuildTools` folder.

### 3.2 Check your GPU driver and compute capability

```powershell
nvidia-smi
```
Note the **driver version** and **GPU name** shown. Look up your GPU's compute
capability (e.g. RTX 3050 → `sm_86`, RTX 4050/4060 → `sm_89`, RTX 4090 → `sm_89`,
RTX 50-series → `sm_90`+) — you'll need this as the `ARCH` value in §4.

### 3.3 Install the CUDA Toolkit

1. Download from https://developer.nvidia.com/cuda-downloads
   Choose: Windows → x86_64 → your Windows version → `exe (local)`.
2. Run the installer, choose **Custom (Advanced)** install.
3. On the components screen, **uncheck "Driver components"** if your driver
   (checked in §3.2) is already newer than what the installer bundles — this
   avoids downgrading your display driver. Keep the CUDA Toolkit, compiler
   (`nvcc`), and libraries checked. Also keep **Nsight Compute** and
   **Nsight Systems** checked if offered — they're optional but useful for
   deeper profiling.
4. Finish the install.
5. Open a **brand new** PowerShell window (PATH changes need a fresh shell)
   and verify:
   ```powershell
   nvcc --version
   ```
   You should see something like `release 13.x, V13.x.xx`. If `nvcc` is not
   found, add it to PATH manually for the current session:
   ```powershell
   $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;$env:PATH"
   ```
   (Adjust the version folder — `v13.5` — to whatever you installed. Check with:
   ```powershell
   Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
   ```
   )

---

## 4. Build the CUDA kernels

All CUDA builds must run from a shell where **both** `vcvars64.bat` (MSVC) and
`nvcc` (CUDA) are on PATH. Do this every new terminal session before building:

```powershell
cd C:\dev\cuda-lof-cybersecurity
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;$env:PATH"
```
> If you installed Build Tools instead of Community, adjust the path to
> `...\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`.

> `call` only works inside `cmd.exe`-style batch contexts. From PowerShell,
> either run the two lines above inside `cmd /c "... && ..."`, or use the
> provided `.bat` scripts directly from `cmd.exe`, which already `call`
> `vcvars64.bat` internally where applicable. The simplest reliable approach
> that was verified during development:
> ```powershell
> cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;%PATH%" && set "ARCH=sm_89" && scripts\build_cuda_lof.bat run'
> ```
> Replace `sm_89` with your GPU's compute capability from §3.2.

### 4.1 Create the output directory

```powershell
New-Item -ItemType Directory -Force -Path bin | Out-Null
```

### 4.2 Set your GPU architecture and build

The build scripts read an `ARCH` environment variable (defaults to `sm_86` if
unset). Set it to match your GPU, then build:

```cmd
set ARCH=sm_89
scripts\build_smoke_test.bat
scripts\build_distance_kernel.bat
scripts\build_knn_kernel.bat
scripts\build_cuda_lof.bat run
scripts\build_cuda_lof_optimized.bat
```

- `build_smoke_test.bat` — sanity-checks the toolchain (vector add). Run its
  resulting `bin\smoke_test.exe` — it should print your GPU's name, SM count,
  memory, etc. and finish with `CUDA SMOKE TEST PASSED`.
- `build_cuda_lof.bat run` — builds the **naive** CUDA LOF pipeline
  (`bin\cuda_lof.dll`) and runs a standalone C++ verification
  (`bin\test_cuda_lof.exe`), which should end with
  `COMPLETE CUDA LOF PIPELINE VERIFICATION PASSED`.
- `build_cuda_lof_optimized.bat` — builds the **optimized** (shared-memory
  tiled) CUDA LOF pipeline (`bin\cuda_lof_optimized.dll`).

### 4.3 Troubleshooting the build

- `nvcc: command not found` → new terminal didn't pick up PATH, or §3.3 step 5
  wasn't done — re-run the PATH line manually.
- `cl.exe not found` / vcvars errors → MSVC Build Tools (§3.1) missing or
  `vcvars64.bat` path doesn't match your install location — check with
  `Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022"`.
- `class "cudaDeviceProp" has no member "clockRate"` → you have CUDA Toolkit
  12 or earlier instead of 13+, or vice versa with older source — this repo's
  kernels already use the modern `cudaDeviceGetAttribute` API, so this
  shouldn't recur, but if you see it after modifying kernel code, replace any
  `prop.clockRate` / `prop.memoryClockRate` reads with
  `cudaDeviceGetAttribute(&var, cudaDevAttrClockRate, device_id)`.
- `LINK : fatal error LNK1104: cannot open file 'bin\...'` → the `bin\`
  directory doesn't exist yet — run §4.1.
- Wrong `-arch` for your GPU → kernel launches will fail at runtime with a
  "no kernel image is available" error. Double-check your compute capability
  from §3.2 and rebuild with the correct `ARCH`.

---

## 5. Prepare the dataset (M1)

### 5.1 If you have the real CICIDS2017 dataset

1. Download the 8 CSV files ("GeneratedLabelledFlows") from the Canadian
   Institute for Cybersecurity: https://www.unb.ca/cic/datasets/ids-2017.html
2. Place all 8 CSVs into `data\raw\`.
3. Inspect and clean:
   ```powershell
   python preprocessing\inspect_dataset.py
   python preprocessing\prepare_dataset.py --seed 42 --raw-dir data\raw --output-dir data\processed
   ```

### 5.2 If you don't have the raw dataset (synthetic fallback)

If `data\raw\` is empty, `preprocessing\prepare_dataset.py` generates a
synthetic, CICIDS2017-**shaped** placeholder dataset instead (same column
count/float32 layout, fabricated values) so every downstream milestone still
runs end-to-end. This is what was used during this project's own development
and verification. Just run:
```powershell
python preprocessing\prepare_dataset.py --seed 42 --output-dir data\processed
```
The dashboard and README both display a clear banner whenever synthetic data
is in use — check `data\processed\dataset_metadata.json`'s
`cleaning_summary.file_stats[].filename` field: if it says
`synthetic_cicids2017.csv`, you're on the fallback path, not real data.

Either way, this produces:
```
data\processed\cicids2017_clean_full.npz
data\processed\dataset_metadata.json
data\processed\subsets\{10k,25k,100k,250k,500k,1m}\subset_*.npz
```

---

## 6. Run the CPU baseline (M2)

```powershell
python scripts\run_cpu_baseline.py --subset 10k --k 20 --output-dir results\m2_cpu_baseline\benchmarks
python scripts\run_cpu_baseline.py --subset 25k --k 20 --output-dir results\m2_cpu_baseline\benchmarks
```
This is pure NumPy — no CUDA required. Results land in
`results\m2_cpu_baseline\benchmarks\*.json` (timings) and `*.npy` (LOF scores).

---

## 7. Run the CUDA baseline (M3) — requires §3–§4

```powershell
python scripts\run_cuda_baseline.py --subset 10k --k 20
python scripts\run_cuda_baseline.py --subset 25k --k 20
```
This loads `bin\cuda_lof.dll` via `ctypes` and times the naive 5-kernel
pipeline with CUDA events.

> **VRAM limit**: the pairwise-distance kernel materializes a dense N×N
> float32 matrix (N²×4 bytes). 25K rows → ~2.5 GB, which fits comfortably in
> 6GB+ GPUs. 100K+ rows need 40GB+ and will fail/OOM on consumer GPUs — this
> is a known, documented architectural limit (see `docs/M6_FINAL_BENCHMARKS.md`),
> not a bug.

---

## 8. Run profiling, optimization, and final benchmarks (M4–M6)

These must run from a shell with CUDA on PATH (same as §4) if you want real
GPU numbers; otherwise they still run and honestly report `UNAVAILABLE` for
GPU-only metrics.

```powershell
# Make sure PATH includes nvcc (see §4) if you want real GPU profiling:
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5\bin;$env:PATH"

python scripts\run_m4_profiling.py        # per-kernel CUDA-event profiling → results\m4_profiling\
python scripts\run_m5_optimization.py     # naive vs. optimized kernel comparison → results\m5_cuda_optimized\
python scripts\run_m6_final_benchmarks.py # CPU vs naive vs optimized at feasible sizes → results\m6_final_benchmarks\
```

Each writes its own CSV/JSON/Markdown reports and chart PNGs under
`results\m{4,5,6}_*\`. Expect `run_m6_final_benchmarks.py` to take several
minutes — the CPU LOF reference is O(N²) and the 25K-row run alone takes
roughly 20–25 seconds per repetition, times the configured repeat count.

> **If these report `can_run_gpu_profiling: false` even though you built the
> DLLs in §4**: it means the *shell these scripts were run from* doesn't have
> `nvcc`/CUDA on its PATH (each new terminal starts without it unless you
> re-export it, as shown above) — it does **not** mean the DLLs are broken.
> Re-export PATH in that shell and re-run.

---

## 9. Run the test suite

```powershell
pytest -v tests\
```
- Without built CUDA DLLs: CPU/dataset tests pass, CUDA tests are skipped.
- With built CUDA DLLs (§4 done): all tests run, including kernel correctness
  checks against the CPU reference. Expect all tests to pass.

---

## 10. Launch the dashboard (M7) and view all results

```powershell
streamlit run app.py
```
This opens a browser tab (default `http://localhost:8501`). Use the sidebar
to navigate:

| Page | Shows |
| :--- | :--- |
| Overview | Project summary, milestone status, key metrics |
| Hardware & Environment | Live detected CPU/GPU/OS + historical benchmark hardware |
| Dataset Analytics | M1 dataset stats and charts (with synthetic-data banner if applicable) |
| CPU vs. GPU Benchmarks | M2/M6 timing comparison, speedup charts |
| CUDA Profiling | M4 per-kernel breakdown, bottleneck highlight |
| CUDA Optimization | M5 naive-vs-optimized kernel comparison |
| Benchmark Lab | Run a new live experiment from the UI, with a pre-run VRAM/RAM feasibility check |
| Memory & Scalability | M6 memory analysis, VRAM feasibility wall, live "what-if" size calculator |
| Anomaly Detection Results | LOF score distributions, flagged anomalies |
| Correctness & Validation | CPU vs. naive vs. optimized agreement metrics |
| Experiment History | Timeline of all saved benchmark/profiling runs |
| Exports | Download any generated CSV/JSON/Markdown report |

If a results file for a given milestone doesn't exist yet (because you skipped
a step above), that page shows an honest "not available" message instead of
crashing — just go back and run the missing script.

To stop the dashboard, press `Ctrl+C` in the terminal it's running in.

---

## 11. Quick end-to-end checklist

- [ ] Python installed, `pip install -r requirements.txt` done
- [ ] (GPU path) VS2022 Build Tools + CUDA Toolkit installed, `nvcc --version` works in a fresh terminal
- [ ] (GPU path) `bin\cuda_lof.dll`, `bin\cuda_lof_optimized.dll` built via §4
- [ ] `data\processed\subsets\` populated via §5 (real or synthetic)
- [ ] M2 CPU baseline run (§6)
- [ ] M3 CUDA baseline run (§7) — GPU path only
- [ ] M4/M5/M6 scripts run (§8) — GPU path only for real GPU numbers
- [ ] `pytest -v tests\` passes (§9)
- [ ] `streamlit run app.py` opens and every page shows data (§10)
