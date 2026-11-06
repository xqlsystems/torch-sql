# Samudra on Empire AI Beta: a production plan for torch-sql

_author_: Alex Merose
_co-author_: Claude (Opus 5.5), via Claude Code
_status_: Draft v2, for discussion. The design's spikes are done; nothing
else is built.
_last updated_: 2026-11-05
_companion to_: [torch-sql: PyTorch models on composable databases](design.md)

---

## 1. Purpose

[Samudra](https://github.com/m2lines/Samudra) is an ocean emulator: a neural
network, written in PyTorch, that predicts the ocean's next state from its
current one. torch-sql is a prototype that runs PyTorch programs, or parts of
them, as SQL on [XQL](https://xql.systems), a group of database projects for
scientific arrays. The [design doc](design.md) says what torch-sql is and how
it works. This doc says what we put into Samudra's production training, in
what order, how we measure each step, and when we stop.

torch-sql has three frontends (design §6). As a *co-processor*, it leaves the
model in PyTorch and takes over work a database should do well, starting with
the training data pipeline and evaluation. As a *device* (`device="sql"`), it
turns every tensor into a table. As a *compile backend*, it receives the model
from `torch.compile`, PyTorch's whole-program compiler. Production starts with
the co-processor, because it changes nothing about the model and is useful on
its own. The other two frontends stay off the production path until XQL can
run programs on the GPU.

## 2. Samudra in production today

Everything in this section comes from Samudra's repository at commit
`4f86be4`.

**The model.** The production configuration, `samudra_om4_v2`, is a U-Net (a
convolutional network that shrinks the grid level by level, then grows it
back) with channel widths 280, 380, 480 and 520, dilated 3×3 convolutions,
batch normalization, and a capped GELU activation. Every layer uses gradient
checkpointing (recomputing activations in the backward pass to save memory).
It trains in float32; bfloat16 is off. Training uses a 4-step autoregressive
rollout (the model's output is fed back as its next input four times), a
batch size of 4, Adam at a learning rate of 6 × 10⁻⁴ with a cosine schedule,
a per-channel MSE loss with dynamic weights, and an EMA (an exponential moving
average of the weights) for evaluation, over 70 epochs.

**The data.** Training reads OM4, the output of the ocean model of NOAA's Geophysical Fluid Dynamics Laboratory, regridded
to half a degree (360 × 720 cells) as five-day means: a 367 GiB Zarr store
(Zarr is a chunked array format), with per-channel means and standard
deviations in two more Zarr stores. Training covers 1975 to 2013, validation
the following year, and inference 2014 to 2022.

**The data loaders.** Samudra chooses a loader through its data config
(`data.loading.type`), and each one implements the same two methods
(`build_source_backend` and `build_batch_loader`):

| Loader | What it does |
|---|---|
| `cpu` | PyTorch `DataLoader` workers read Zarr through Xarray, with pinned host memory. The production configuration uses it. |
| `gpu` | Reads through kvikio, NVIDIA's library for reading files straight into GPU memory. |
| `rust` | A Rust extension reads each rollout's planes once, in parallel, into pinned buffers, and a separate CUDA stream normalizes and masks the next batch while the model computes the current one. |

The `rust` loader is the strongest baseline for anything torch-sql does with
data.

**The job.** `scripts/empireai_nvl72_train.sbatch` runs training on the Beta
cluster (§3) under Slurm, the cluster's batch scheduler. Pyxis, a Slurm plugin,
starts each task in Samudra's container: NVIDIA's PhysicsNeMo 26.05 image with
Samudra installed, built for ARM and pinned by digest. `torchrun` starts one
process per GPU, and DDP (PyTorch's Distributed Data Parallel, which keeps a
copy of the model on each GPU and averages the gradients over NCCL, NVIDIA's
GPU communication library) spreads the batch. A full run is 36 GPUs (9 hosts
of 4) for 12 hours.

## 3. The Beta cluster

Empire AI's
[hardware page](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
describes Beta as 288 NVIDIA B200 GPUs in a DGX GB200 NVL72 SuperPOD, with
Arm Neoverse V2 CPUs (NVIDIA's Grace), about 54 TB of HBM3e (GPU memory) and
about 70 TB of LPDDR5X (CPU memory). Storage is more than 20 PB of flash from
VAST, served over NFS, and more than 10 PB of flash from DDN, served as
Lustre (a parallel file system). Jobs go through login nodes, data transfer
nodes and Slurm.

In a GB200 system, each Grace CPU is paired with two B200 GPUs through
NVLink-C2C, a cache-coherent link that lets a GPU read the CPU's memory
directly. An NVL72 rack joins 72 GPUs with NVLink, NVIDIA's GPU-to-GPU
interconnect; InfiniBand connects the racks.

Four properties of Beta shape the plan:

1. **The CPUs are ARM.** Every XQL package must run on linux-aarch64, inside
   Samudra's container. Spike S7 of the design doc found that each one has
   an ARM wheel, so nothing builds from source (§4, stage 0).
2. **CPU memory sits next to the GPUs.** If the data path leaves its batches
   in Grace memory as Arrow buffers (Arrow is the columnar memory format XQL's
   engines share), a GPU might read them in place instead of copying them.
   Whether PyTorch and Arrow can do that in practice is part of spike S4.
3. **The CPUs have work to spare.** The data path is CPU work (reading,
   decompressing, normalizing, windowing), and each GPU process has Arm cores
   next to it that the model doesn't use.
4. **One rack holds a whole run.** 36 GPUs fit in one NVLink domain of 72,
   which matters later, for distribution (§4, stage 6).

## 4. Stages

Each stage is useful in production on its own, can be turned off in Samudra's
config, and never changes Samudra's results without saying so.

### Stage 0: the stack runs on Beta

torch-sql, xarray-sql, ddx and their engines (DataFusion, a Rust SQL engine,
and DuckDB, an embedded SQL database) install in Samudra's ARM container and
pass a smoke test in a Beta job (the script's `SMOKE=1` mode, 2 GPUs on one
host).

Spike S7 settled most of this without ARM hardware. Samudra's image runs
Python 3.12 and pins `numpy<2` and `zarr<3`; the XQL packages install
alongside those pins and pass the smoke test on x86_64. Three things remain:

1. **One DataFusion version.** xarray-sql 0.6.0 pins DataFusion 54, while
   ddxdb's and einfold's DataFusion extras pin 55, so the three can't share a
   session until xarray-sql releases on 55. ddx works on 54; einfold needs 55.
2. **DuckDB's Substrait extension.** ddx runs on DuckDB through a community
   extension that exists for DuckDB 1.5 but not 2.0, and that DuckDB
   downloads when first loaded. The image pins DuckDB 1.5 and installs the
   extension at build time, since a compute node may have no network.
3. **A run on ARM.** The spike's smoke test,
   `docs/spikes/s7_arm_packaging/smoke.py`, has to run in the image on Beta.

**Done when:** the smoke test passes in a pinned container image on Beta.

### Stage 1: the data path

torch-sql's data loader (design §6.3) becomes a fourth loader type in
Samudra, `data.loading.type: xql`, behind an optional dependency, as the
`rust` loader is. It reads the same Zarr stores through xarray-sql (XQL's
bridge between Xarray and SQL engines), normalizes and masks in SQL, cuts the
rollout windows, and hands each DDP rank its batches as GPU tensors.

**Correct when:** for the same windows, its batches match the `cpu` loader's:
masks exactly, normalized values within a stated floating-point bound.

**Measured:** time per training step, GPU idle time per step, read
throughput from VAST and from Lustre, CPU and host memory per rank, and time
to the first batch. Baselines are the `cpu` loader (production) and the
`rust` loader (the strongest).

**What we found so far.** Spike S9 built this loader over a synthetic store
shaped like OM4. Its batches equal those of Samudra's own normalization and
masking, but it is 1.6 to 3.2 times slower than the `cpu` loader per
rollout window. Most of the time is xarray-sql's scan from Zarr to rows,
which costs about as much as the `cpu` loader's whole pipeline, so stage 1
waits on upstream work in that scan (§6). The gates are in order: first match
the `cpu` loader, then compare with the `rust` loader.

**What we expect.** Parity with the `rust` loader would be a good result;
beating it on raw throughput is unlikely, because it already reads each plane
once and overlaps everything with compute. The case for the XQL loader is
what it can express: preprocessing as queries that the other loaders would
need new code for, such as regridding to another resolution, joining
observations, or selecting ocean basins. This is experiment E2 of the design
doc (design §3.3).

**Stop or narrow when:** after tuning, it is slower than the `cpu` loader.
Then it stays a tool for preprocessing experiments, not a production loader.

### Stage 2: evaluation

Samudra's rollout metrics (RMSE over ocean cells, means over ocean basins,
comparisons with observations) run as queries over the model's outputs and
the dataset, and return Xarray Datasets. They run after training, on CPU
nodes, and don't touch the training job.

**Correct when:** they match `samudra.metrics` on a finished run's outputs.

**Measured:** time and memory against the current evaluation, and how much
code a new metric takes in each.

### Stage 3: checkpoints as data

A reader opens a Samudra checkpoint as an Xarray Dataset with one variable
per parameter (design §7.2), so weights from many runs can be compared with
queries.

**Done when:** a query over two checkpoints gives the same answer as loading
them in PyTorch.

### Stage 4: the model on SQL, offline

This stage is off the production path. It follows the design doc's
milestones M1 to M5: nanoGPT and then Samudra on the device, a masked Samudra
that drops land cells (design §7.3), and then Samudra through the compile
backend with DDP. It runs on the CPU, at small sizes, until the runtime (the
XQL layer that will run programs on GPUs, design §5) exists. Experiment E1,
masked against dense, happens here.

Dropping land cells changes the model. It is the more correct model
scientifically, because no information crosses coastlines, but it would need
the Samudra team's agreement before it goes anywhere near production.

### Stage 5: the model in production, through the compile backend

Once the runtime runs XQL programs on B200s, Samudra trains through
`torch.compile` with torch-sql's backend. Fallbacks to PyTorch are allowed
and reported (design P3), so the model can move over one module at a time.

**Correct when:** a run matches a PyTorch run over the same windows, step by
step for the first steps, and in validation score at the end.

**Measured:** time per step and peak memory against Samudra under plain
PyTorch and under `torch.compile` with its default backend, and the fallback
report.

### Stage 6: distribution

Within one NVL72 rack, split the grid across GPUs by tile, with neighbours
exchanging only the cells along tile edges (*domain decomposition*, design
§13), and compare with DDP over the same GPUs. This is experiment E4.

## 5. What changes in Samudra

Stages 1 to 3 change no model code. Samudra gets one new loader type and its
config, an evaluation entry point, and an optional dependency on torch-sql.
Stages 4 and 5 add `torch.compile` with torch-sql's backend and, if the
Samudra team agrees, the masked model. The design doc lists every model change
with its reason (design §7.3).

## 6. What we need from XQL projects

These are requests to XQL's own repositories, made as each stage needs them:

| Project | Request | For |
|---|---|---|
| xarray-sql | a release on DataFusion 55, the version ddxdb and einfold use ([xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)) | stage 0 |
| xarray-sql | a cheaper scan from Zarr to rows: today it costs as much as a whole dense loader ([xarray-sql#311](https://github.com/xqlsystems/xarray-sql/issues/311)) | stage 1 |
| xarray-sql | Arrow streams split by DDP rank, with read-ahead | stage 1 |
| xarray-sql | the land mask as a filter pushed into the Zarr scan, so land chunks are never read | stage 1 |
| xarray-sql | a fact that rows are ordered within each partition, so batches need no scatter (S4) | stage 1 |
| xarray-sql, the runtime | Arrow device streams (`__arrow_c_device_stream__`), for batches in GPU or Grace memory | stage 1, later |
| ddx | recompute regions ([ddx#251](https://github.com/xqlsystems/ddx/issues/251)), auxiliary outputs ([ddx#258](https://github.com/xqlsystems/ddx/issues/258)), `erf` ([ddx#256](https://github.com/xqlsystems/ddx/pull/256)) | stage 4 |
| the runtime | running programs on GPUs ([xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)) | stage 5 |

## 7. Risks

- **The baseline is strong.** The `rust` loader already deduplicates reads
  and overlaps them with compute, so stage 1 may show no speedup. We report
  that if so, and judge stage 1 on what the loader can express.
- **Version skew.** ARM wheels exist for everything (S7), but xarray-sql,
  ddxdb and einfold must agree on one DataFusion version, and DuckDB 2.0
  breaks ddx's DuckDB route. Each new release can reopen this.
- **The scan.** If xarray-sql's scan stays as slow as S9 measured, stage 1
  can't match even the `cpu` loader, and the XQL loader stays a tool for
  preprocessing experiments.
- **Shared file systems.** 36 ranks reading one Zarr store over NFS behave
  differently from one laptop. Stage 1 measures both VAST and Lustre.
- **Numerical drift.** Normalizing in SQL can sum in a different order from
  NumPy. Every comparison uses a stated bound rather than equality.
- **Time.** This work competes with Samudra's own research. The stage order
  puts the parts that help Samudra soonest first.

## 8. Open questions for the Samudra team

1. **Which windows each rank reads.** Must the XQL loader reproduce
   Samudra's sampler exactly, so runs are comparable step by step, or may it
   assign windows by Zarr chunk to read less? (Design open question 10.)
2. **The masked model.** Is dropping land cells inside the network a change
   the team would consider for production, once it is shown to match a
   masked PyTorch reference? (Stage 4.)
3. **Where the loader lives.** Is a fourth loader type inside Samudra, behind
   an optional dependency, acceptable, or should it live entirely in
   torch-sql, with Samudra importing it?
4. **Which storage.** Is the production data on VAST or on Lustre, and should
   stage 1 target one first?
