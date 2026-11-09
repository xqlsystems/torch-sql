# torch-sql: PyTorch models on composable databases

_author_: Alex Merose
_co-author_: Claude (Opus 5.5), via Claude Code
_status_: Draft v3, for discussion. The spikes of §16 are done; nothing else
is built.
_last updated_: 2026-11-05
_companion_: [Samudra on Empire AI Beta](samudra-on-beta.md), the
production plan

---

## 1. What torch-sql is

[PyTorch](https://pytorch.org/) is a Python library for machine learning. A
PyTorch model is ordinary Python code that calls operations on *tensors*
(N-dimensional arrays), and each tensor lives on a *device*: `"cpu"`, `"cuda"`
for an NVIDIA GPU, and so on.

torch-sql runs PyTorch programs, or parts of them, as SQL on XQL, a group of
database projects for scientific arrays (§2). It offers three ways in, which
§6 compares:

- **As a co-processor.** PyTorch keeps running the model on the GPU, and
  torch-sql runs the parts that suit a database, starting with the training
  data pipeline and evaluation.
- **As a device.** torch-sql adds one more device, `"sql"`. A tensor on that
  device is a table, and each operation on it becomes part of a relational
  query.
- **As a compiler backend.** `torch.compile` is PyTorch's whole-program
  compiler. torch-sql can be one of its backends, and receive the model as a
  graph instead of op by op.

On the device, nothing runs when an operation is called. torch-sql records
what the code asked for, and runs it as SQL when a result is needed:

```python
import torch
import torch_sql

model = Samudra(config).to("sql")       # parameters become tables
opt = torch.optim.Adam(model.parameters(), lr=6e-4)

for batch in loader:                    # batches are tables too (§7)
    loss = model.loss(batch)            # recorded, not run
    loss.backward()                     # ddx (§2) writes the backward pass
    opt.step()                          # Adam's update is SQL too
    opt.zero_grad()
    torch_sql.sync()                    # plan and run the whole step
```

The project has one immediate question and three longer ones.

**The immediate question.** If a real scientific model is written in PyTorch,
with very few changes, and run as SQL, do efficient kernels come out of the SQL
program on their own? A *kernel* here is the low-level routine that does the
arithmetic of one operation on the hardware, such as a matrix product on a GPU.
Today a PyTorch user gets fast kernels because someone wrote them by hand for
each operation. A database instead plans a whole query and picks algorithms
for it. torch-sql tests whether a query optimizer can find kernels as good as
the hand-written ones, starting from nothing but the program.

**The longer-term hypotheses.** The target is training Samudra (§1.1) where
it trains today: the Beta cluster of [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-), an
HPC system (a high-performance computing cluster) with 288 NVIDIA B200 GPUs
in GB200 NVL72 racks. The companion doc describes the cluster.

- **H1. Distribution.** A database spreads one query over many machines by
  partitioning its tables and moving rows between partitions. Expressing a
  training step as queries may give a better way to spread a model over many
  GPUs than PyTorch's FSDP (Fully Sharded Data Parallel, which splits a model's
  weights across GPUs and gathers each layer's weights back before using
  them). §13 describes how.
- **H2. Whole-graph optimization.** A query optimizer sees the whole program at
  once: the data, the weights, the forward pass, the gradient and the
  optimizer's update. It may use that view to use the hardware better than a
  framework that runs one operation at a time.
- **H3. Composition.** Each piece (differentiation, kernel optimization,
  execution, distribution) is a separate project with a narrow interface, so a
  user takes only what they need.

### 1.1 Samudra, the first real model

[Samudra](https://github.com/m2lines/Samudra) is an ocean emulator: a neural
network trained to predict the ocean's next state (temperature, salinity,
currents and sea-surface height at 19 depths) from the current one, on a
latitude-longitude grid of the globe. It is a *U-Net*, a convolutional network that
shrinks the grid level by level, then grows it back, with skip connections
that carry each level's features across. Its building blocks are 3×3
convolutions with dilations 1, 2, 4 and 8, batch or instance normalization,
average pooling, and bilinear upsampling. Longitude wraps around the globe, so
its convolutions use circular padding in longitude. It is trained on a 4-step
*autoregressive rollout*: the model's prediction is fed back as the next
input, four times, and the loss sums over the four predictions.

Samudra is the model torch-sql is built for. Its production configuration
trains in float32 on a half-degree grid (360 × 720 cells), on 36 GPUs of the Beta
cluster, with PyTorch's DDP (Distributed Data Parallel, which keeps a copy of
the model on every GPU and averages gradients between them). nanoGPT (a small
GPT-2 language model, §16) comes first on the device only because it is
smaller and already has a hand-written SQL version to compare against.

### 1.2 Non-goals

- **Executing plans.** torch-sql produces plans; other layers run them (§5).
- **A distributed engine.** torch-sql must produce plans that can be
  distributed (§13), but distribution belongs to the runtime and the engines.
- **A training framework.** PyTorch is the training framework. torch-sql
  doesn't replace `torch.optim` or training loops. Its data loader (§6.3)
  yields ordinary PyTorch tensors, so it fits under any training loop.
- **`torch.export`.** This is PyTorch's path for saving a model as a graph
  without its Python code. torch-sql captures programs as they run, on the
  device or through `torch.compile`, and reads weights from checkpoints
  instead (decision D1).
- **All of PyTorch.** torch-sql covers the operations its target models use.
  The rest raise an error, or run in PyTorch if the user allows it (§4, P3).
- **Bitwise agreement with CUDA.** Results agree with PyTorch on the CPU to a
  stated floating-point bound (§14), not bit for bit.

### 1.3 Success criteria

1. nanoGPT trains on `device="sql"` and matches both PyTorch on the CPU and
   xarray-sql's hand-written SQL version, step by step (M1).
2. A masked Samudra (§7.3) runs inference, and later trains, on
   `device="sql"`, and matches a masked PyTorch reference (M2, M3).
3. The changes to Samudra's model code fit on one page, and each one is listed
   in §7.3 with its reason.
4. torch-sql never gives a wrong number. Anything it can't lower raises an
   error, unless the user has allowed it to run in PyTorch, and then every
   such fallback is reported (P3).
5. We can report how fast the SQL program runs with and without op hints
   (§10), which answers the immediate question.
6. Experiments E1 to E3 (§3.3) run against strong baselines, and we report
   the results whichever way they go.
7. Production Samudra uses torch-sql as a co-processor on the Beta cluster,
   with results that match its current training (§6.3, and the companion
   doc).

---

## 2. Background

This section introduces the projects and ideas the rest of the doc builds on.

**PyTorch's dispatcher and ATen.** Every PyTorch operation, whether called as
`x + y`, `torch.matmul(x, y)` or a layer like `nn.Conv2d`, ends up as one or
more *ATen ops*. ATen is PyTorch's C++ library of about 2,000 low-level tensor
operations, such as `aten.add.Tensor` and `aten.convolution`. The *dispatcher*
routes each ATen op to an implementation based on the tensors' device and
other properties. PyTorch can also break most ops into a fixed subset of about
180 called *core ATen*, using its *decompositions*: rules that rewrite one op
into simpler ones.

**Tensor subclasses and modes.** Python code can join the dispatcher in two
ways. A *wrapper tensor subclass* is a `torch.Tensor` subclass that holds no
data of its own. PyTorch calls its `__torch_dispatch__` method for every ATen
op that touches it. A *mode* (`TorchDispatchMode` for ATen ops,
`TorchFunctionMode` for the higher-level `torch.*` functions) intercepts every
op while it is active, including ops that create tensors. PyTorch also
reserves one spare device slot, `PrivateUse1`, which a library can rename (to
`"sql"`, here) so that `device="sql"` is accepted everywhere.

**torchax.** [torchax](https://google.github.io/torchax/user_guide/how-it-works/)
runs PyTorch code on [JAX](https://jax.readthedocs.io/), Google's array
library. Its tensor is a wrapper subclass holding a JAX array. An `Environment`
object holds a function mode, a dispatch mode and a table from ATen ops to JAX
functions, and it renames `PrivateUse1` to `"jax"`. Ops missing from the table
are decomposed first. Each op runs as its own JAX call, and a separate `jax_jit`
function traces a whole function and compiles it at once. torch-sql borrows
torchax's capture mechanism almost unchanged (§9.1).

**torch_xla's lazy tensors.** [torch_xla](https://github.com/pytorch/xla)
runs PyTorch on Google's TPUs. In its *lazy tensor* mode, an op records a node
in a graph instead of running, and `mark_step()` compiles and runs everything
recorded so far. torch-sql works the same way: `torch_sql.sync()` plays the
role of `mark_step()` (§9.6). torch_xla later added a `torch.compile` backend
as well, so that a training loop doesn't pay for tracing every op on every
step.

**`torch.compile`.** `torch.compile(model, backend=...)` captures a model as
it runs, using *Dynamo*, PyTorch's tracer for Python bytecode. Dynamo turns
the parts of a program it can follow into *FX graphs* (PyTorch's graph format,
whose nodes are PyTorch or ATen ops), and hands each graph to a *backend* that
compiles it. Where it can't follow the Python, it makes a *graph break*: that
code runs as ordinary PyTorch, and capture resumes after it. Each compiled
graph comes with *guards*, the conditions (shapes, dtypes and some Python
values) under which it can be reused without tracing again.

**Custom ops and autograd functions.** `torch.library` lets a library register
its own op, which PyTorch then treats like a built-in one, including inside
`torch.compile`. `torch.autograd.Function` lets a library define both the
forward computation and the backward computation of a piece of a model.
Together they let part of a model run somewhere other than PyTorch while
the rest of the model, its gradients and DDP work as usual.

**XQL.** [XQL Systems](https://xql.systems) is a group of projects that give
SQL access to large scientific arrays. Its data model is the *long* (or tidy)
table: an N-dimensional array is a table with one row per coordinate tuple,
one column per dimension and a column per variable. A temperature array
`temp(time, lat, lon)` becomes rows of `(time, lat, lon, temp)`. A matrix
product is then a join on the shared index followed by `SUM(a * b)` grouped by
the other indices. In XQL, a row that is absent means that coordinate has no
value; this makes tables naturally sparse (§8.3).

**xarray-sql.** [Xarray](https://xarray.dev) is a Python library for labelled
N-dimensional arrays, used throughout climate and ocean science, often over
[Zarr](https://zarr.dev), a chunked array storage format.
[xarray-sql](https://github.com/xqlsystems/xarray-sql) registers an Xarray
Dataset with a SQL engine as a long table, without copying it: each chunk
becomes a partition of the table. It translates data, not queries. It owns two
seams, *register* (a Dataset becomes a table) and *round-trip* (a query result
becomes a Dataset), and leaves everything between them to the engine.

**DataFusion and DuckDB.** [Apache DataFusion](https://datafusion.apache.org/)
is a SQL query engine in Rust, designed to be embedded and extended.
[DuckDB](https://duckdb.org/) is an in-process analytical SQL database. Both
run on a single machine today. xarray-sql supports both, and others through
[ADBC](https://arrow.apache.org/adbc/) (Arrow Database Connectivity, a
standard API for databases that exchange data as Arrow).
[Apache Arrow](https://arrow.apache.org/) is the in-memory columnar format all
of these engines share.

**Moving arrays between libraries.** Arrow's
[PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
lets one Python library hand Arrow data to another without copying, through
methods such as `__arrow_c_stream__` (a stream of record batches).
xarray-sql's tables already speak it, and DataFusion, DuckDB and the Polars dataframe
library consume it. Its device variant (`__arrow_c_device_array__` and
`__arrow_c_device_stream__`) does the same for Arrow data in GPU memory,
following Arrow's
[C Device Data Interface](https://arrow.apache.org/docs/format/CDeviceDataInterface.html).
[DLPack](https://dmlc.github.io/dlpack/latest/) is the corresponding standard
for dense tensors, which `torch.from_dlpack` reads. The Python
[dataframe interchange protocol](https://data-apis.org/dataframe-protocol/latest/API.html)
covers similar ground for dataframes, but
[xarray-sql#86](https://github.com/xqlsystems/xarray-sql/issues/86) chose the
PyCapsule interface instead, because it streams lazily and SQL engines read it
directly.

**Substrait.** [Substrait](https://substrait.io/) is a standard,
engine-neutral format for query plans. A plan written by one engine's
*producer* can be read by another engine's *consumer*.

**ddx.** [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries.
A user writes a loss as SQL and asks for `grad(loss, w)`; ddx writes the
backward pass as a *program*, a sequence of Substrait plans that the engine
runs in order, saving some intermediate tables (the *tape*) and recomputing
others. ddx refuses anything it can't differentiate correctly, with an error.

**einfold.** [einfold](https://github.com/xqlsystems/einfold) is a query
optimizer. It finds array computations in a query plan, such as matrix
products and attention, and replaces them with array algorithms, without
changing the results beyond a stated rounding bound. Its central object is a
*fold over a join*: a join of operands, grouped by output dimensions, with an
aggregate folding each group's values. It never executes a plan itself and
never touches a device. It also defines *facts*: what is known about each
operand (extents, layouts, completeness), carried in a side-channel table
keyed by plan node.

---

## 3. Thesis, and how we'll test it

The fair objection to this project is that it finds correspondences between
machine learning and databases, and then builds a system that does no better
than doing machine learning the usual way. This section states which parts of
that objection we accept, where we think the relational model gives
capabilities that tensor systems lack, and the experiments that will decide
it.

### 3.1 What isn't new

**A global graph.** An *ML compiler* takes a whole program of tensor
operations and compiles it for the hardware. Google's
[XLA](https://openxla.org/xla) is the compiler behind JAX's `jit`, and
PyTorch's `torch.compile` does the same for PyTorch. Both already see a whole
training step: forward pass, backward pass and optimizer update. XLA's
automatic sharding, [GSPMD](https://arxiv.org/abs/2105.04663), splits that
program across devices from a few annotations, including splitting a
convolution across devices by space with halo exchange (§13). So "the
optimizer sees the whole program" does not by itself set torch-sql apart.

**The correspondence.** That an einsum is a join followed by an aggregate has
been worked out before. Examples are FAQ/InsideOut (Abo Khamis, Ngo and
Rudra, 2016), Tensor Relational Algebra (Yuan et al., VLDB 2021) and SPORES
(Wang et al., VLDB 2020), which optimizes linear algebra by equality
saturation over relational algebra, much as einfold plans to. Systems that ran
ML inside databases, such as MADlib and SystemDS, didn't replace ML
frameworks.

**Dense kernels.** On dense operations over a regular grid, hand-written GPU
libraries such as NVIDIA's cuDNN (for convolutions) and cuBLAS (for matrix
products) are very hard to beat. The fastest SQL training result so far
supports this: xarray-sql's nanoGPT work got its largest gains by keeping
intermediates as dense arrays between operators, and its best step is still
3.76× slower than PyTorch on Apple's GPU. We don't expect to beat these
libraries on dense layers. The best realistic outcome there is parity, by
calling them.

### 3.2 Where the relational model should do better

The claim is narrower than "SQL is faster". Relational systems handle four
things that tensor systems handle poorly, and scientific ML at scale runs
into all four.

1. **Irregular domains.** A tensor is a dense box. A land mask, an
   unstructured ocean mesh (as in the MPAS and ICON models), the mesh of a
   graph network, ragged sequences and mixture-of-experts routing all have to
   be padded, masked, or given custom kernels. A join handles them directly,
   and only rows that exist cost anything. The masked Samudra (§7.3) is a
   small example: in SQL, the scientifically correct model is also the
   easier one to write.
2. **Data and model in one plan.** ML compilers see the model, not the data
   pipeline. Training on large Zarr archives is often limited by reading,
   normalizing, regridding and windowing data, not by arithmetic. A
   relational plan includes the data, so it can read only the chunks a step
   needs and fuse preprocessing into the first layer. The strongest result on
   the database side is *factorized learning* (Olteanu, Schleich and others):
   when training data is a join of tables, learning over the join without
   materializing it gives asymptotic speedups, not constant ones.
3. **Memory and failure.** Databases plan within a memory budget, spill to
   disk, and recover when a node fails. ML training mostly runs out of memory
   or restarts from a checkpoint. On a shared HPC system, this decides which
   jobs finish.
4. **Partitioning uneven data.** GSPMD splits a dense array into even pieces
   from annotations someone writes. With a land mask, even pieces give some
   devices only land and nothing to do; a mesh has no box to split. A
   database partitions tables by key with a cost model, which was designed
   for uneven data, and derives the partitioning from the plan (§13).

### 3.3 Experiments

Each experiment compares against the strongest usual baseline, written by
someone who knows it well, and states what counts as losing.

| | Workload | Baseline | torch-sql wins if | torch-sql loses if |
|---|---|---|---|---|
| E1 | Masked Samudra, one training step | dense PyTorch with masks, under `torch.compile` | it does measurably less work and less time per step | it is slower per step with the same mask |
| E2 | Samudra training from its Zarr archive, end to end, on the Beta cluster's file systems and later from cloud storage | Samudra's own data loaders: Dask, and its Rust loader, which already reads each rollout's planes once and prefetches to the GPU | less GPU idle time or less time to a given loss; or the same speed with preprocessing the Rust loader doesn't do, such as regridding or joining observations | the data path is slower, or the GPU waits longer |
| E3 | A mesh or graph model (GraphCast-like message passing) | JAX with `segment_sum` (its gather-and-sum primitive for graphs), and PyTorch Geometric | faster per step at equal accuracy | slower per step |
| E4 | Multi-node training (later) | GSPMD, hand-written spatial parallelism, and FSDP | better scaling, or larger models in the same memory | worse scaling at every size |

E2 can start before torch-sql has fast kernels, because the co-processor's
data path can be tested on its own (C1, §16).

### 3.4 What we expect, and when to change course

The floor is solid even if no experiment wins. ddx (reverse-mode
differentiation of relational queries) and einfold (finding fused array
algorithms in plain query plans) stand on their own as research results, and
training models where the data lives, on any engine, is useful in itself.

A clear win on an irregular or data-bound workload is a real chance, but a
minority one, and the main risk is engineering effort rather than the idea.
On dense models we expect parity at best, with any gains coming from the data
path and memory rather than from kernels.

The stopping rule: if by M4 (§16) the einfold path with hints is far behind
`torch.compile` on dense layers, and neither E1 nor E2 shows a win, torch-sql
becomes an interoperability tool rather than a performance project: the
co-processor (§6.3) stays, and the device and compile backend run PyTorch
models on XQL data and engines without a claim to speed.

---

## 4. Design principles

**P1. The SQL is the meaning; hints never are.** Every plan torch-sql writes
is plain relational SQL that gives the right answer on any engine that runs
it. torch-sql may also attach *hints* (§10), such as "this join is a 3×3
convolution", but a host that ignores every hint still gets the right answer.

**P2. Plan whole programs, never single ops.** Running each op as its own
query costs one query plan per op, which is far too slow (§9.6). On the
device, torch-sql records ops and runs them at `sync()`; through
`torch.compile`, it receives whole graphs. The natural unit is one training
step: forward pass, backward pass and optimizer update, planned together.

**P3. Fail loud; fall back only when asked.** An op torch-sql can't lower, or
a construct ddx can't differentiate, raises an error that names it. torch-sql
never gives a wrong number, and never moves a tensor to the CPU behind the
user's back. A user may allow unsupported ops or modules to run in PyTorch
instead, so that a model can adopt torch-sql a piece at a time. A fallback
then crosses the same boundary as the co-processor (§6.6), and torch-sql
reports each one: which op, how often, and how much data crossed.

**P4. Own the frontend only.** torch-sql turns PyTorch programs into plans,
facts and hints, and owns the boundary between PyTorch and XQL (§6.6).
Differentiation belongs to ddx, plan rewriting to einfold, execution to the
runtime, and registration to xarray-sql (§5). When torch-sql
needs something from those layers, the fix goes in that layer.

**P5. Public interfaces only.** torch-sql depends on its neighbours' public,
released APIs. No private imports across repositories, and no forks.

**P6. A dense tensor is complete; absence is meaningful.** A tensor that
PyTorch considers dense has a row for every index. A tensor may also be
*partial*: defined only on a domain, such as the ocean cells of a grid (§8.3).
Absent rows are never silently treated as zero.

**P7. Correct science over a faithful port.** Where the faithful port and the
scientifically correct model differ, we choose the correct one and say so
(§7.3).

---

## 5. Where torch-sql sits

XQL is organized as layers. Each is a separate project, and each is optional:

```
frontends:   SQL text · xarray (xarray-sql) · PyTorch (torch-sql)
transforms:  ddx (differentiation) · einfold (plan rewrites)
runtime:     execution, device residency, kernels, memory, scheduling, distribution
engines:     DataFusion · DuckDB · GPU engines (Sirius, GQE) · any ADBC database
```

- **Frontends** turn a user's program into plans. SQL text is the simplest
  frontend. torch-sql is the frontend for PyTorch programs.
- **Transforms** take plans and return plans. ddx returns a program that
  computes gradients; einfold rewrites plans into array algorithms.
- **The runtime** runs programs. It owns device memory, keeps data on the
  device between operators and between steps, chooses kernel implementations
  for the work einfold recognizes, schedules steps, and later distributes
  them. It is proposed in
  [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307) and
  doesn't exist yet. (It needs a name.)
- **Engines** plan and run SQL. [Sirius](https://github.com/sirius-db/sirius)
  and NVIDIA's GQE are SQL engines that run on GPUs.

torch-sql's dependencies follow from that:

| Dependency | What torch-sql uses | Required? |
|---|---|---|
| PyTorch | the dispatcher, modes, decompositions, Dynamo, `torch.library`, `nn`, `optim` | yes |
| xarray-sql | register tensors and index tables; Arrow streams; round-trip results | yes |
| ddx | gradient programs | for training |
| einfold | plan rewrites (through the engine session) | optional |
| the runtime | execution and residency, once it exists | later |
| DataFusion, DuckDB | engines | at least one |

Until the runtime exists, torch-sql runs programs through ddx's engine
backends (§12.1).

---

## 6. Three ways in

How a PyTorch program reaches XQL decides two things: how much of Samudra
has to change, and how much of a training step XQL can see. We compared seven
interfaces and kept three. The three share most of their code: the op table
(§9.3), facts and hints (§10), and one boundary between PyTorch and XQL
(§6.6).

### 6.1 What the interface has to do

The goal is production Samudra (§1.1), run elegantly and efficiently. For an
interface, that means:

1. few changes to Samudra's model code;
2. the whole step visible to the planner (data, forward pass, backward pass
   and optimizer), which H2 needs;
3. the structure the thesis depends on kept intact: masks, irregular domains
   and the data pipeline (§3.2);
4. ddx writing the backward pass (decision D2);
5. low overhead per step, once a training loop is running;
6. fit with production: DDP, Slurm jobs on the Beta cluster, adoption a piece
   at a time, and a way out when an op isn't supported;
7. fit with XQL's layers (§5).

### 6.2 The alternatives

Four more terms are needed to describe the alternatives. *StableHLO* is the
tensor IR (intermediate representation) that the XLA compiler (§3.1)
consumes; JAX produces it, and so does PyTorch through torch_xla or torchax.
A *jaxpr* is JAX's own program format. *Triton* is a language and compiler
for GPU kernels. *DTensor* is PyTorch's tensor type for arrays split across
GPUs.

| | Interface | Verdict |
|---|---|---|
| A | **A device.** A lazy `device="sql"`, as torchax and torch_xla do it (§9.1). | Kept, for the prototype. |
| B | **A `torch.compile` backend.** Dynamo captures graphs, and torch-sql lowers them with the same op table. | Kept, as the production frontend for the model. |
| C | **Lowering from StableHLO.** One translator for every framework that produces it. ddx's tests already turn jaxprs into SQL. | Dropped. StableHLO arrays are dense boxes, so masks become multiplications and the irregular-domain advantage (§3.2) is lost before planning starts. |
| D | **Writing models relationally**, in SQL as xarray-sql's `04_samudra.py` does, or with a Python builder for SQL. | Dropped for Samudra: a rewrite, and a second implementation of the model to maintain. |
| E | **An Xarray-style model API**, with layers as operations on named dimensions. | Dropped for Samudra for the same reason. It may suit new models written by scientists. |
| F | **A co-processor.** PyTorch runs the model; XQL runs the data pipeline, evaluation and chosen ops. | Kept, and built first. |
| G | **XQL as planner only.** Optimize the step relationally, then lower it to cuDNN calls, Triton kernels and DTensor placements. | Not a separate interface. It describes what the runtime does when it calls existing kernels for work einfold recognizes (§12.2); A and B become G once the runtime exists. |

Against the requirements of §6.1, the three kept interfaces trade off as
follows. The device sees every op and keeps tensors as tables between steps,
but traces every op in Python on every step, and adopting it a piece at a
time needs explicit fallbacks (P3). The compile backend needs a one-line
change, traces again only when a guard fails (§2), and its graph breaks give
a fallback for free, but it sees the step in pieces, split wherever Dynamo
breaks the graph. The co-processor is the easiest to adopt and needs no
model change, but it can't see the whole step, so it can't test H2.

### 6.3 The co-processor

PyTorch keeps running Samudra on the GPU, with DDP, and torch-sql takes the
work where a database should do better (§3.2) and the GPU's arithmetic is
not involved:

- **The data path.** A relational plan over the Zarr archive reads only the
  chunks a batch needs, normalizes and masks them, cuts rollout windows, and
  hands each DDP rank its batches as tensors on its GPU. This is experiment
  E2 (§3.3). Spike S9 built it on xarray-sql over a synthetic store shaped
  like OM4: its batches equal those of Samudra's own normalization and
  masking, but it is 1.6 to 3.2 times slower than Samudra's `cpu` loader.
  Most of the time is xarray-sql's scan from Zarr to rows, which costs about
  as much as the `cpu` loader's whole pipeline. Dropping land rows saves
  little, because the mask is a join, so land chunks are still read. The
  data path therefore depends on upstream work in xarray-sql's scan.
- **Evaluation.** Rollout metrics such as RMSE over ocean cells, means over
  ocean basins, and comparisons with observations are queries over the
  model's outputs and the dataset. Their results come back as Xarray
  Datasets through xarray-sql's round-trip.
- **Checkpoints as data.** The checkpoint reader of §7.2 makes weights
  queryable, for example to compare runs.
- **XQL ops, later.** An *XQL op* is a piece of the model written as a SQL
  query and registered as a custom op (§2), with its backward pass from ddx.
  It runs where XQL runs. Until the runtime keeps data on the GPU (§12.2), an
  XQL op on the CPU is only worth it outside the model's inner loop.

A sketch of the data path, whose details are left to the implementation:

```python
from torch_sql import data

loader = data.loader(
    xr.open_zarr("OM4.zarr"),
    windows=data.rollout(history=1, steps=4),
    normalize=(means, stds),           # Datasets of per-channel statistics
    domain="wetmask",                  # land rows dropped in SQL, zero in the tensor
    rank=rank, world_size=world_size,  # each DDP rank reads its own windows
    device="cuda",
)
for batch in loader:                   # dense tensors on this rank's GPU
    ...
```

### 6.4 The device

The device (§7 to §12) is where torch-sql's semantics get settled. It sees
every op, including ops that create tensors. A partial tensor (§8.3) carries
its domain on the tensor itself. Values stay tables between steps, so
parameters and optimizer state never leave XQL. Its cost is tracing: the
program for a step is cached (§9.6), but Python still dispatches every op on
every step, which adds up over a deep U-Net and a 4-step rollout.

### 6.5 The compile backend

```python
model = torch.compile(model, backend=torch_sql.backend)
```

Dynamo hands torch-sql an FX graph for each part of the program it captures.
torch-sql decomposes the graph to core ATen ops (§2), lowers it with the
device's op table (§9.3), and wraps the result in an autograd function (§2)
whose backward pass is ddx's program for that graph. PyTorch's autograd then
connects the graphs to each other and to any code that broke the graph, as
it does for any compiled model. Normally a `torch.compile` backend receives
the backward pass from *AOTAutograd*, the part of `torch.compile` that traces
gradients ahead of time; torch-sql takes the forward graph only, because ddx
writes the backward pass (D2).

Spike S6 (§16) built such a backend. Parameters arrive as ordinary graph
inputs, so each is an input table. Its gradients match eager autograd to
1e-16, across a graph break and under DDP on two processes. Three details
come out of it:

- **Checkpoints** arrive as a `tag_activation_checkpoint` node, a marked
  subgraph. Tracing to ATen inlines it and loses the mark, so the backend
  reads regions (§11.2) from Dynamo's graph first.
- **Partial tensors** pass through as a custom op, `torch_sql::restrict(x,
  mask)`, which Dynamo and the tracer keep as one node, and which lowers to
  the join with the mask (§8.3).
- **Shapes.** By default, Dynamo makes a size symbolic the second time it
  changes. The first version compiles with `dynamic=False`, so each batch
  size compiles once, until lowerings can take sizes as parameters.

Graph breaks are the fallback that P3 allows: code torch-sql can't lower runs
as PyTorch, and torch-sql reports it. Guards (§2) decide when a cached
program can be reused, which answers most of open question 7 (§17).

The compile backend is the production frontend for Samudra's model, once the
runtime runs XQL programs on the GPU. Before then it runs on the CPU, for
correctness: in S6, a step of a small model took about 200 ms in SQL against
0.2 ms in PyTorch, and Dynamo itself added nothing measurable.

### 6.6 One boundary, crossed both ways

The co-processor is PyTorch handing part of a program to XQL. A fallback is
XQL handing part of a program to PyTorch. Both cross the same boundary, so torch-sql
builds it once, for the co-processor, and the device and the compile backend
reuse it. It has three parts.

- **Transport** moves the bytes. It is Arrow's PyCapsule interface on the
  host and its device variant for GPU memory, with DLPack on the PyTorch side
  (§2). Transport belongs upstream: xarray-sql owns the round-trip seam (§2),
  and the runtime owns device memory (§5). Spike S4 found that today it is
  the host stream, then DLPack chunk by chunk: an Arrow array with no nulls
  reaches PyTorch without a copy, and one with nulls is refused. Nothing in
  the stack exports the device stream, and PyTorch reads no Arrow capsule,
  so a device path needs both the upstream export and a small converter.
- **Layout** turns long tables into dense tensors and back. Index columns
  hold positions (§8.1), so a complete tensor (§8.3) whose rows arrive in
  order is one buffer. A partial tensor becomes a dense tensor and a mask,
  with absent cells filled as P6 says, before DLPack sees it. Layout is
  torch-sql's, because the mapping from dimensions to strides is PyTorch's.
  S4 measured it on 10.5 million rows: a reshape of rows in order takes
  9 µs, checking that they are in order 231 ms, a scatter by position
  182 ms, and sorting them in DataFusion 3.9 s. So layout trusts an Exact
  fact (§10) that rows are in order, scatters when there is none, and never
  asks the engine to sort. DataFusion returns rows out of order today.
- **Gradients** cross through autograd functions: in one direction, ddx's
  backward pass for an XQL op or graph; in the other, PyTorch's backward pass for
  a fallback. Accounting for each crossing (P3) lives here too.

### 6.7 Order

1. **The co-processor first.** It reaches production soonest, it builds the
   boundary, and it runs E2 (§3.3).
2. **The device next**, for nanoGPT and Samudra on SQL (M1 to M3, §16),
   where the semantics are settled.
3. **The compile backend last**, as the production frontend for the model,
   behind the same op table (M5, §16).

The companion doc, [Samudra on Empire AI Beta](samudra-on-beta.md), turns
this order into a production plan.

---

## 7. User experience

The co-processor's interface is its data loader and its evaluation queries
(§6.3). This section is about running the model itself on XQL.

### 7.1 A training step

On the device, the example in §1 is the whole interface for training. The
pieces that are new to a PyTorch user are:

- **`.to("sql")`** registers each parameter and buffer as a table and returns
  tensors that refer to them.
- **`torch_sql.sync()`** plans and runs everything recorded since the last
  `sync()`. Reading a value (`loss.item()`, `print(t)`, `t.numpy()`) also
  syncs, because it needs a result.
- **`loss.backward()`** works as in PyTorch, but ddx computes the gradients
  (§11.1).

Through the compile backend (§6.5), the training loop is Samudra's own, with
`torch.compile(model, backend=torch_sql.backend)` added once.

### 7.2 Data and weights

On the device, training data enters as tables, through xarray-sql. A Samudra
batch is a window of an Xarray Dataset opened from Zarr, so it can be
registered without copying:

```python
ds = xr.open_zarr("OM4.zarr")
batch = torch_sql.from_dataset(ds.isel(time=slice(t, t + 5)),
                               dims=("time", "c", "lat", "lon"))
```

Pretrained weights take the same route. A PyTorch *checkpoint* is a file
holding a `state_dict`, a dictionary from parameter names to tensors. An
xarray extension can open a checkpoint as a Dataset with one variable per
parameter, which xarray-sql then registers. This gives the structure of a
model from running its code, and its weights from data, with no `torch.export`
step (decision D1). Where the extension lives is open (§17).

### 7.3 Changes to Samudra

The goal is very few changes to how the model is written. The co-processor
needs none: its data loader replaces Samudra's, and the model is untouched.
For the device and the compile backend, these are the changes we expect,
each with its reason:

1. **Masked operations.** Samudra zeroes land cells at its input and masks its
   output and loss, but its hidden layers compute values on land, because the
   bias, the normalization and GELU make land cells nonzero. torch-sql drops
   land rows instead, so land never enters a convolution or a normalization
   (§8.3). This is a different model, and the more correct one scientifically:
   no information leaks across coastlines, and normalization statistics cover
   only the ocean. A masked PyTorch reference implements the same model
   densely (`conv(x * mask) * mask`, and norms averaged over ocean cells), so
   torch-sql has something to match.
2. **No bfloat16.** The production configuration trains in float32, which
   SQL engines have. Samudra's higher-resolution configurations use bfloat16
   through `torch.autocast`, which SQL engines lack (open question 2, §17).
3. **No DDP, on the device only.** On `device="sql"`, the batch is a column
   of a table, and spreading it across machines is the runtime's job (§13).
   Through the compile backend, DDP stays: each rank runs its compiled
   graphs, and DDP averages the gradients that ddx writes into `.grad`.
4. **`torch_sql.sync()`** once per step, on the device only.
5. **Exact GELU.** Samudra's CappedGELU uses the error function, `erf`. It
   needs ddx's `erf` rule ([ddx#256](https://github.com/xqlsystems/ddx/pull/256)),
   and spike S8 found that neither DataFusion 54 nor DuckDB 2.0 has the
   function. Until both exist, torch-sql can use GELU's tanh form, which differs from
   it by at most 5 × 10⁻⁴, or decompose `erf` into supported functions.
6. **Gradient clipping reads back.** Samudra clips with
   `error_if_nonfinite=True`, which reads two values back every step, before
   the optimizer runs (spike S5). On the device, that is a sync in the middle
   of the step; torch-sql should offer to check at `sync()` instead.

`torch.utils.checkpoint`, in-place `+=`, `F.pad` and `F.interpolate` should
need no changes.

---

## 8. Tensors as tables

### 8.1 Schema

A tensor with `n` dimensions is a table with `n` integer index columns and one
value column:

```
i0 BIGINT, i1 BIGINT, ..., v DOUBLE
```

Index columns hold positions `0 .. size - 1`, as duckdb-zarr (XQL's Zarr
reader for DuckDB) does for a dimension with no coordinate array. A scalar is a one-row table with only `v`.
A tensor that came from Xarray keeps its dimension names (`lat`, `lon`) as
column names, which makes generated SQL easier to read. The Python side
tracks which column is which dimension.

A torch-sql tensor holds, on the Python side:

- its shape and dtype;
- a reference to a node in the recorded graph (§9.2), or, once synced, to a
  table;
- its *existence*: complete or partial (§8.3), and for a partial tensor, its
  domain.

It holds no strides. A view (§8.5) is a different query over the same table.

### 8.2 Dtypes

Float64 is the default for verification, because it makes disagreements with
PyTorch easy to see. Float32 is supported. Integer and boolean tensors are
supported for indices and masks. bfloat16 and other low-precision types are
open (§17), since neither SQL engine has them and einfold only handles
`DOUBLE` today. An op whose result PyTorch gives in an unsupported dtype,
such as `int / int`, which is float32, is refused with an error.

Integer overflow differs by engine. DataFusion wraps around, as PyTorch
does, and DuckDB raises an error (`2**62 * 4`, say). Neither gives a
wrong number silently.

### 8.3 Existence, sparsity and masks

PyTorch's tensors are dense: every index has a value. XQL tables are sparse:
an absent row means that coordinate has no value. torch-sql keeps the two
apart:

- A **complete** tensor has a row for every index. Every tensor PyTorch
  creates (`torch.zeros`, `torch.randn`, a parameter) is complete. torch-sql
  can state this as an Exact fact (§10), which lets einfold use dense kernels.
- A **partial** tensor has rows only on its *domain*, a set of indices given
  by a mask. Ops preserve the domain: adding a bias to a partial tensor adds
  it only on rows that exist. Reductions such as `mean` count only rows that
  exist.

Absence is not zero. A missing row and a zero row differ as soon as a NaN or an
infinity meets them, since `0 × ∞` is NaN in PyTorch but no product is formed
in SQL. einfold's design makes the same distinction (it calls it *group
existence*), and torch-sql follows it.

Ops whose output could reach outside the domain, such as convolution and
upsampling, restrict their output to the domain by joining with the mask. At a
coastline, a convolution's output exists only at ocean cells, and land
neighbours contribute nothing. Each level of the U-Net has its own grid, so
it needs its own mask; how to coarsen a mask (an ocean cell where any child is
ocean, or only where all are) is a scientific choice we haven't made (§17).

A user makes a tensor partial with one call, for example
`torch_sql.restrict(x, mask)`. Turning a partial tensor back into a dense
array (`.numpy()`) fills absent cells with zero, or with a value the user
chooses. This is the change §7.3 item 1 makes to Samudra.

### 8.4 Elementwise ops and broadcasting

An elementwise op is a projection over one table, or a join on all index
columns for two tables of the same shape. Broadcasting, where a tensor of
shape `(C,)` is added to one of shape `(N, C, H, W)`, is a join on the shared
index columns.

### 8.5 Views and shape ops

- **`permute` and `transpose`** rename index columns. No data moves.
- **Slicing** (`x[:, 2:5]`) is a `WHERE` on the index plus an offset.
- **`cat`** is a `UNION ALL` with offsets on the concatenated dimension.
- **`unsqueeze`, `squeeze` and `expand`** add or drop index columns; `expand`
  joins with a range of indices.
- **`reshape` and `view`** need integer arithmetic on the indices (flatten
  to one position, then split it). The arithmetic is correct SQL, but it
  produces computed keys, which einfold doesn't recognize as dimensions and
  ddx refuses to group by in some cases
  ([ddx#143](https://github.com/xqlsystems/ddx/issues/143)). torch-sql emits
  a reshape as a join with an index table (§9.5) where that avoids computed
  keys.

### 8.6 In-place ops

An in-place op such as `fts += skip` replaces the tensor's graph node with a
new one; the old value stays available to anything that already read it.
This is how torchax handles in-place ops too. An op with `out=`
(`torch.maximum(a, b, out=a)`, which `Adam(amsgrad=True)` calls) is the
same: the op without `out=`, whose result replaces `out`'s node. PyTorch
resizes an `out=` of another shape; torch-sql refuses it. PyTorch's own
checks of the `out=` call run first, so a cast it refuses (a `float64`
result into an `int64` `out=`) raises, as in PyTorch.

A view (`x[:2]`, `x.view(2, 2)`, `x.t()`) and `x.detach()` share their
base's storage in PyTorch, so they must see an in-place op on it. In
torch-sql they share a *storage* object, which holds the owner's current
node. Each alias keeps the tensor it was taken from and the function that
derives its node from that tensor's. After a write, each alias derives its
node again, from its base's current node, so a chain of aliases (a view of
an `nn.Parameter`, itself an alias of the tensor it wraps) shares one node
per value. DataFusion Accelerated, the planned DataFusion extension that
runs plans on accelerators (open question 8), has table handles that work
the same way: a query reads a handle's current buffer, not the one it was
planned against (its design, §5.2). `p.data = t` makes `p` share `t`'s
storage, as in PyTorch. An in-place op on
`x.detach()` writes `x`'s storage, which is how PyTorch's EMA of weights
(`AveragedModel`) updates its average.

Writing through a view that selects or rearranges values (`x[:, 0] += 1`)
is refused in the first version, with an error. Samudra doesn't need it.

### 8.7 Small constants

A tensor created from Python values (`torch.tensor([0.25, 0.75])`) becomes a
`VALUES` list if it is small, or a registered table otherwise.

---

## 9. Capture and lowering

### 9.1 Capturing ops

The device copies torchax's mechanism (§2):

- `import torch_sql` renames `PrivateUse1` to `"sql"`, so `device="sql"`
  and `.to("sql")` parse.
- `torch_sql.Tensor` is a wrapper subclass, allocated on PyTorch's `meta`
  device (which tracks shapes and dtypes but holds no data).

Spikes S1 and S2 found that the rename alone is fragile. A wrapper tensor
whose device is `"sql"` makes the first op that autograd records abort the
process, in C++, because PyTorch finds no device guard for it. PyTorch's
`torch.utils.backend_registration._setup_privateuseone_for_python_backend`
registers one from Python, and then autograd, checkpointing and
`torch.compile` all work. It is experimental and private, against the spirit
of P5. Allocating on `meta`, as torchax does, needs neither. torch-sql
allocates on `meta`, as torchax does (D16). PyTorch's C++ code sees `meta`,
but `torch_sql.Tensor` overrides the Python `device` property to read
`"sql"`. So code that makes a tensor on another tensor's device
(`torch.zeros(n, device=x.device)`, `t.to(x.device)`) makes a torch-sql
tensor, and `device="meta"` keeps meaning PyTorch's meta device.
- An `Environment` holds a `TorchFunctionMode`, a `TorchDispatchMode` and the
  op table. While it is active, every op on a `"sql"` tensor, and every op that
  creates one, goes through `Environment.dispatch`.

The difference from torchax is what `dispatch` does. In torchax it runs a JAX
function. In torch-sql it adds a node to the recorded graph and returns a new
lazy tensor.

On the device, capture is line by line (decision D1): the user's Python code
runs normally, with its loops, branches and in-place updates, and torch-sql
records the ops it calls. A branch on a tensor's value (`if loss.item() > 1`)
syncs at that point. The compile backend (§6.5) gets the same graph from
Dynamo instead, and the rest of this section applies to both.

### 9.2 The recorded graph

The graph is a DAG (directed acyclic graph) of relational nodes, because
values are read more than once: each Samudra block reads its input twice, once
for the convolution and once for the skip connection. The node types follow
the relational primitives ddx differentiates: a scan of a table, a projection
(elementwise expressions), a filter, a join, an aggregate, a `UNION ALL`, and
`VALUES`. Each node has a stable id, so facts and hints (§10) can refer to it.

From the graph, torch-sql writes plans. The first version writes SQL text,
because ddx's SQL surface, xarray-sql's `Context` and every engine accept it,
and it is easy to read when debugging. Writing Substrait directly is a later
option, after which ddx's `to_sql` could print SQL for each dialect. Either
way, the graph stays torch-sql's own small IR, so the output format can change
without touching the op table.

Generated names (of CTEs and saved tables) must never equal a registered
table's name. In SQL, a subquery's name hides a table of the same name, and
spike S8 got wrong numbers, with no error, from exactly that.

### 9.3 The op table

Each supported ATen op has a lowering: a Python function from input nodes to
an output node, registered with a decorator, as in torchax:

```python
@op(torch.ops.aten.mm)
def _mm(a: Node, b: Node) -> Node:
    # a(i, j) · b(j, k): join on j, SUM(a.v * b.v) GROUP BY i, k
    return contraction(a, b, contract=[(a.dims[1], b.dims[0])])
```

Ops with no lowering are decomposed with `core_aten_decompositions()`.
PyTorch's full decomposition table goes below core ATen, to its `prims` ops,
which torch-sql shouldn't need to lower (spike S1). Decompositions also emit
casts to the same dtype, which lower to nothing. An op with neither raises `OperatorNotFound`, or runs in PyTorch
if the user allows it (P3). The first
version covers what nanoGPT and Samudra use, plus the optimizer's ops.

### 9.4 Lowering patterns

| PyTorch | SQL shape |
|---|---|
| elementwise (`add`, `mul`, `exp`, `tanh`, ...) | a projection, or a join on all indices |
| `clamp`, `where` | `LEAST`/`GREATEST`, `CASE` |
| `gelu` | `erf`, which no engine has yet (§7.3 item 5); the tanh form until then |
| `mm`, `bmm`, `linear`, `einsum` | a join on the contracted index, `SUM(a * b)` grouped by the rest |
| `sum`, `mean`, `amax` | `GROUP BY` the kept indices |
| `softmax`, `log_softmax` | a `MAX` for the shift, then a `SUM` for the normalizer |
| `conv2d` (any dilation, circular or zero padding) | a join through index tables (§9.5) |
| `avg_pool2d` | a join through a pooling index table, then `AVG` |
| `interpolate` (bilinear) | a join through a tap table with weights, then `SUM` |
| `pad` | rows of an index table |
| `batch_norm`, `instance_norm`, `layer_norm` | an `AVG` for the mean, a second `AVG` for the variance, joined back |
| `cat`, slicing, `permute` | §8.5 |

Spike S8 wrote each of the rows from `conv2d` to `batch_norm` as SQL, with
Samudra's padding and upsampling copied exactly. The same SQL runs on
DataFusion and DuckDB, matches PyTorch to 3e-15 in float64, and ddx's
gradients match autograd to 3e-14.

These are the shapes einfold detects today (folds over joins, softmax
pipelines), or will detect: convolution is planned for einfold, and we will
agree on the convolution shape with einfold's maintainers as that work
proceeds (decision D7).

The norms need nothing beyond portable SQL. BatchNorm in training mode, with
statistics per channel:

```sql
WITH mean AS (SELECT c, AVG(v) AS mu FROM x GROUP BY c),
var AS (SELECT x.c, AVG((x.v - m.mu) * (x.v - m.mu)) AS var
        FROM x JOIN mean m ON x.c = m.c GROUP BY x.c)
SELECT x.n, x.c, x.lat, x.lon,
       (x.v - m.mu) / SQRT(s.var + 1e-5) * g.w + b.w AS v
FROM x JOIN mean m ON x.c = m.c JOIN var s ON x.c = s.c
JOIN bn_weight g ON x.c = g.c JOIN bn_bias b ON x.c = b.c
```

InstanceNorm is the same with `GROUP BY n, c`. The variance is a second pass
on purpose: `AVG(v * v) - AVG(v)^2` reads `x` once fewer, but loses precision
through cancellation. On a partial tensor, both averages cover only the rows
that exist.

### 9.5 Index tables

An *index table* lists, for an op whose inputs and outputs are related by
position, which input positions feed which output positions. For a 3×3
convolution along latitude with dilation `d`, `idx_lat(out_lat, ky, in_lat)`
has a row for each output position, filter tap and input position it reads.
The convolution is then a join in which every condition is an equality:

```sql
SELECT w.co AS c, ia.out_lat AS lat, io.out_lon AS lon, SUM(x.v * w.v) AS v
FROM x
JOIN idx_lat ia ON x.lat = ia.in_lat
JOIN idx_lon io ON x.lon = io.in_lon
JOIN weight w ON x.c = w.ci AND ia.ky = w.ky AND io.kx = w.kx
GROUP BY w.co, ia.out_lat, io.out_lon
```

The geometry lives in the rows. Circular padding in longitude is a matter of
which rows exist (output 0 reads input 359). Zero padding in latitude is
absent rows. Dilation is a different table. Pooling and bilinear upsampling
use the same form, with a weight column for upsampling. Samudra's upsampling
is separable: one weight table for latitude and one for longitude, with its
circular padding and cropping folded into the longitude weights.

This follows a principle xarray-sql already states for geospatial work:
arrays compute the geometry (the weights), and SQL applies it (the join).
It has three practical benefits:

- every join is an equality, which is the shape einfold builds dimensions from;
- ddx differentiates it with its existing join and aggregate rules, with no
  computed keys; the gradient with respect to the input is the same join read
  in the other direction;
- the index tables are small (proportional to the grid size times the filter
  size), are computed once in NumPy from the op's parameters, and are
  registered and cached by shape.

On a partial tensor, the same SQL over ocean rows only equals the dense
masked reference, `conv(x * mask) * mask` with norms over ocean cells,
forward and backward (S8). It also shows why absence isn't zero: with one
NaN in a land cell, the dense reference gives NaN at 36 ocean outputs,
because `NaN × 0` is NaN, while the SQL never reads that row.

Whether index tables belong in torch-sql or in a shared XQL helper that
einfold could also use is open (§17).

### 9.6 When to run, and what to save

At `sync()`, torch-sql turns the recorded graph into a program: a sequence of
plans, with some intermediate results saved as tables between them. This is
the same shape as a ddx program, and it is forced by how engines plan
queries. A single query for a whole Samudra step would be very large, and
DataFusion plans a CTE (a named subquery, `WITH x AS (...)`) again at every
place that reads it, so a value read twice in each of many layers multiplies
the plan. xarray-sql measured this on nanoGPT: one GPT layer's loss is 13,756
physical operators on DataFusion against 126 on DuckDB, and two layers run out
of 8 GB while planning.

Through the compile backend, the same happens when Dynamo hands over a graph,
and the compiled program runs each time that graph is called.

Spike S8 measured the same on Samudra's blocks. For a stack of ConvNeXt-like
blocks written as one query:

| Blocks | DataFusion operators | DataFusion plan + run | DuckDB operators | DuckDB plan + run |
|---|---|---|---|---|
| 1 | 441 | 0.2 s | 75 | 0.05 s |
| 2 | 7,978 | 6.4 s | 151 | 0.09 s |
| 3 | 136,107 | 212 s | 227 | 0.20 s |

DataFusion grows about 17 times per block, because each BatchNorm reads its
input three times; DuckDB grows by 75 operators per block.

So torch-sql chooses *cut points*, where a value is saved as a table instead
of written into every query that reads it. The default is to save the input
of every norm and every other value read more than once. On DataFusion this
is required, not an optimization
([xarray-sql#267](https://github.com/xqlsystems/xarray-sql/issues/267)). Choosing cut points is the same trade-off as
gradient checkpointing (memory against recomputation), and the runtime should
eventually make it, using memory budgets for the hardware it runs on.

The program for a step depends only on the graph's structure, the shapes and
the dtypes, not on the values. Values that change every step without
changing the graph, such as the learning rate, Adam's bias corrections and
the EMA decay (Python numbers or 0-dimensional CPU tensors, spike S5), are
parameters of the program, never literals in its SQL. torch-sql caches it, keyed on a hash of those,
so a training loop builds it once and runs it every step, with prepared
statements where the engine has them (xarray-sql's `Context.prepare`, ddx's
`prepare`).

---

## 10. Facts and hints

torch-sql knows things about its plans that are hard to recover from SQL. When
a plan comes from `aten.convolution`, torch-sql knows it is a convolution, with
which stride, dilation and padding. Writing it as SQL loses that knowledge,
and an optimizer then has to rediscover it. torch-sql passes it along in two
forms, both through einfold's side-channel fact table, keyed by plan node:

- **Facts** are true statements that an optimizer may rely on: extents of each
  index, that a complete tensor has every row, that index columns are unique,
  and that positions are a simple range `0 .. n - 1`. Only Exact facts may
  affect correctness, as einfold requires.
- **Hints** are provenance: the ATen op that produced a node, its parameters,
  and the module path it came from (`unet.layers.3.convblock.0`, recorded with
  `nn.Module` hooks). A host may use a hint to choose a kernel, but must not
  need it for the meaning (P1).

This gives the immediate question (§1) a clean experiment: run the same
program on einfold with hints off and with hints on. Without hints, any speed
comes from the SQL alone. The difference measures how much structure SQL
loses.

Hints must also reach the backward pass. The gradient of a convolution is two
different computations, one with respect to its input and one with respect to
its weights, and GPU libraries have a separate kernel for each. ddx writes the
backward program, so ddx must record which forward node and role each
backward step comes from
([ddx#259](https://github.com/xqlsystems/ddx/issues/259)).

---

## 11. Gradients and the training step

### 11.1 `backward()` calls ddx

PyTorch normally computes gradients with *autograd*: while the forward pass
runs, it records how each tensor was computed, and `loss.backward()` walks
that record in reverse. On `device="sql"`, ddx computes gradients instead
(decision D2), because it writes them as SQL programs with a tape designed for
databases, and it refuses what it can't differentiate.

`torch_sql.Tensor` overrides `backward()`. It finds the parameters the loss
depends on (leaf tensors with `requires_grad`), asks ddx for the gradient of
the loss with respect to all of their tables at once, and sets each
parameter's `.grad` to a lazy tensor referring to its gradient. One call,
`ddxdb.ad.grad(ctx, loss, ["w1.val", "w2.val", ...])`, gives one program
with one backward pass; asking per parameter, as SQL's `grad()` does,
repeats the forward pass for each. PyTorch's autograd still
records its graph during the forward pass, but nothing ever runs it.
`torch.optim` reads only `.grad`, so it works unchanged. Spike S1 confirmed
this: ddx's gradients match autograd to 1e-16, PyTorch accepts a subclass as
`.grad` (it checks device, dtype and size), `nn.Parameter` keeps the
subclass, and autograd's graph never runs. Through the compile backend, ddx's program is instead
the backward pass of an autograd function (§6.5), and PyTorch's autograd
calls it.

### 11.2 Checkpointing as ddx regions

`torch.utils.checkpoint(block, x)` tells PyTorch to save only the block's
inputs and recompute everything inside it during the backward pass, trading
compute for memory. Samudra checkpoints every layer. ddx is designing a
recompute option in [ddx#251](https://github.com/xqlsystems/ddx/issues/251);
torch-sql needs it in the form of *regions*: a named relation inside which
nothing is saved except what a policy names. torch-sql replaces
`torch.utils.checkpoint` while its environment is active, records the block's
ops as a region, and passes ddx the region's name and a policy.

Spike S2 found how. Samudra calls `torch.utils.checkpoint.checkpoint` through
the module, and some of its models use `apply_activation_checkpointing`,
which wraps modules in a `CheckpointWrapper`. While its environment is
active, torch-sql replaces the module's function, and at `.to("sql")` the
`checkpoint_fn` of each `CheckpointWrapper`, with one that opens a region
named after the block's module path, calls the block, and closes the region.
A call bound before that (`from torch.utils.checkpoint import checkpoint`)
would be missed, so torch-sql also raises when checkpoint's saved-tensor
hook is active with no region open. Through the compile backend, regions
come from Dynamo's `tag_activation_checkpoint` nodes instead (§6.5).

Spike S3 found that a region can't be named by a CTE or a view: neither
DataFusion's nor DuckDB's Substrait producer keeps those names. A marker
function, such as `ddx_name(expr, 'unet.layers.3')`, survives both, so ddx
needs one (ddx#251).

### 11.3 Stopping gradients

`x.detach()` and `torch.no_grad()` become `ddx_stop_gradient`, ddx's marker for
a value that gets no gradient. It must be registered on the session first
(`ddxdb.ad.register_stop_gradient`, spike S3), which takes only DOUBLE
([ddx#260](https://github.com/xqlsystems/ddx/issues/260)). Samudra uses this to cut the rollout's gradient
every few steps.

### 11.4 Forward values besides the loss

A step needs some forward values that aren't differentiated: BatchNorm's
batch statistics, to update its running mean and variance, and Samudra's
per-channel losses, which set its loss weights. ddx would return these from
the same forward pass as auxiliary outputs, as `has_aux=True` does in JAX
([ddx#258](https://github.com/xqlsystems/ddx/issues/258)). Until then they
are separate queries.

### 11.5 The optimizer

`torch.optim.Adam` and `clip_grad_norm_` are ordinary tensor ops, so torch-sql
lowers them like any other. PyTorch's optimizers can use `_foreach_*` ops,
which apply one op to a list of tensors at once, or fused kernels. Spike S5
found that on a tensor subclass the default is the plain loop, one tensor at
a time, while `foreach=True` sends 19 different `_foreach_*` ops, none of
which PyTorch decomposes. One rule lowers all but one of them: apply the
plain op to each tensor. The exception, `_foreach_norm`, is
`linalg_vector_norm` per tensor. torch-sql refuses `fused=True`. Samudra's
whole optimizer path (Adam, its cosine schedule, clipping and EMA) needs 18
ops. Updating a parameter in place replaces its graph node
(§8.6); at `sync()`, the new parameter tables are saved and the old ones
dropped.

### 11.6 The rollout

A 4-step rollout is a Python loop, so capture, on the device or through
Dynamo, unrolls it into a graph four times as deep. ddx shares repeated subplans instead of copying
them, so the backward program doesn't blow up, but every saved value is saved
four times. Regions (§11.2) are what keep that within memory.

---

## 12. Engines and the runtime

### 12.1 Before the runtime exists

The first versions run programs through what already exists: xarray-sql to
register tables and read results, and ddx's engine backends (DataFusion, and
DuckDB through `ddxdb.duckdb`) to run programs. einfold joins through
DataFusion's session extensions.

### 12.2 The runtime contract

Once the runtime exists, torch-sql hands it a program: the plans, their facts
and hints, which inputs persist across steps (parameters and optimizer state),
and which outputs the user asked for. The runtime returns handles to the
results. A synced torch-sql tensor then refers to a table the runtime owns,
which may live in GPU memory. Reading it (`.numpy()`) is xarray-sql's
round-trip.

Two interfaces from other layers make this work without private imports (P5):

- ddx's `Backend` trait (return rows, save a table, drop a table, read a
  schema), made public and versioned, which the runtime implements
  ([ddx#157](https://github.com/xqlsystems/ddx/issues/157));
- einfold's dense backend interface, through which einfold hands recognized
  work to the runtime's kernels, which may decline it
  ([einfold#156](https://github.com/xqlsystems/einfold/pull/156)).

How dense data moves between operators is the runtime's question. Arrow's C
Device Data Interface (§2) describes Arrow arrays in GPU memory, and Arrow has
a canonical
[fixed-shape tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
type, whose values are dense arrays of one shape. If dense intermediates can
travel between an engine's operators as Arrow values of that kind, the runtime needs no second implementation of SQL's
semantics. Spike S4 found that a fixed-shape tensor column keeps its type
through DataFusion's projections, filters and sorts, and a projection keeps
the same buffer. DuckDB drops the type to a plain list of fixed size, which
keeps the values but loses the shape. Whether a GPU can read such values in
place from Grace memory, on the Beta cluster, is still untested. The same
transport carries tensors across the boundary with PyTorch (§6.6).

### 12.3 Engine order

1. **DataFusion**, with einfold. This is where xarray-sql's `Context` and
   einfold live. They need one DataFusion version: today xarray-sql 0.6.0
   pins 54, and ddxdb's and einfold's DataFusion extras pin 55 (spike S7).
2. **DuckDB**, which plans deep queries far more compactly (§9.6). ddx runs
   on DuckDB through its community Substrait extension, which exists for
   DuckDB 1.5 but not yet 2.0 (spike S7), so torch-sql holds DuckDB at 1.5
   for now.
3. **Any Arrow or ADBC engine**, through SQL text in each engine's dialect.
   Engines with no Substrait producer need ddx's plan for them
   ([ddx#159](https://github.com/xqlsystems/ddx/issues/159)).

---

## 13. Distribution (later)

torch-sql doesn't distribute anything, but its plans must not prevent
distribution, and it should describe them in a way a distributed runtime can
use. A database distributes a query by *partitioning*: splitting each table
by some key across machines, and moving rows between machines where a join
or aggregate needs rows that live elsewhere. In a training step, the choice
of partitioning key gives the familiar parallelism strategies:

- by batch index: data parallelism, what DDP does;
- by a weight's index: weight sharding, what FSDP does;
- by grid tile (`lat`, `lon`): *domain decomposition*, which is how numerical
  ocean and climate models run on HPC. Each node owns a tile of the grid and
  exchanges a thin border with its neighbours, the *halo*. For a convolution
  through an index table (§9.5), the rows that cross a tile edge are the
  halo, so a partitioned join moves only those.

A query planner can choose a partitioning for each join separately, using the
whole step's plan. PyTorch has no equivalent of domain decomposition for a
CNN; H1 is the hypothesis that this is where SQL gives a better distribution
than FSDP. For a partial tensor (§8.3), land rows never move at all.

On the Beta cluster, the hardware favours this. Each NVL72 rack joins 72 GPUs
with NVLink, NVIDIA's GPU-to-GPU interconnect, so halo exchange between tiles
in one rack never touches the slower InfiniBand network between racks.
Samudra's production runs use DDP over 36 GPUs (§1.1), which one rack could
hold.

torch-sql's part is to state partitioning as facts (aligned with einfold's
tiling and with Zarr chunks, so that a chunk is a partition), never to write
device placement into the SQL, and to keep collective operations such as
DDP's `all_reduce` out of the program.

---

## 14. Testing

- **Per op.** PyTorch's *OpInfo* database lists sample inputs for most ATen
  ops and is how PyTorch tests its own backends. torch-sql reuses it: each
  lowered op runs on OpInfo's samples and must match PyTorch on the CPU in
  float64. Gradients are checked against PyTorch's autograd the same way.
- **Per model.** nanoGPT is checked step by step against PyTorch and against
  xarray-sql's `benchmarks/ml/02_nanogpt.py`. Samudra is checked against a
  masked PyTorch reference (§7.3) and against xarray-sql's
  `benchmarks/ml/04_samudra.py`.
- **Tolerance.** Results must agree within einfold's floating-point bound,
  `(n − 1) · u · Σ|terms|` (with `u` the unit roundoff), rather than a fixed
  tolerance.
- **Refusals.** Each unsupported op and construct has a test that it raises
  (P3).
- **Partial tensors.** Tests with missing rows, including NaN and infinity
  next to absent rows (§8.3).
- **Frontends.** The device and the compile backend must write the same
  program for the same model. For the same windows, the co-processor's batches
  must match those of Samudra's own loader: masks exactly, normalized values
  within the floating-point bound.
- **Fallbacks.** Each allowed fallback is reported, and a fallback that isn't
  allowed raises (P3).
- **Spikes.** The spikes of §16 exit non-zero when a finding stops holding,
  so they also check that a newer PyTorch, ddx or engine hasn't changed an
  answer.

---

## 15. Package layout

A Python package, `torch_sql`, for Python 3.12 and later:

```
torch_sql/
  __init__.py          # public API
  boundary/            # PyTorch <-> XQL, shared by all three frontends (§6.6)
    layout.py          #   long tables <-> dense tensors and masks
    autograd.py        #   autograd functions across the boundary
    report.py          #   accounting for crossings and fallbacks
  data/                # the co-processor's data path (§6.3)
  eval/                # the co-processor's evaluation queries (§6.3)
  device/              # the device frontend (§6.4, §9.1)
    tensor.py          #   torch_sql.Tensor; renames PrivateUse1
    environment.py     #   modes, dispatch, configuration
  compile/             # the compile backend (§6.5)
  graph.py             # the recorded DAG (§9.2)
  ops/                 # lowerings, by family (§9.3, §9.4)
  index_tables.py      # §9.5
  sql.py               # writes SQL from the graph
  program.py           # cut points, caching, sync (§9.6)
  gradients.py         # backward passes through ddx (§11)
  facts.py             # facts and hints (§10)
  engines.py           # running programs until the runtime exists (§12.1)
```

The modules listed after `compile/` are shared by the device and the compile
backend.

Extras keep each layer optional (H3): `torch-sql[ddx]` for training,
`torch-sql[einfold]`, `torch-sql[duckdb]`.

---

## 16. Milestones

**M0. Spikes (done).** Each answers one question with a small program in
[docs/spikes](spikes/README.md), whose docstring holds the evidence. All
pass.

- **S1.** Can a wrapper subclass override `backward()` so that ddx computes
  the gradients and `torch.optim` works unchanged? Yes (§11.1). A `"sql"`
  device needs more than a rename (§9.1).
- **S2.** How can torch-sql replace `torch.utils.checkpoint` cleanly while
  its environment is active? Replace the module's function and each
  `CheckpointWrapper`'s, and guard with checkpoint's hook (§11.2).
- **S3.** Do CTE and view names survive DataFusion's and DuckDB's Substrait
  producers? No. Regions need a marker function (§11.2).
- **S4.** Can dense arrays pass between DataFusion's operators, and from
  there to PyTorch, as Arrow values without copies? On the host, chunk by
  chunk, yes; no device path exists yet (§6.6, §12.2). Whether a GPU can
  read them in place from Grace memory is still to test on Beta.
- **S5.** Which `_foreach_*` ops do Adam and `clip_grad_norm_` call, and do
  PyTorch's decompositions cover them? 19, none decomposed; one rule lowers
  them (§11.5).
- **S6.** Can a `torch.compile` backend take Dynamo's forward graph, run it
  as SQL, and supply ddx's backward pass through an autograd function, with
  DDP still averaging the gradients? How do partial tensors pass through
  Dynamo? Yes, and as a custom op (§6.5).
- **S7.** Do xarray-sql, ddx, einfold and their engines install in Samudra's
  container for the Beta cluster? Every package has an ARM wheel; DataFusion
  versions conflict (§12.3). Not yet run on ARM.
- **S8.** Do Samudra's building blocks lower to portable SQL through index
  tables, match PyTorch, and differentiate with ddx, masked and unmasked?
  Yes, except `erf` (§7.3, §9.4, §9.5). A deep stack can't be one query on
  DataFusion (§9.6).
- **S9.** How fast is the co-processor's data path today? Correct, and 1.6
  to 3.2 times slower than Samudra's `cpu` loader, mostly in xarray-sql's
  scan (§6.3).

**M1. nanoGPT.** First a two-layer MLP (a small fully connected network). Then
nanoGPT on `device="sql"`, matching PyTorch on the CPU and `02_nanogpt.py`
step by step, on DataFusion and on DuckDB.

**C1 to C3. The co-processor**, in parallel with M1 to M3. Each step is
usable in production on its own:

- **C1. The data path.** Samudra's training, unchanged, with its data loader
  replaced by torch-sql's (§6.3). Measure throughput and GPU idle time
  against Samudra's Dask and Rust loaders. This is experiment E2 (§3.3), and
  it tests §3.2's second claim before any kernel work.
- **C2. Evaluation.** Samudra's rollout metrics as queries, matching its
  current evaluation.
- **C3. Checkpoints as data**, through the checkpoint reader (§7.2).

**M2. Samudra inference.** A masked Samudra at `04_samudra.py`'s small size
(widths 4 to 10, a 10 × 20 grid), matching the masked PyTorch reference.

**M3. Samudra training.** A 4-step rollout with checkpointed layers and Adam.
Needs ddx's regions (ddx#251), auxiliary outputs (ddx#258) and `erf`
(ddx#256).

**M4. The hints experiment.** Samudra and nanoGPT on einfold with hints off
and on, with times and peak memory, compared with PyTorch.

**M5. The compile backend.** Samudra through `torch.compile` with
torch-sql's backend, matching M3, with DDP.

After M5: the runtime (with others), GPUs, and distribution.

---

## 17. Open questions

1. **Coarse masks.** At each U-Net level, is a coarse cell ocean when any of
   its children is, or only when all are? This changes the model, so it is a
   scientific choice.
2. **Low precision.** How to support bfloat16 and other low-precision types,
   which SQL engines lack and GB200 hardware is built for. Probably through
   the runtime's kernels, with the SQL types as the reference.
3. **Random numbers.** Dropout and Samudra's DropPath need random numbers, and
   ddx refuses volatile functions such as `random()`. A hash of a seed, the
   step and the indices gives reproducible randomness in portable SQL; this
   needs agreement with ddx.
4. **Where index tables live.** In torch-sql, or in a shared XQL helper that
   einfold's convolution support also uses.
5. **Where the checkpoint reader lives.** An xarray backend for PyTorch
   checkpoints could live in torch-sql, in xarray-sql, or on its own.
6. **How torch-sql calls ddx.** SQL text through ddx's SQL surface is simplest.
   Plans would avoid printing and parsing SQL. Which ddx API fits best needs
   checking against `ddxdb.ad`.
7. **Cache keys.** Which properties of the graph a cached program must check
   before it is reused (shapes, dtypes, existence, the set of parameters).
   Through the compile backend, Dynamo's guards check most of them.
8. **The runtime's name and home.** Decided: it is DataFusion Accelerated
   (`datafusion-accelerated`), a DataFusion extension under xqlsystems for
   now.
9. **How much to fall back.** Whether a fallback is allowed per op, per
   module, or per graph, and what the default is.
10. **Splitting data across ranks.** Whether the co-processor's loader
    reproduces Samudra's sampler exactly (same windows on the same rank), or
    may choose its own split, for example by Zarr chunk.
11. **How to register the device.** Decided: the `meta` device, as torchax
    does (D16).
12. **Pooling a partial tensor.** Whether an average pool over a coastline
    averages only its ocean children, or counts land as zero. Like open
    question 1, this changes the model.

---

## Decision log

**D1. Capture as the program runs (revised 2026-11-05).** torch-sql captures
ops as the user's code runs, in two ways over one op table: line by line on
the device, through a tensor subclass and modes as torchax does, and through
Dynamo as a `torch.compile` backend (§6.5). The first version chose line by
line only. We added the compile backend after comparing interfaces (§6),
because it traces once instead of on every step and its graph breaks give
production a fallback. There is no `torch.export` path. What `torch.export`
would add (a model without its Python code) is mostly weights, and weights
can be read into SQL directly from checkpoints (§7.2).

**D2. Gradients from ddx, not PyTorch's autograd.** Lowering PyTorch's own
backward ops to SQL would work without ddx, but would give up ddx's tape
design, its checks and its refusals, and would leave XQL's differentiation in
two places.

**D3. Plan whole programs.** Running each op as a query costs one plan per
op. nanoGPT's hand-written SQL already spends most of each 5.4 ms ddx step on
planning (about 4 ms), with 76 to 77 steps per training step. A whole step
must be planned at once (P2), which is why the device is lazy and why the
compile backend suits torch-sql.

**D4. Missing rows mean absence, and Samudra drops land rows.** Absence is not
zero (§8.3). Samudra on torch-sql masks land throughout, which is a different
and more correct model than the original, checked against a masked PyTorch
reference.

**D5. Hints as facts; the SQL is the meaning.** torch-sql passes what it knows
about each op through einfold's fact table. A host that ignores hints still
gets the right answer, which also lets us measure what hints are worth (M4).

**D6. A runtime layer owns execution.** torch-sql, ddx and einfold produce and
rewrite plans. Running them, keeping data on devices, choosing kernels and
distributing work belong to a separate runtime, proposed in xarray-sql#307.
einfold's own design rules out execution ("einfold rewrites query plans; it
never executes them").

**D7. Convolution through index tables.** Index tables give equality-only
joins that einfold and ddx handle with existing rules, and they match how
xarray-sql already does upsampling and geospatial regridding. Spike S8
showed that they are correct, forward and backward, for every block Samudra
uses. What is still open is speed: einfold will add convolution support, and
the shape it recognizes is to be settled together.

**D8. Engine order.** DataFusion, then DuckDB, then any Arrow or ADBC engine.
DuckDB stays at 1.5 until ddx's Substrait route works on 2.0 (§12.3).

**D9. Positional index columns.** Index columns hold positions `0 .. n - 1`,
as in duckdb-zarr and xarray-sql's default indexes, and keep their Xarray
names when they have them.

**D10. The thesis is tested, with a stopping rule.** The relational model is
expected to win on irregular and data-bound workloads, not on dense kernels.
Experiments E1 to E4 (§3.3) decide it against the strongest usual baselines,
and §3.4 says when to treat torch-sql as an interoperability tool instead.

**D11. Three frontends, built in order.** torch-sql is a co-processor, a
device and a compile backend (§6), sharing one op table and one boundary. The
co-processor comes first because it reaches production soonest and builds
the boundary. It lives in torch-sql rather than in a separate package because
it crosses the same boundary as a fallback, in the other direction (§6.6).

**D12. Fallback only when asked (revises P3).** torch-sql may run
unsupported ops or modules in PyTorch, but only when the user allows it, and
it reports every fallback. It still never gives a wrong number.

**D13. Transport upstream, layout here.** Moving bytes between XQL and
PyTorch uses Arrow's PyCapsule interface and its device variant, owned by
xarray-sql and the runtime, with DLPack on the PyTorch side. torch-sql owns
only the mapping between long tables and dense tensors, and the gradients
across the boundary (§6.6).

**D14. Regions are marked with a function (2026-11-05).** Neither engine's
Substrait producer keeps CTE or view names (spike S3), so torch-sql marks
each checkpointed block with a function call that names it, and ddx learns
that function (ddx#251).

**D15. Cut points at every norm (2026-11-05).** On DataFusion, a stack of
three ConvNeXt blocks as one query is 136,107 operators (spike S8). torch-sql
saves the input of every norm, and every other value read more than once, as
a table between plans (§9.6).

**D16. Allocate on `meta`, as torchax does (Alex, 2026-11-08).** A
torch-sql tensor is a wrapper subclass on PyTorch's `meta` device. Its
Python `device` property reads `sql` (§9.1). The alternative, PyTorch's
private Python-backend setup, is experimental (P5). torch-sql follows
torchax's choice here, and will revisit it if torchax does.

**D17. Defaults for now, revisited with experience (Alex, 2026-11-08).**
These choices from the device's first version stand until a model, a user
or a profile calls for something else. Each has an issue, labelled
`provisional`, which says what would reopen it:
- division with a rounding mode is refused
  ([#12](https://github.com/xqlsystems/torch-sql/issues/12));
- in-place ops that change a tensor's shape are refused, so 1-D `matmul`
  is too ([#13](https://github.com/xqlsystems/torch-sql/issues/13));
- writing through a view is refused (§8.6,
  [#14](https://github.com/xqlsystems/torch-sql/issues/14));
- at a tie, `maximum` and `minimum` give the gradient to the first
  argument, where PyTorch splits it
  ([#15](https://github.com/xqlsystems/torch-sql/issues/15));
- random numbers are drawn on the CPU and uploaded, as in DataFusion
  Accelerated's design (open question 3,
  [#16](https://github.com/xqlsystems/torch-sql/issues/16));
- every op runs on the `meta` device before it is lowered
  ([#17](https://github.com/xqlsystems/torch-sql/issues/17)).
