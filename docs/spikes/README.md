# Spikes

A *spike* is a small program that answers one question the
[design](../design.md) can't settle on paper. Each one here is a single
script (S7 has a helper), whose docstring states the question, the sections
of the design it backs, how to run it, and its findings. Each prints every
check as `PASS` or `FAIL`, and exits non-zero if a finding stops holding, so
a newer PyTorch, ddx or engine that changes an answer shows up as a failure.

The code is here only in service of the design. It isn't the start of
torch-sql's package, and none of it should be imported from elsewhere.

## Running them

From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --project docs/spikes python docs/spikes/s1_backward_ddx.py
```

`docs/spikes/pyproject.toml` pins the environment: PyTorch 2.15 (CPU),
ddxdb 0.4.6, xarray-sql 0.6.0, einfold 0.2.0, DataFusion 54 and DuckDB 2.0.
On NixOS, first put `libstdc++` and `zlib` on `LD_LIBRARY_PATH`, as the
XQL repositories' `flake.nix` files do. S7 needs network access, and takes
an optional path to a Samudra checkout.

## Results

All nine spikes pass, run on 2026-11-05 on an x86_64 Linux machine with no
GPU in use. Section numbers refer to [design.md](../design.md).

| Spike | Question | Answer | Design |
|---|---|---|---|
| [S1](s1_backward_ddx.py) | Can a wrapper subclass override `backward()` so that ddx computes the gradients and `torch.optim` works unchanged? | Yes. ddx's gradients match autograd to 1e-16, and SGD and Adam match plain PyTorch over 5 steps. A `"sql"` device needs PyTorch's private Python-backend setup, not just a rename. | §9.1, §11.1, §11.5 |
| [S2](s2_checkpoint_regions.py) | How can torch-sql replace `torch.utils.checkpoint` cleanly? | Replace the module function and each `CheckpointWrapper.checkpoint_fn` while the environment is active, and use checkpoint's saved-tensor hook as a guard against calls bound before that. | §11.2 |
| [S3](s3_substrait_names.py) | Do CTE and view names survive the Substrait producers? | No, on DataFusion 54 or DuckDB 1.5.6. A marker function survives both, so ddx regions need one. | §11.2 |
| [S4](s4_arrow_transport.py) | Can dense arrays pass from DataFusion to PyTorch as Arrow without copies? | On the host, yes: per chunk, through DLPack, when there are no nulls. Nothing in the stack exports an Arrow device stream, and PyTorch reads no Arrow capsule. Fixed-shape tensor columns survive DataFusion's operators but not DuckDB's. Grace and CUDA were not tested. | §6.6, §12.2 |
| [S5](s5_foreach_ops.py) | Which `_foreach_*` ops do Adam and `clip_grad_norm_` call, and are they decomposed? | 19 `_foreach_*` ops, none decomposed by PyTorch; one generic rule (apply the plain op per tensor) covers them, except `_foreach_norm`. Samudra's optimizer path needs 18 ops, and clipping reads two values back every step. | §11.5, §9.6 |
| [S6](s6_compile_backend.py) | Can a `torch.compile` backend run the forward graph as SQL and get its backward pass from ddx, with DDP? | Yes, including graph breaks, DDP on 2 processes and a `torch_sql::restrict` custom op. Checkpoints arrive as marked subgraphs, which must be read before tracing to ATen. | §6.5 |
| [S7](s7_arm_packaging.py) | Does the XQL stack install in Samudra's ARM container? | Every package has an aarch64 wheel. The blocker is versions: xarray-sql 0.6.0 pins DataFusion 54, and ddxdb's and einfold's extras pin 55. DuckDB 2.0 has no Substrait extension. Not run on ARM. | Samudra plan, stage 0 |
| [S8](s8_index_table_conv.py) | Do Samudra's building blocks lower to portable SQL through index tables, and differentiate? | Yes: convolutions, pooling, upsampling and norms match PyTorch to 3e-15 on DataFusion and DuckDB, gradients to 3e-14, and the land-masked versions too. No engine has `erf`. One query per block stack blows up on DataFusion (136,107 operators for 3 blocks). | §9.4, §9.5, §9.6 |
| [S9](s9_data_path.py) | How fast is the co-processor's data path today? | Correct, but 1.6 to 3.2 times slower than Samudra's `cpu` loader, mostly in xarray-sql's scan from Zarr to rows. | §6.3, Samudra plan, stage 1 |
