# torch-sql

torch-sql is a skunkworks prototype that runs PyTorch programs, or parts of
them, as SQL on [XQL](https://xql.systems), a group of database projects for
scientific arrays. It has three frontends:

- a co-processor, which leaves the model in PyTorch and takes over the
  training data pipeline and evaluation;
- a device, `device="sql"`, on which every tensor is a table;
- a `torch.compile` backend.

Its first real model is [Samudra](https://github.com/m2lines/Samudra), an
ocean emulator.

The first milestone, M1 in the design, is under way: a two-layer MLP, then
nanoGPT (Andrej Karpathy's small GPT), trained on `device="sql"`. Start with
the docs:

- [docs/design.md](docs/design.md): the design.
- [docs/samudra-on-beta.md](docs/samudra-on-beta.md): the production plan for
  Samudra on Empire AI's Beta cluster.
- [docs/spikes/](docs/spikes/README.md): small programs that answer the
  design's open questions, with their results.

## Development

torch-sql uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --all-extras   # PyTorch (CPU), ddxdb, DataFusion and DuckDB
uv run pytest          # every test runs on both engines
uv run ruff check && uv run ruff format --check
```

On NixOS, first put `libstdc++` and `zlib` on `LD_LIBRARY_PATH`, as the XQL
repositories' `flake.nix` files do.

## Example

A two-layer MLP on `device="sql"`, run on DuckDB:

```python
import torch, torch_sql

with torch_sql.Environment("duckdb"):
    mlp = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
    mlp = mlp.double().to("sql")
    y = mlp(torch.randn(4, 3, dtype=torch.float64, device="sql"))
    print(y)  # runs one SQL query on DuckDB
```
