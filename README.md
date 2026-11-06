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

Nothing is built yet. Start with the docs:

- [docs/design.md](docs/design.md): the design.
- [docs/samudra-on-beta.md](docs/samudra-on-beta.md): the production plan for
  Samudra on Empire AI's Beta cluster.
- [docs/spikes/](docs/spikes/README.md): small programs that answer the
  design's open questions, with their results.
