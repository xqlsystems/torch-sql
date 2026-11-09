`model.py` is nanoGPT's model, copied unchanged from
[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)
at commit `3adf61e`, under its MIT license (`LICENSE`). nanoGPT is Andrej
Karpathy's small, readable GPT-2. The tests run it unchanged on
`device="sql"`, except that they replace its exact GELU with the tanh form
(see `tests/test_nanogpt.py`).
