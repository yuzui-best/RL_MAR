# Local sentence encoder provenance

The local `all-MiniLM-L6-v2` snapshot is the Sentence Transformers model
`sentence-transformers/all-MiniLM-L6-v2`, licensed Apache-2.0 as declared in
its model card. The public Git selection excludes the snapshot directory and
weight files. Obtain the model separately from
[Sentence Transformers on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
review its terms, and set `LOCAL_MODEL_PATH` to its local directory.
Runtime loading does not automatically download missing files.

- Recorded local snapshot path: `framework/models/all-MiniLM-L6-v2`
- Path-and-size tree SHA256: `9b7edabe9fdddc5c27bdccc30f58ff8906869c21a295e8e8fd06216a0a9b42f8`
- `model.safetensors` SHA256: `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`

Smoke audit plans record a content-based encoder tree digest. Loading is
performed with `local_files_only=True`; a missing directory fails before any
provider process or download attempt begins.
