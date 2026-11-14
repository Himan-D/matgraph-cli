# Publishing matgraph-cli to conda-forge

conda-forge is a second install base: many materials scientists use conda, not
pip. The recipe is ready in `meta.yaml` — submission is a short manual flow
(conda-forge requires a PR from your account, it can't be automated here).

## One-time submission

1. **Release v3.0.0 on PyPI first** (the recipe points at the sdist):
   `git tag v3.0.0 && git push origin v3.0.0` (CD workflow publishes).
2. **Get the sdist hash:**
   ```bash
   curl -s https://pypi.org/pypi/matgraph-cli/3.0.0/json \
     | python3 -c "import json,sys; print([ (f['url'], f['digests']['sha256']) for f in json.load(sys.stdin)['urls'] if f['packagetype']=='sdist'][0])"
   ```
   Paste the URL + sha256 into `meta.yaml` (replacing the placeholder).
3. **Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)**,
   copy this recipe to `recipes/matgraph-cli/meta.yaml`, push, open a PR
   titled "Add matgraph-cli".
4. Address reviewer feedback (usually lint nits), wait for merge. The
   `matgraph-cli-feedstock` repo is then created and builds
   `conda install -c conda-forge matgraph-cli`.

## Later releases (automated)

conda-forge's bot watches PyPI and opens version-bump PRs on the feedstock
automatically — you just merge them. No further work per release.

## What conda users get

The feedstock ships the **slim base** (CLI + cache + `--demo`). Heavy stacks
stay pip extras inside the conda env:

```bash
conda install -c conda-forge matgraph-cli
matgraph predict Si --demo            # works immediately, no key
pip install "matgraph-cli[ml]"        # full ML stack in the same env
export MP_API_KEY="..."
matgraph predict LiFePO4 --model m3gnet
```
