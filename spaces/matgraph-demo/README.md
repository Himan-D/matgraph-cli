---
title: MatGraph Materials ML Demo
emoji: "\U0001F52C"
colorFrom: teal
colorTo: amber
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Predict properties, simulate XRD, triage compounds — no API key
---

# MatGraph Demo (Hugging Face Spaces)

Zero-config demo of [matgraph-cli](https://pypi.org/project/matgraph-cli/):
property prediction samples, **real** Cu-Ka XRD patterns, and substitution
triage — all from bundled demo data, no API key, free CPU hardware.

For live predictions:

```bash
pip install "matgraph-cli[ml]"
export MP_API_KEY="your_key_here"   # free at materialsproject.org
matgraph predict LiFePO4 --model m3gnet
```

Docs: <https://himan-d.github.io/matgraph-cli/> ·
Code: <https://github.com/Himan-D/matgraph-cli> ·
If this saved you time, please star the repo!

## Publishing (maintainer)

```bash
pip install huggingface_hub
huggingface-cli login
cd spaces/matgraph-demo
huggingface-cli repo create matgraph-demo --type space --space_sdk gradio
huggingface-cli upload ./ . --repo-type space   # uploads app.py + requirements + README
```

Then add the Space URL to the main README badges.
