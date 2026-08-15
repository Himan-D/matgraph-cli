#!/usr/bin/env python3
"""Marketing copy for release — prints posts for X/HN/Reddit."""
ver="2.12.1"
msg=f"""MatGraph v{ver} — materials ML for researchers & AI agents

pip install matgraph-cli

6 FMMs (M3GNet/MEGNet/CGCNN/CHGNet/OMat24) + 7 verticals (battery/PV/catalysis/thermo/2D/alloy/defect) — all ML/DL, zero hardcodes via MATGRAPH_* env, pymatgen/BoltzTraP2 wired, provenance+cache, GraphQL+REST+MCP.

For agents: llms.txt + .well-known/ai-plugin.json + docs/openapi.json + mcp.json
Why: 7k PyPI downloads, now agent-discoverable.

GitHub: github.com/Himan-D/matgraph-cli  ⭐ star helps discovery!
PyPI: pypi.org/project/matgraph-cli
Colab: colab badge in README

#materialsInformatics #machineLearning #pymatgen #matgl
"""
print(msg)
# also write to marketing/ folder
import pathlib
pathlib.Path("marketing").mkdir(exist_ok=True)
(pathlib.Path("marketing")/f"v{ver}_posts.md").write_text(msg)
print("Wrote marketing/v"+ver+"_posts.md")
