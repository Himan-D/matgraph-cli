# Upstream examples & community distribution (ready to use)

The goal: MatGraph shows up where materials scientists already learn —
inside upstream galleries, course material, and community threads.

## 1. Example notebook for upstream galleries

`notebooks/MatGraph_60sec_Demo.ipynb` is deliberately dependency-free (no API
key, base install only), which is exactly what gallery curators want. Propose
it to:

- **MatGL Discussions / examples** — "end-to-end screening with MatGL
  potentials via a one-line SDK", linking the notebook + comparison doc
  (`docs/comparison.md`).
- **pymatgen community examples** — the XRD + battery-capacity cells use real
  crystallography and stand alone without any key.

Outreach post template (Discussions, Discourse, or mailing list):

```markdown
Title: MatGraph — one-line screening over pymatgen/MatGL (+ no-key demo notebook)

Hi all — I maintain MatGraph (MIT), a workflow layer over pymatgen/MatGL/mp-api:
`matgraph predict LiFePO4`, `matgraph battery LiFePO4`, XRD, substitution
triage, dataset export, and a served GraphQL API. New in v3: a slim ~30 MB
base install and a `--demo` mode that runs the whole interface with no API key.

No-key notebook (great for teaching): <link to MatGraph_60sec_Demo.ipynb on Colab>
Docs: https://himan-d.github.io/matgraph-cli/
Comparison with the raw stack: <link to comparison page>

Feedback welcome — especially which vertical (battery/PV/catalysis/...) you'd
like deepened next.
```

## 2. Course / workshop packet

One adopter teaching with MatGraph is worth hundreds of installs per semester.
The packet is ready:

- `notebooks/MatGraph_60sec_Demo.ipynb` — runs in Colab, zero setup.
- `docs/tutorials.md` Tutorial 0 → Tutorial 1 path (demo, then live screening).
- HF Space (once published) for in-lecture demos without installs.

Pitch: email the notebook + one paragraph ("your students can screen battery
cathodes in the first 10 minutes of lab, no keys, no installs beyond pip")
to instructors of computational materials / materials informatics courses.

## 3. Release-driven distribution (every version)

- GitHub Release notes with the headline workflow (watchers + SEO).
- Short demo video/GIF of `matgraph battery LiFePO4 --demo` for social posts.
- Update the JOSS paper's software link once the DOI lands; add the DOI badge
  to the README.
