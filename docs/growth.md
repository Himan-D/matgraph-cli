# Growth & Distribution Runbook

How MatGraph goes from thousands to millions of installs. This is the
maintainer's checklist — product, packaging, and distribution work that
compounds. (Download counts include CI/mirror reinstalls, so track stars,
citations, and dependents alongside them.)

## Shipped in v3.0.0

- [x] **Slim base install** (~30 MB): CLI + cache + demo work from
  `pip install matgraph-cli`; heavy stacks are opt-in extras.
- [x] **Zero-config demo mode**: `matgraph predict Si --demo` needs no API
  key and no ML stack. Install-to-first-success in under 60 seconds.
- [x] **Actionable missing-extra errors**: every heavy import failure names
  the exact `pip install "matgraph-cli[<extra>]"` command.
- [x] **Hosted docs** (this site) via MkDocs + GitHub Pages for SEO.
- [x] **`CITATION.cff`** + JOSS paper draft in `paper/` (submit to
  [JOSS](https://joss.theoj.org/) for a citable artifact).
- [x] **conda-forge recipe** in `conda-recipe/` (submit after the v3.0.0
  PyPI release — steps in `conda-recipe/README.md`).
- [x] **Hugging Face Spaces demo** in `spaces/` (zero-config, runs on a
  free CPU Space — publish steps in `spaces/matgraph-demo/README.md`).
- [x] **60-second Colab notebook** in `notebooks/` (no API key needed —
  ideal for courses and workshops).

## Distribution checklist (ongoing)

- [ ] Submit the JOSS paper; add the DOI badge to the README.
- [ ] Submit the conda-forge recipe; add install instructions once live.
- [ ] Publish the Hugging Face Space; embed the link in the README.
- [ ] Publish the Docker image to Docker Hub (`matgraph serve` ready).
- [ ] Get listed in `awesome-materials-informatics` and adjacent awesome lists.
- [ ] Upstream examples: contribute a MatGraph notebook to pymatgen/MatGL
  community galleries where appropriate.
- [ ] Course adoption: one university course or workshop using the no-key
  notebook is worth hundreds of installs per semester.
- [ ] Release notes on GitHub for every version (drives watchers + SEO).
- [ ] Benchmarks page: MatGraph vs raw pymatgen+MatGL for the hero workflow.

## Hero workflow

Lead all marketing with ONE workflow done 10x better: **battery cathode
screening** (`matgraph vertical LiFePO4 --domain battery`, `matgraph
substitute`, formation-energy ranking). Breadth is documented; the wedge
is what spreads.
