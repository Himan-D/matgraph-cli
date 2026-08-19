sed -i '' '/<<<<<<< HEAD/d' pyproject.toml
sed -i '' '/all = \["matgraph-cli\[parquet,polars,ml,dft,api,ui,tracking,diffusion,omat24\]"\]/d' pyproject.toml
sed -i '' '/=======/d' pyproject.toml
sed -i '' '/>>>>>>> e93f2c1 (feat: Add PennyLane Quantum ML and VQE integrations)/d' pyproject.toml
