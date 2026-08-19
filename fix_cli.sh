sed -i '' '/from matgraph.data.versioning import version_dataset/d' matgraph/cli.py
sed -i '' '/vid=version_dataset(path, meta={"cli":"version"})/d' matgraph/cli.py
sed -i '' 's/console.print(f"Versioned as \[bold\]{vid}\[\/bold\] in local store")/console.print("[dim]Hint: Version this dataset natively using `dvc add` or `huggingface-cli upload`[\/dim]")/g' matgraph/cli.py
sed -i '' '/console.print(f"\[dim\]Hint: matgraph dataset list to view\[\/dim\]")/d' matgraph/cli.py
