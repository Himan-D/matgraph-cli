def test_wandb_dropin(tmp_path, monkeypatch):
    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path/"tracking"))
    # drop-in: import matgraph.tracking as wandb
    import matgraph.tracking as wandb
    run = wandb.init(project="compat-test", name="dropin", config={"model":"m3gnet"})
    # wandb.Table compatible
    t = wandb.Table(columns=["f","e"], data=[["Si",-0.5]])
    assert t.columns == ["f","e"]
    run.log_table("preds", ["f","e"], [["Si",-0.5]])
    # wandb.Artifact compatible
    art = wandb.Artifact(name="my-data", type="dataset")
    art.add_file(__file__)
    assert wandb.Artifact
    # wandb.log compatible via module
    wandb.log({"loss":0.1})
    # wandb.Image
    img = wandb.Image(__file__)
    assert img.path == __file__
    run.finish()
    # verify stored
    from matgraph.tracking.store import get_run
    fetched = get_run(run.id)
    assert fetched is not None
    assert len(fetched["metrics"])>=2

def test_wandb_sync_if_installed(tmp_path, monkeypatch):
    # if wandb installed, should not crash
    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path/"tracking2"))
    import matgraph.tracking as mt
    assert hasattr(mt, "_HAS_WANDB")
