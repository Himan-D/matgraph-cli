def test_wandb_like_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path/"tracking"))
    from matgraph.tracking import init, list_runs
    from matgraph.tracking.store import get_run
    r = init(project="test-proj", name="run1", config={"model":"m3gnet","formula":"Si"})
    r.log({"eform":-0.5, "step":1})
    r.log({"eform":-0.45})
    r.log_table("preds", ["f","e"], [["Si",-0.5]])
    r.log_artifact(__file__, type="code")
    r.finish()
    runs = list_runs(project="test-proj")
    assert len(runs)>=1
    fetched = get_run(r.id)
    assert fetched["id"]==r.id
    assert len(fetched["metrics"])==3  # 2 logs + table
    assert any(a["type"]=="table" for a in fetched["artifacts"])

def test_cli_track(tmp_path, monkeypatch):
    monkeypatch.setenv("MATGRAPH_TRACKING_DIR", str(tmp_path/"tracking2"))
    from typer.testing import CliRunner
    from matgraph.cli import app
    runner = CliRunner()
    res = runner.invoke(app, ["track","init","--project","cli-proj","--name","cli-run"])
    assert res.exit_code==0
    # ls
    res2 = runner.invoke(app, ["track","ls","--project","cli-proj"])
    assert res2.exit_code==0
    assert "cli-run" in res2.output
