from aegis.training.train import load_config


def test_config_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_ARTIFACTS", "/tmp/aegis_art")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "name: x\nmodel:\n  init: ${AEGIS_ARTIFACTS}/teacher/best\n  num_layers: 3\n"
        "distill:\n  teacher_logits: ${AEGIS_ARTIFACTS}/t.npz\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg["model"]["init"] == "/tmp/aegis_art/teacher/best"
    assert cfg["distill"]["teacher_logits"] == "/tmp/aegis_art/t.npz"
    assert cfg["model"]["num_layers"] == 3


def test_shipped_configs_have_no_user_specific_paths():
    from pathlib import Path

    for f in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"):
        assert "C:/Users" not in f.read_text(), f
