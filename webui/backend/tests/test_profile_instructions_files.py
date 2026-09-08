"""File-backed personalization: the inputs write real files + one-time
migration of the legacy locations (settings field / sidecar / project field)."""
import json
import os
from pathlib import Path

import pytest

from app.backends.ms_agent import instructions, profile, sidecar
from app.backends.ms_agent.bootstrap import bootstrap
from app.schemas.instruction import InstructionUpsert
from app.schemas.profile import ProfileUpsert


@pytest.fixture()
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MS_AGENT_HOME", str(home))
    from ms_agent.prompting import workspace_files as wf

    wf.reset_cache()
    yield home
    wf.reset_cache()


def test_bootstrap_materializes_empty_prompt_templates(fresh_home):
    from ms_agent.prompting import workspace_files as wf

    bootstrap()

    for name in ("SOUL.md", "AGENTS.md", "PROFILE.md"):
        assert (fresh_home / name).is_file()
    assert wf.strip_for_injection(
        (fresh_home / "AGENTS.md").read_text(encoding="utf-8")) == ""
    assert wf.strip_for_injection(
        (fresh_home / "PROFILE.md").read_text(encoding="utf-8")) == ""


def test_bootstrap_preserves_existing_prompt_files(fresh_home):
    agents = "Always answer in Chinese.\n"
    profile_text = "---\nversion: 1\n---\n\n# About Me\n- Call me: Alice\n"
    (fresh_home / "AGENTS.md").write_text(agents, encoding="utf-8")
    (fresh_home / "PROFILE.md").write_text(profile_text, encoding="utf-8")

    bootstrap()

    assert (fresh_home / "AGENTS.md").read_text(encoding="utf-8") == agents
    assert (fresh_home / "PROFILE.md").read_text(
        encoding="utf-8") == profile_text


def test_bootstrap_respects_deleted_builtin_prompt_file(fresh_home):
    from ms_agent.prompting import workspace_files as wf

    bootstrap()
    (fresh_home / "AGENTS.md").unlink()
    wf.reset_cache()  # simulate the next service process

    bootstrap()

    assert not (fresh_home / "AGENTS.md").exists()


def test_global_instruction_writes_agents_md(fresh_home):
    instructions.upsert_instruction(
        "global", InstructionUpsert(content="Always answer in French."))
    text = (fresh_home / "AGENTS.md").read_text(encoding="utf-8")
    assert "Always answer in French." in text
    assert text.lstrip().startswith("---")  # seeded header preserved
    got = instructions.get_instruction("global")
    assert got.content == "Always answer in French."
    # and the SDK actually injects it
    from ms_agent.prompting import workspace_files as wf

    assert "Always answer in French." in wf.global_instructions_block()


def test_global_migration_moves_settings_field_once(fresh_home):
    (fresh_home / "settings.json").write_text(json.dumps(
        {"personalization": {"global_instruction": "你的名字是小黑"}},
        ensure_ascii=False), encoding="utf-8")
    got = instructions.get_instruction("global")
    assert got.content == "你的名字是小黑"
    # moved into the file...
    assert "你的名字是小黑" in (fresh_home / "AGENTS.md").read_text("utf-8")
    # ...and cleared from settings.json
    data = json.loads((fresh_home / "settings.json").read_text("utf-8"))
    assert data["personalization"]["global_instruction"] == ""


def test_profile_fields_write_profile_md(fresh_home):
    profile.update_profile(
        ProfileUpsert(agent_calls_user="Alice", description="主要从事 Agent 相关工作"))
    text = (fresh_home / "PROFILE.md").read_text("utf-8")
    assert "- Call me: Alice" in text
    assert "主要从事 Agent 相关工作" in text

    got = profile.get_profile()
    assert got.agent_calls_user == "Alice"
    assert got.description == "主要从事 Agent 相关工作"

    # clearing the name removes the line, description stays
    profile.update_profile(ProfileUpsert(agent_calls_user=""))
    got = profile.get_profile()
    assert got.agent_calls_user == ""
    assert got.description == "主要从事 Agent 相关工作"


def test_profile_sidecar_migration_and_retirement(fresh_home):
    sidecar.put("profile", "agent_calls_user", "老韩")
    got = profile.get_profile()
    assert got.agent_calls_user == "老韩"
    assert "- Call me: 老韩" in (fresh_home / "PROFILE.md").read_text("utf-8")
    # the sidecar key is gone even if it held the boilerplate default
    assert sidecar.get("profile", "agent_calls_user", None) is None


def test_profile_sidecar_default_user_not_migrated(fresh_home):
    sidecar.put("profile", "agent_calls_user", "User")
    got = profile.get_profile()
    assert got.agent_calls_user == ""  # boilerplate dropped, not migrated
    assert sidecar.get("profile", "agent_calls_user", None) is None


def test_project_instruction_writes_private_slot_only(fresh_home, tmp_path):
    """The UI writes <project>/.ms_agent/AGENTS.md and must NEVER touch the
    repo-root AGENTS.md — AI-native repos commit that file for their own
    coding agents."""
    from app.backends.ms_agent.common import pm

    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    root_file = proj_dir / "AGENTS.md"
    root_file.write_text("Team rules, committed to git.\n", encoding="utf-8")
    project = pm().open_folder(str(proj_dir))
    scope = f"project:{project.id}"

    # the box shows the UI-owned slot (empty), not the team file
    assert instructions.get_instruction(scope).content == ""

    instructions.upsert_instruction(
        scope, InstructionUpsert(content="This project uses uv."))
    private = proj_dir / ".ms_agent" / "AGENTS.md"
    assert private.read_text("utf-8").strip() == "This project uses uv."
    assert instructions.get_instruction(scope).content == "This project uses uv."
    # the team file is byte-identical
    assert root_file.read_text("utf-8") == "Team rules, committed to git.\n"

    # and the SDK injects BOTH, root (shared) before private
    from ms_agent.prompting import workspace_files as wf

    block = wf.project_instructions_block(str(proj_dir))
    assert "Team rules" in block and "This project uses uv." in block
    assert block.index("Team rules") < block.index("This project uses uv.")


def test_project_legacy_field_migrates_to_private_slot(fresh_home, tmp_path):
    from app.backends.ms_agent.common import pm

    proj_dir = tmp_path / "proj2"
    proj_dir.mkdir()
    project = pm().open_folder(str(proj_dir))
    pm().update(project.id, instruction="老项目规矩")
    scope = f"project:{project.id}"

    got = instructions.get_instruction(scope)
    assert got.content == "老项目规矩"
    assert (proj_dir / ".ms_agent" / "AGENTS.md").read_text("utf-8").strip() \
        == "老项目规矩"
    assert not (proj_dir / "AGENTS.md").exists()  # root stays user-owned
    assert (pm().get(project.id).instruction or "") == ""
