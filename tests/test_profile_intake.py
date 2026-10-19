from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.db import engine
from app.schemas.profile import ChatMessage, Criteria, ProfileSnapshotData, Skills
from app.services import profile_intake


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def make_profile(**overrides):
    defaults = dict(
        experience=[],
        projects=[],
        skills=Skills(),
        criteria=Criteria(),
    )
    defaults.update(overrides)
    return ProfileSnapshotData(**defaults)


EMPTY_SNAPSHOT_TOOL_INPUT = {
    "experience": [],
    "projects": [],
    "skills": {"core": [], "developing": [], "coursework": []},
    "credentials": {"education": [], "oss_contributions": [], "publications": []},
    "criteria": {
        "skip_patterns": [],
        "target_signals": [],
        "location_prefs": {
            "cities": [],
            "countries": [],
            "exclude_cities": [],
            "exclude_countries": [],
            "remote_ok": True,
            "hybrid_ok": True,
            "onsite_ok": True,
        },
        "seniority_floor": None,
        "seniority_ceiling": None,
    },
}


def test_chat_raises_stage2_unavailable_without_api_key():
    with patch.object(profile_intake.settings, "anthropic_api_key", None):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            with pytest.raises(profile_intake.Stage2UnavailableError):
                profile_intake.chat(make_profile(), [], "I worked at Acme as an engineer.")
            mock_client_cls.assert_not_called()


def test_chat_returns_assistant_message_for_a_plain_text_turn():
    fake_text_block = MagicMock(type="text", text="Was that role full-time or a contract?")
    fake_response = MagicMock(content=[fake_text_block])
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            result = profile_intake.chat(make_profile(), [], "I worked at Acme as an engineer.")

    assert result["assistant_message"] == "Was that role full-time or a contract?"
    assert result["proposed_snapshot"] is None
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "auto"}
    assert call_kwargs["model"] == profile_intake.PROFILE_INTAKE_MODEL


def test_chat_returns_proposed_snapshot_for_a_tool_use_turn():
    tool_input = {
        "snapshot": {
            **EMPTY_SNAPSHOT_TOOL_INPUT,
            "skills": {"core": ["Acme engineering"], "developing": [], "coursework": []},
        },
        "changelog": "Added Acme engineering role to core skills.",
        "open_questions": ["Confirm employment dates."],
    }
    fake_tool_use_block = MagicMock(type="tool_use", input=tool_input)
    fake_response = MagicMock(content=[fake_tool_use_block])
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            result = profile_intake.chat(make_profile(), [], "I worked at Acme as an engineer.")

    assert result["assistant_message"] is None
    assert result["proposed_snapshot"]["skills"]["core"] == ["Acme engineering"]
    assert result["changelog"] == "Added Acme engineering role to core skills."
    assert result["open_questions"] == ["Confirm employment dates."]


def test_chat_includes_current_snapshot_and_history_in_the_prompt():
    fake_text_block = MagicMock(type="text", text="Got it.")
    fake_response = MagicMock(content=[fake_text_block])
    profile = make_profile(skills=Skills(core=["DistinctiveExistingSkill"]))
    history = [
        ChatMessage(role="user", content="Hi"),
        ChatMessage(role="assistant", content="Hello!"),
    ]
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            profile_intake.chat(profile, history, "Here is more about my background.")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "Hi"}
    assert messages[1] == {"role": "assistant", "content": "Hello!"}
    assert "DistinctiveExistingSkill" in messages[-1]["content"]
    assert "Here is more about my background." in messages[-1]["content"]


def test_append_and_list_messages_orders_oldest_first(db):
    # Checks the last two entries, not the whole list -- this table holds
    # one real, ever-growing conversation (not isolated test fixtures), so
    # real prior messages may already exist ahead of whatever this test adds.
    profile_intake.append_message(db, "user", "first")
    profile_intake.append_message(db, "assistant", "second")

    messages = profile_intake.list_messages(db)
    assert [(m.role, m.content) for m in messages[-2:]] == [
        ("user", "first"),
        ("assistant", "second"),
    ]


def test_clear_messages_empties_the_log(db):
    profile_intake.append_message(db, "user", "first")
    profile_intake.clear_messages(db)

    assert profile_intake.list_messages(db) == []


def test_run_chat_turn_persists_both_sides_of_a_plain_text_turn(db):
    fake_text_block = MagicMock(type="text", text="Was that role full-time or a contract?")
    fake_response = MagicMock(content=[fake_text_block])
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            result = profile_intake.run_chat_turn(
                db, make_profile(), "I worked at Acme as an engineer."
            )

    assert result["assistant_message"] == "Was that role full-time or a contract?"
    messages = profile_intake.list_messages(db)
    assert [(m.role, m.content) for m in messages[-2:]] == [
        ("user", "I worked at Acme as an engineer."),
        ("assistant", "Was that role full-time or a contract?"),
    ]


def test_run_chat_turn_persists_a_proposal_turn_as_a_summary(db):
    tool_input = {
        "snapshot": EMPTY_SNAPSHOT_TOOL_INPUT,
        "changelog": "Added Acme engineering role to core skills.",
        "open_questions": [],
    }
    fake_tool_use_block = MagicMock(type="tool_use", input=tool_input)
    fake_response = MagicMock(content=[fake_tool_use_block])
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            profile_intake.run_chat_turn(db, make_profile(), "I worked at Acme as an engineer.")

    messages = profile_intake.list_messages(db)
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "Proposed update: Added Acme engineering role to core skills."


def test_run_chat_turn_passes_persisted_history_to_chat(db):
    profile_intake.append_message(db, "user", "Earlier turn")
    profile_intake.append_message(db, "assistant", "Earlier reply")

    fake_text_block = MagicMock(type="text", text="Understood.")
    fake_response = MagicMock(content=[fake_text_block])
    with patch.object(profile_intake.settings, "anthropic_api_key", "fake-key"):
        with patch("app.services.profile_intake.Anthropic") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_client_cls.return_value = mock_client

            profile_intake.run_chat_turn(db, make_profile(), "New turn")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    # The two just-appended messages are the last ones in persisted history,
    # immediately before the new turn's own combined message (always last).
    assert messages[-3] == {"role": "user", "content": "Earlier turn"}
    assert messages[-2] == {"role": "assistant", "content": "Earlier reply"}
