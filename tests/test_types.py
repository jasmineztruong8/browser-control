from browser_agent.types import (
    Action,
    ActionKind,
    Element,
    Observation,
    StepRecord,
    Trajectory,
    load_jsonl,
    save_jsonl,
)


def _trajectory() -> Trajectory:
    obs = Observation(
        url="file:///shop.html#home",
        elements=[Element(id=0, role="searchbox", label="Search")],
        errors=["Please choose a color."],
    )
    step = StepRecord(
        index=0,
        observation=obs,
        messages=[{"role": "user", "content": "hi"}],
        model_output='type [0] "mug" enter',
        action=Action(kind=ActionKind.TYPE, element_id=0, text="mug", submit=True),
        subgoals_hit=["searched"],
        latency_s=0.5,
    )
    return Trajectory(
        task_id="t1", steps=[step], success=True, reward=1.0, final_state={"cart": []}
    )


def test_jsonl_round_trip(tmp_path):
    trajs = [_trajectory(), _trajectory().model_copy(update={"task_id": "t2"})]
    path = tmp_path / "runs" / "traj.jsonl"
    save_jsonl(trajs, path)
    save_jsonl([_trajectory()], path, append=True)
    loaded = load_jsonl(path)
    assert loaded[:2] == trajs
    assert len(loaded) == 3


def test_action_to_text():
    assert Action(kind="click", element_id=3).to_text() == "click [3]"
    assert Action(kind="type", element_id=1, text='say "hi"').to_text() == 'type [1] "say \\"hi\\""'
    assert (
        Action(kind="type", element_id=1, text="x", submit=True).to_text() == 'type [1] "x" enter'
    )
    assert Action(kind="scroll", direction="up").to_text() == "scroll up"
    assert Action(kind="stop").to_text() == "stop"
    assert Action(kind="stop", answer="done").to_text() == 'stop "done"'


def test_num_invalid():
    traj = _trajectory()
    traj.steps.append(traj.steps[0].model_copy(update={"index": 1, "error": "invalid_action: x"}))
    assert traj.num_invalid == 1
