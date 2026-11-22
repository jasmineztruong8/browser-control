import pytest

from browser_agent.actions import InvalidActionError, parse_action
from browser_agent.types import Action, ActionKind


@pytest.mark.parametrize(
    "output, expected",
    [
        ("click [12]", Action(kind=ActionKind.CLICK, element_id=12)),
        ("  CLICK[3].  ", Action(kind=ActionKind.CLICK, element_id=3)),
        ('type [5] "blue mug"', Action(kind=ActionKind.TYPE, element_id=5, text="blue mug")),
        (
            'type [5] "mug" enter',
            Action(kind=ActionKind.TYPE, element_id=5, text="mug", submit=True),
        ),
        (
            'type [1] "a \\"quoted\\" word"',
            Action(kind=ActionKind.TYPE, element_id=1, text='a "quoted" word'),
        ),
        ("scroll down", Action(kind=ActionKind.SCROLL, direction="down")),
        ("stop", Action(kind=ActionKind.STOP)),
        ('stop "added to cart"', Action(kind=ActionKind.STOP, answer="added to cart")),
        ("```\nclick [7]\n```", Action(kind=ActionKind.CLICK, element_id=7)),
        ("I will search first.\nclick [2]", Action(kind=ActionKind.CLICK, element_id=2)),
        (
            "Plan: open the product page, then pick the color.\nAction: click [4]",
            Action(kind=ActionKind.CLICK, element_id=4),
        ),
    ],
)
def test_parse_valid(output, expected):
    assert parse_action(output) == expected


@pytest.mark.parametrize(
    "output",
    [
        "",
        "   ",
        "click the add button",
        "click [abc]",
        "type [3] mug",
        "scroll sideways",
        "press enter",
    ],
)
def test_parse_invalid(output):
    with pytest.raises(InvalidActionError):
        parse_action(output)


def test_disallowed_action():
    with pytest.raises(InvalidActionError, match="not allowed"):
        parse_action("scroll down", allowed=["click", "type", "stop"])


def test_round_trip_through_text():
    for action in [
        Action(kind=ActionKind.CLICK, element_id=0),
        Action(kind=ActionKind.TYPE, element_id=9, text='x "y" z', submit=True),
        Action(kind=ActionKind.SCROLL, direction="up"),
        Action(kind=ActionKind.STOP, answer="ok"),
    ]:
        assert parse_action(action.to_text()) == action
