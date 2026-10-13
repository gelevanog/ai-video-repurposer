from __future__ import annotations

import json

from recut.config import Settings
from recut.content import limits
from recut.content.grounding import TranscriptIndex, check, check_numbers, check_quotes
from recut.content.posts import assemble_description, generate, plain_transcript
from recut.models import Chapter, SourceInfo
from recut.text import sentences_from_words
from tests.conftest import LONG_TALK, make_words
from tests.stubs import ScriptedModel

INDEX = TranscriptIndex(LONG_TALK)


def test_x_counts_urls_as_23_and_emoji_as_2() -> None:
    assert limits.count("hello", "x") == 5
    assert limits.count("see https://example.com/a/very/long/path/indeed ok", "x") == len("see ") + 23 + len(" ok")
    assert limits.count("🎬🎬", "x") == 4
    assert limits.count("🎬🎬") == 2


def test_truncate_cuts_at_a_word_boundary_with_an_ellipsis() -> None:
    text = "one two three four five six seven"
    cut = limits.truncate(text, 15)
    assert cut.endswith("…") and len(cut) <= 15 and "thr" not in cut.split()[-1][:-1] or cut == "one two three…"
    assert limits.truncate("short", 100) == "short"


def test_tags_total_follows_youtube_counting() -> None:
    assert limits.tags_total(["a", "b c"]) == 1 + 3 + 2 + 1


def test_platform_limits_are_the_published_ones() -> None:
    assert limits.PLATFORMS["x"].limit == 280
    assert limits.PLATFORMS["linkedin"].limit == 3000
    assert limits.PLATFORMS["instagram"].limit == 2200
    assert limits.PLATFORMS["threads"].limit == 500
    assert all(p.target <= p.limit for p in limits.PLATFORMS.values())


def test_numbers_from_the_transcript_pass_and_invented_ones_are_flagged() -> None:
    assert check_numbers("I charged $200 and the client made $40,000.", INDEX) == []
    issues = check_numbers("Creators who do this earn 8x more, a 75% jump.", INDEX)
    assert {i.text for i in issues} == {"8", "75%"}


def test_list_counts_and_timestamps_are_not_claims() -> None:
    assert check_numbers("5 takeaways from this episode:\n00:00 Intro\n12:30 Pricing", INDEX) == []


def test_quotes_must_be_said() -> None:
    assert check_quotes('He said "price the result, not the hours" and it stuck.', INDEX) == []
    issues = check_quotes('“Never work for exposure, it is the worst deal in business”', INDEX)
    assert len(issues) == 1 and issues[0].kind == "quote"


def test_check_combines_numbers_and_quotes() -> None:
    assert len(check('"I made 9 million dollars in a single day"', INDEX)) == 2


def test_description_gets_chapters_and_the_credit() -> None:
    chapters = [Chapter(start=0, title="Intro"), Chapter(start=60, title="Pricing"), Chapter(start=120, title="Tests")]
    source = SourceInfo(path="x", filename="x.mp4", license="CC BY 4.0", attribution="Video by Jane Doe, CC BY 4.0")
    text = assemble_description("A talk about pricing.", chapters, source)
    assert "Chapters:\n00:00 Intro\n01:00 Pricing" in text
    assert text.endswith("Video by Jane Doe, CC BY 4.0")
    no_credit = assemble_description("A talk.", chapters[:2], SourceInfo(path="x", filename="x.mp4", license="Public domain (NASA)"))
    assert "Chapters" not in no_credit and "Public domain" not in no_credit


def test_plain_transcript_thins_long_inputs() -> None:
    sentences = sentences_from_words(make_words(LONG_TALK * 10))
    text = plain_transcript(sentences, max_words=300)
    assert "long transcript" in text
    assert len(text.split()) < 900


def _source() -> SourceInfo:
    return SourceInfo(path="x", filename="talk.mp4", title="Pricing talk", license="CC BY 4.0", attribution="Credit: Jane Doe (CC BY 4.0)")


def test_generate_enforces_limits_retries_and_grounding(settings: Settings) -> None:
    sentences = sentences_from_words(make_words(LONG_TALK))
    long_x = "Price the result. " * 30
    model = ScriptedModel(
        {
            "metadata": json.dumps(
                {
                    "title": "Why your price is too low",
                    "description": "Most creators undercharge. Here is a fix that doubled revenue by 300%.",
                    "tags": ["pricing", "#Freelance", "pricing", "creator economy"],
                    "summary": "A short talk on pricing.",
                    "takeaways": ["Price the result.", "Quote the stretch number first."],
                    "quotes": [4, 999],
                }
            ),
            "blog_post": "```markdown\n# Pricing\n\nSome text.\n```",
            "social_posts": [
                json.dumps({"x": long_x, "linkedin": "New talk on pricing. #pricing", "instagram": "", "tiktok": "Price it right", "threads": "ok", "facebook": "fb"}),
                json.dumps({"x": long_x}),
            ],
        }
    )
    outputs, stats = generate(sentences, [], _source(), model, settings)
    assert stats["calls"] == 4  # metadata, blog, social, one retry for the over-limit X post
    x = next(p for p in outputs.social if p.platform == "x")
    assert x.truncated and x.within_limit and x.chars <= 280
    assert any(i.kind == "length" for i in x.issues)
    assert "instagram" not in {p.platform for p in outputs.social}  # empty posts are dropped
    assert outputs.tags == ["pricing", "freelance", "creator economy"]
    assert "“So the lesson is simple: price the result, not the hours.”" in outputs.show_notes
    assert outputs.blog_post.startswith("# Pricing") and "Credit: Jane Doe" in outputs.blog_post
    assert any("300%" in i.text for i in outputs.issues)
    assert outputs.seo_description.endswith("Credit: Jane Doe (CC BY 4.0)")


def test_generate_survives_model_failures(settings: Settings) -> None:
    sentences = sentences_from_words(make_words(LONG_TALK))
    outputs, stats = generate(sentences, [], _source(), ScriptedModel(fail=True), settings)
    assert len(stats["errors"]) == 3
    assert outputs.seo_title == "Pricing talk" and outputs.social == []
