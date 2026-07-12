import pytest

from processing.chunking import split_text


def test_empty_input_returns_empty_list():
    assert split_text("", max_words=500) == []
    assert split_text("   \n\t", max_words=500) == []


def test_splits_into_expected_number_of_chunks():
    text = " ".join(["word"] * 1000)
    chunks = split_text(text, max_words=500)

    assert len(chunks) == 2
    assert len(chunks[0].split()) == 500
    assert len(chunks[1].split()) == 500


def test_single_chunk_when_text_is_short():
    text = "Hello world this is a test"
    chunks = split_text(text, max_words=500)

    assert chunks == [text]


def test_exact_boundary():
    text = " ".join(["word"] * 500)
    chunks = split_text(text, max_words=500)

    assert len(chunks) == 1
    assert len(chunks[0].split()) == 500


@pytest.mark.parametrize(
    "max_words,expected_chunks",
    [
        (100, 3),
        (150, 2),
        (500, 1),
    ],
)
def test_respects_custom_chunk_size(max_words: int, expected_chunks: int):
    text = " ".join(["word"] * 300)
    chunks = split_text(text, max_words=max_words)

    assert len(chunks) == expected_chunks


def test_preserves_original_content_order():
    text = "one two three four five six seven eight nine ten"
    chunks = split_text(text, max_words=3)

    assert " ".join(chunks) == text