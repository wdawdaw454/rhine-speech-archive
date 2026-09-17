from src.data.normalize import normalize_chinese_text


def test_lowercase_and_whitespace():
    assert normalize_chinese_text("  Hello  World  ", lowercase=True, remove_whitespace=True) == "helloworld"


def test_remove_punctuation():
    out = normalize_chinese_text("你好，世界！", remove_punctuation=True)
    assert "，" not in out
    assert "！" not in out
    assert "你好世界" in out


def test_traditional_to_simplified_if_available():
    out = normalize_chinese_text("繁體字", to_simplified_chinese=True)
    assert isinstance(out, str)


def test_fullwidth_to_halfwidth():
    out = normalize_chinese_text("Ｈｅｌｌｏ１２３", normalize_width=True)
    assert out == "hello123"


def test_remove_punctuation_disabled():
    out = normalize_chinese_text("你好，世界", remove_punctuation=False)
    assert "，" in out


def test_chain_all():
    out = normalize_chinese_text(
        "  Hello，世界！  ",
        lowercase=True,
        remove_punctuation=True,
        remove_whitespace=True,
        normalize_width=True,
    )
    assert out == "hello世界"