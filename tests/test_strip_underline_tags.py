"""strip_underline_tags — 밑줄 태그와 서식용 첨자 태그 제거 (실제 첨자는 보존)."""

from strip_underline_tags import strip_underline_tags


def test_underline_tags_removed_text_kept():
    assert strip_underline_tags("<u>**12. 안전**</u>") == "**12. 안전**"


def test_korean_script_tag_is_unwrapped():
    assert strip_underline_tags("**<sup>(수급인)</sup> 안전보건관리계획서**") == "**(수급인) 안전보건관리계획서**"
    assert strip_underline_tags("<sub>《 의무사항 제4조》</sub>") == "《 의무사항 제4조》"


def test_real_scripts_are_kept():
    assert strip_underline_tags("면적 100m<sup>2</sup>, 10<sup>4</sup>배, H<sub>2</sub>O") == \
        "면적 100m<sup>2</sup>, 10<sup>4</sup>배, H<sub>2</sub>O"


def test_mixed_line():
    assert strip_underline_tags("<sub>재해예방</sub> 100m<sup>2</sup>") == "재해예방 100m<sup>2</sup>"


def test_circled_number_marker_is_unwrapped():
    assert strip_underline_tags("<sub>①</sub>\xa0재해예방에") == "①\xa0재해예방에"
