import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_attribution_scott_eliezer import clean_html


def test_nested_and_unclosed_blocks_do_not_duplicate_descendant_text():
    html = """<p>First paragraph has <span>one span</span>.
    <p>Second paragraph stays once.
    <ul><li>Parent item<p>Nested paragraph</p>
      <ul><li>Child item</li></ul></li></ul>"""

    text = clean_html(html)

    assert text.split().count("paragraph") == 3
    assert text.split().count("span") == 1
    assert text.split().count("Parent") == 1
    assert text.split().count("Nested") == 1
    assert text.split().count("Child") == 1
    assert "First paragraph has one span" in text
    assert "Second paragraph stays once." in text
