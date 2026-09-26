import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_fiction_historical import own_text
from bs4 import BeautifulSoup


def test_nested_list_and_paragraph_text_is_not_repeated():
    soup = BeautifulSoup("<li>parent <span>inline</span><p>child paragraph</p><ul><li>grandchild</li></ul></li>", "lxml")
    outer = soup.find("li")
    text = own_text(outer)

    assert text == "parent inline"
    assert own_text(outer.find("p")) == "child paragraph"
    assert own_text(outer.find("li")) == "grandchild"
