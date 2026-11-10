from src.features import algorithm_tags, jaccard

def test_tags():
    tags = algorithm_tags("find shortest path", "from collections import deque\nq=deque()\nwhile q:\n pass")
    assert "queue_bfs" in tags

def test_jaccard():
    assert jaccard({"a","b"},{"b","c"}) == 1/3
