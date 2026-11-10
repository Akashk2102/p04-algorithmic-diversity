from src.data import load_jsonl, normalize_record
from src.retrieval import Retriever

def test_retrieval_methods():
    rows=[normalize_record(x,i) for i,x in enumerate(load_jsonl('data/sample/demo.jsonl'))]
    r=Retriever(rows,k=3)
    out=r.retrieve_all(0)
    assert set(out)=={'random','semantic','mmr','algorithm_aware'}
    for v in out.values(): assert len(v)==3
