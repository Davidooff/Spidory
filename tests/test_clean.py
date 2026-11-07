import numpy as np

from spidory import clean_v5

EOD = 1


def stream(*docs):
    return np.concatenate([np.array(d + [EOD], dtype=np.uint16) for d in docs])


def test_equal_windows_hash_equal_and_different_ones_differ():
    a = clean_v5.ngram_hashes(np.array([5, 6, 7, 8, 9], dtype=np.uint16), 3)
    assert len(a) == 3 and a[0] != a[1]
    b = clean_v5.ngram_hashes(np.array([9, 6, 7, 8, 2], dtype=np.uint16), 3)
    assert a[1] == b[1] and a[2] != b[2]


def test_leaking_documents_are_dropped_and_others_kept():
    held = clean_v5.held_hashes([np.array([40, 41, 42, 43, 44, 45], dtype=np.uint16)], 4)
    train = stream([2, 3, 4, 5, 6, 7], [9, 41, 42, 43, 44, 10], [11, 12, 13, 14, 15])
    bad = clean_v5.leaking_documents(train, held, 4, EOD, chunk=5)  # small chunks test the overlap
    assert bad.tolist() == [1]
    kept = clean_v5.drop_documents(train, bad, EOD)
    assert kept.tolist() == stream([2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15]).tolist()


def test_a_shorter_overlap_is_not_a_leak():
    held = clean_v5.held_hashes([np.array([40, 41, 42, 43, 44], dtype=np.uint16)], 4)
    train = stream([9, 41, 42, 43, 10])  # only 3 shared tokens in a row
    assert len(clean_v5.leaking_documents(train, held, 4, EOD)) == 0


def test_clean_rewrites_files_and_reports(tmp_path):
    for name in ("story", "qa", "code"):
        stream([30, 31, 32] * 10).tofile(tmp_path / f"{name}.val.bin")
        stream([50, 51, 52] * 10).tofile(tmp_path / f"{name}.test.bin")
        stream([1 + 7] * 30, [30, 31, 32] * 10, [60, 61] * 15).tofile(tmp_path / f"{name}.train.bin")
    report = clean_v5.clean(tmp_path, EOD)
    for name in ("story", "qa", "code"):
        assert report[name]["dropped_documents"] == 1
        kept = np.fromfile(tmp_path / f"{name}.train.bin", dtype=np.uint16)
        assert 30 not in kept.tolist() and 60 in kept.tolist()
