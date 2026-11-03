"""python -m audit.ext.test_epoch_protocols"""
import os, tempfile
from . import epoch_protocols as E


def test_protocols():
    acc = [0.5, 0.9, 0.9, 0.8, 0.95, 0.95, 0.7]
    loss = [1.0, 0.3, 0.2, 0.2, 0.25, 0.1, 0.4]
    p = E.protocols(acc, loss)
    assert p["author"]["epoch"] == 5 and p["last"]["epoch"] == 7 and p["minvalloss"]["epoch"] == 6 and p["budget50"]["epoch"] == 4


def test_parse_log():
    lines = ["10-15 12:00:00 -----Epoch 0/1-----",
             "10-15 12:00:01 Epoch: 0 train-Loss: 1.2000 train-Acc: 0.5000, Cost 1.0000 sec",
             "10-15 12:00:02 Epoch: 0 val-Loss: 0.9000 val-Acc: 0.6000, Cost 1.2000 sec",
             "10-15 12:00:03 Epoch: 1 source_val-Loss: 0.5 source_val-Acc: 0.9, Cost 1.0 sec",
             "10-15 12:00:04 Epoch: 1 val-Loss: 0.4000 val-Acc: 0.8000, Cost 1.2000 sec",
             "10-15 12:00:05 Epoch: 0 target_val-Loss: 0.7 target_val-Acc: 0.7, Cost 1.0 sec"]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f: f.write("\n".join(lines))
    assert E.parse_log(f.name, "val") == ([0.6, 0.8], [0.9, 0.4])
    assert E.parse_log(f.name, "target_val") == ([0.7], [0.7])
    os.unlink(f.name)


if __name__ == "__main__":
    test_protocols(); test_parse_log(); print("ok")
