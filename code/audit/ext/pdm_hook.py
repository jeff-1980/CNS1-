"""P5：pdm-bench DL 流水线原样运行，只加逐 epoch 的 test 记录（预注册 §3.2）。

pdm-bench 只在最后一轮评估 test、没有验证集，作者口径就是最后一轮。为了算“如果按 test 挑 epoch 会报多少”，
在每轮训练结束、作者写日志时，用作者自己的 evaluate 与损失函数在 test 上评估一次，放在 RNG 与模型模式隔离里，
不改源码：只替换引擎模块里的 make_model / _build_training_objects / _prepare_dataloaders / _log_epoch_line
四个名字，取到模型、损失函数和 test loader。结束时核对最后一轮的记录与作者自己的 TEST 结果是否一致（cudnn.benchmark 下可能有极小差异，只记录不中断）。

用法（hydra 参数原样透传；由导入的模块调用 main 时相对 config_path 失效，须给绝对 --config-path）：
  PDM_EPOCH_LOG=/abs/out.json PYTHONPATH=$PDM_DIR/src python -m audit.ext.pdm_hook --config-path $PDM_DIR/config task=cwru_cross_load/cnn1d ..."""
import json, os, sys
import torch
from audit.torch_utils import isolate


def install(out_path):
    from pdm_bench.training.dl import engine
    st = dict(model=None, criterion=None, test_loader=None, device=None, rows=[])

    orig_make, orig_build, orig_prep, orig_log = (engine.make_model, engine._build_training_objects,
                                                  engine._prepare_dataloaders, engine._log_epoch_line)

    def make_model(*a, **k):
        st["model"] = orig_make(*a, **k); return st["model"]

    def build(model, cfg, device):
        out = orig_build(model, cfg, device); st["criterion"], st["device"] = out[0], device; return out

    def prep(*a, **k):
        out = orig_prep(*a, **k); st["test_loader"] = out[1][2]; return out

    def log(name, epoch, epochs, *rest):
        orig_log(name, epoch, epochs, *rest)
        if st["test_loader"] is None: raise RuntimeError("pdm_hook: no test loader")
        with isolate([st["model"]], st["device"]):
            acc, loss = engine.evaluate(st["model"], st["test_loader"], st["device"], st["criterion"])
        st["rows"].append(dict(model=name, epoch=epoch, epochs=epochs, test_acc=acc, test_loss=loss))
        print(f"[audit] {name} epoch {epoch} test acc={acc:.4f} loss={loss:.4f}", flush=True)

    engine.make_model, engine._build_training_objects = make_model, build
    engine._prepare_dataloaders, engine._log_epoch_line = prep, log

    orig_train = engine.train_dl_models

    def train(*a, **k):
        res = orig_train(*a, **k)
        for name, r in res.items():
            rows = [x for x in st["rows"] if x["model"] == name]
            last = rows[-1]; t = r["metrics"]["test"]
            ok = abs(last["test_acc"] - t["acc"]) < 1e-12 and abs(last["test_loss"] - t["loss"]) < 1e-9
            json.dump(dict(model=name, author_test=t, epochs=rows, last_matches_author=ok,
                           random_state=k["cfg"].random_state if "cfg" in k else None, torch=torch.__version__),
                      open(out_path, "w"), indent=1)
            print(f"[audit] {name}: last-epoch record {'matches' if ok else 'DIFFERS FROM'} author TEST "
                  f"({last['test_acc']:.4f} vs {t['acc']:.4f})", flush=True)
        return res

    engine.train_dl_models = train
    import pdm_bench.pipelines.dl.pipeline as pipe   # pipeline 以 from-import 绑定了 train_dl_models
    pipe.train_dl_models = train
    return pipe


if __name__ == "__main__":
    out = os.environ["PDM_EPOCH_LOG"]
    pipe = install(os.path.abspath(out))
    sys.argv = ["pdm_bench.pipelines.dl.pipeline"] + sys.argv[1:]
    sys.exit(pipe.main())
