"""正式阶段 §3.2：各对象补丁共用的逐 epoch 记录器。对象补丁只调用这里，输出格式全对象统一。

用法（在作者训练循环里，每轮结束、作者自己的评估之后）：
    from audit.main.epoch_record import Recorder
    rec = Recorder(obj="01", group="DSAGCN|CWRU 0->3", seed=0, select_split="val", report_split="test")
    rec.log(epoch, "test", acc, loss)        # 报告集（论文报告的划分），每轮必记
    rec.log(epoch, "val", acc, loss)         # 选模集（作者用来挑 epoch 的划分）；没有就不记
    rec.author_pick(epoch)                   # 作者代码实际选中（保存 / 报告）的 epoch，可多次调用，取最后一次
    rec.finish(author_reported_acc)          # 作者代码自己打印 / 返回的最终数，用于核对
补丁里另行评估的那一次必须包在 audit.torch_utils.isolate([model], device) 里。
acc 用 0–1 的比例；作者代码用百分数时传 acc / 100。epoch 从 1 起计。
输出：$AUDIT_RECORD_DIR/{obj}/{group 去掉非法字符}/seed{seed}.json；AUDIT_RECORD_DIR 未设时用 ./audit_records。"""
import datetime as dt, json, math, os, re


class Recorder:
    def __init__(self, obj, group, seed, select_split=None, report_split="test", note=""):
        self.d = dict(obj=str(obj), group=group, seed=seed, select_split=select_split, report_split=report_split,
                      note=note, started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      epochs={}, author_epoch=None, author_reported_acc=None)
        root = os.environ.get("AUDIT_RECORD_DIR", "audit_records")
        self.path = os.path.join(root, str(obj), re.sub(r"[^\w.+-]+", "_", group), f"seed{seed}.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, epoch, split, acc, loss=None):
        acc = float(acc); loss = None if loss is None else float(loss)
        if not 0.0 <= acc <= 1.0: raise ValueError(f"acc {acc} not in [0,1]; pass a proportion")
        self.d["epochs"].setdefault(str(int(epoch)), {})[split] = dict(acc=acc, loss=loss)
        self._dump()

    def author_pick(self, epoch):
        self.d["author_epoch"] = int(epoch); self._dump()

    def finish(self, author_reported_acc=None):
        self.d["author_reported_acc"] = None if author_reported_acc is None else float(author_reported_acc)
        self.d["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        ep = self.d["author_epoch"] or (max(map(int, self.d["epochs"])) if self.d["epochs"] else None)
        rec = self.d["epochs"].get(str(ep), {}).get(self.d["report_split"]) if ep else None
        a = self.d["author_reported_acc"]
        self.d["author_check"] = None if (rec is None or a is None) else dict(
            recorded=rec["acc"], reported=a, abs_diff=abs(rec["acc"] - a), matches=math.isclose(rec["acc"], a, abs_tol=1e-6))
        self._dump()

    def _dump(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f: json.dump(self.d, f, indent=1)
        os.replace(tmp, self.path)
