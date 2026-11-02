# 已知真值检验：运行记录

- 2026-11-02 14:01 至 18:22 UTC，`~/audit_ext/known_truth/run_all_kt.sh`，`nohup setsid` 后台依次运行，主进程 PID 61873。
- 6 个对象、16 组、160 次运行全部完成（每组 10 个种子或重复），日志无 Traceback；62 Teacher 的 F1 核对差 4e-5 量级，与主分析相同（批量推理与逐样本推理之差）。
- 全测试集曲线与主分析不逐位相同（0/160）：54、76、37 作者不设种子；58、53 为 GPU 非确定性；62 的加噪数据重新生成。两次运行的最后一轮准确率均值相差 ≤ 1.3 pp，best−last 均值相差 ≤ 0.8 pp（见 `known_truth.json` 与报告），可比。
- 汇总：`code/audit/main/known_truth_curves.py` → `curves/`；`code/audit/main/known_truth.py` → `known_truth.json`。
