"""后台运行 SHAP 分析，输出写入日志文件"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log_path = os.path.join(os.path.dirname(__file__), "outputs", "shap_analysis_log.txt")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

with open(log_path, 'w', encoding='utf-8') as log:
    tee = Tee(sys.stdout, log)
    sys.stdout = tee

    print(f"SHAP 分析开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志文件: {log_path}")
    print()

    t0 = time.time()
    from src.shap_analysis import main
    main()
    t1 = time.time()

    print(f"\n总耗时: {(t1-t0)/60:.1f} 分钟")
    print(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
