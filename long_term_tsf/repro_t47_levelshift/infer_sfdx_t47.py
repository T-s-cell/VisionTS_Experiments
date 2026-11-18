"""EXP-047 推理:StatsForecast AutoETS / AutoARIMA(t46diag env)。

- AutoETS 全 7168 条件;AutoARIMA 仅 ETTh1/ETTh2 每 var 基准清单 rank≤1 前 2 窗
  ×16 条件 = 448(辅助臂,D/代价用 ARIMA 自身 A/B 配对);
- worker 懒加载 baseline_<ds>.npz(f64 基准),按任务 (delta,m_steps,group) 在线
  构造输入,season_length=P,其余默认;确定性统计拟合,无随机数控制需求;
- TimeoutPool(仿 t46):spawn 上下文、到期 terminate+kill 真杀、闲时死亡重生;
  失败分类 timeout/nonfinite/error:/worker_died 全入 ledger,单样本失败不中止;
- 输出 pred_autoets_t47.npz / pred_autoarima_t47.npz(pred (n,96) f32,失败 NaN;
  true_p (n,96) f32;status;item_ids/cond_ids/row_globals)+ ledger_t47.csv +
  sfdx_meta_t47.json。

用法: python infer_sfdx_t47.py <repo_root> [save_dir] [--smoke] [--workers N] [--timeout S] [--selftest]
"""
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from collections import Counter

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np

H = 96
LEDGER_FIELDS = ['row_global', 'sid_t46', 'ds', 'cond_id', 'method', 'status',
                 'elapsed_s', 'note']


def worker_main(task_q, result_q, method, save_dir):
    model_cls = None  # 懒加载:selftest 不依赖 statsforecast
    base_cache = {}   # ds -> (x_f64, y_f64, sid_row)
    while True:
        task = task_q.get()
        if task is None:
            break
        rg, sid, ds, cid, delta, m_steps, group, P = task
        t0 = time.time()
        try:
            if P == -1:  # selftest 慢任务
                time.sleep(delta)
                result_q.put((rg, 'ok', np.arange(H, dtype=np.float64), time.time() - t0))
                continue
            if ds not in base_cache:
                bz = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
                base_cache[ds] = (bz['x'], bz['y'], {int(s): i for i, s in enumerate(bz['sid_t46'])})
            x_f64, y_f64, sid_row = base_cache[ds]
            w = sid_row[sid]
            from t47_common import apply_variant  # spawn 子进程 sys.path 含脚本目录
            x = apply_variant(x_f64[w], delta, m_steps, group).astype(np.float64)
            if model_cls is None:
                from statsforecast.models import AutoARIMA, AutoETS
                model_cls = (AutoETS, AutoARIMA)
            m = model_cls[0](season_length=P) if method == 'AutoETS' else model_cls[1](season_length=P)
            m.fit(x)
            f = m.predict(h=H)
            pred = np.asarray(f['mean'], dtype=np.float64)
            assert pred.shape == (H,)
            if np.isfinite(pred).all():
                result_q.put((rg, 'ok', pred, time.time() - t0))
            else:
                result_q.put((rg, 'nonfinite', None, time.time() - t0))
        except Exception as e:  # 单样本失败,入台账不中止
            result_q.put((rg, 'error:' + str(e)[:180], None, time.time() - t0))


class TimeoutPool:
    """每 worker 单在跑任务;到期真杀;worker_died 视为单样本失败。

    mp 用 spawn 上下文:主进程已 import statsforecast(numba 已加载),
    fork 后在继承的 LLVM/JIT 状态上编译是不安全区(t46 冒烟实测 worker
    静默死亡);spawn 全新解释器规避该问题。
    """

    def __init__(self, n_workers, method, timeout_s, save_dir):
        self.method, self.timeout, self.save_dir = method, timeout_s, save_dir
        self.ctx = mp.get_context('spawn')
        self.task_qs, self.result_qs, self.procs = [], [], []
        self.events = Counter()
        for _ in range(n_workers):
            self._spawn()

    def _spawn(self):
        w = len(self.task_qs)
        tq, rq = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=worker_main, args=(tq, rq, self.method, self.save_dir),
                             daemon=True)
        p.start()
        self.task_qs.append(tq)
        self.result_qs.append(rq)
        self.procs.append(p)

    def _kill(self, w, reason):
        self.events[reason] += 1
        print('[pool] worker %d killed (%s), respawn' % (w, reason), flush=True)
        try:
            self.procs[w].terminate()
            self.procs[w].join(5)
        except Exception:
            pass
        if self.procs[w].is_alive():
            self.procs[w].kill()
            self.procs[w].join(5)
        self.task_qs[w], self.result_qs[w] = self.ctx.Queue(), self.ctx.Queue()
        p = self.ctx.Process(target=worker_main,
                             args=(self.task_qs[w], self.result_qs[w], self.method, self.save_dir),
                             daemon=True)
        p.start()
        self.procs[w] = p

    def _send(self, w, task):
        try:
            self.task_qs[w].put(task, timeout=10)
            return True
        except Exception:
            return False

    def run(self, tasks, ledger):
        """tasks: [(rg, sid, ds, cid, delta, m_steps, group, P)];返回 {rg: (status, pred, elapsed)}。"""
        results = {}
        inflight = {}  # worker -> (rg, deadline, ds, sid, cid)
        nxt = 0
        while nxt < len(tasks) or inflight:
            for w in range(len(self.procs)):
                if w not in inflight:
                    if not self.procs[w].is_alive():
                        self._kill(w, 'idle_worker_died')
                    elif nxt < len(tasks):
                        t = tasks[nxt]
                        if self._send(w, t):
                            inflight[w] = (t[0], time.time() + self.timeout, t[2], t[1], t[3])
                            nxt += 1
                        else:
                            self._kill(w, 'send_failed')
            for w in list(inflight):
                rg, deadline, ds, sid, cid = inflight[w]
                got = None
                try:
                    got = self.result_qs[w].get(timeout=0.05)
                except queue.Empty:
                    pass
                if got is not None:
                    rg2, status, pred, el = got
                    results[rg2] = (status, pred, el)
                    ledger.append([rg2, sid, ds, cid, self.method, status, '%.3f' % el, ''])
                    del inflight[w]
                elif not self.procs[w].is_alive():
                    results[rg] = ('worker_died', None, float('nan'))
                    ledger.append([rg, sid, ds, cid, self.method, 'worker_died', '', 'process died'])
                    self._kill(w, 'worker_died')
                    del inflight[w]
                elif time.time() > deadline:
                    self._kill(w, 'timeout')
                    results[rg] = ('timeout', None, float('nan'))
                    ledger.append([rg, sid, ds, cid, self.method, 'timeout', '>%d' % self.timeout, 'killed'])
                    del inflight[w]
        return results

    def close(self):
        for tq in self.task_qs:
            try:
                tq.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(3)


def build_tasks(man, method, smoke):
    rows = []
    for r in man:
        if method == 'AutoARIMA':
            if r['ds'] not in ('ETTh1', 'ETTh2'):
                continue
            if int(r['rank']) > 1:
                continue
        rows.append(r)
    if smoke:
        assert len(rows) > 0
    tasks = []
    for r in rows:
        P = int(r['P'])
        m_steps = int(r['m_steps']) if r['m_steps'] != '' else 0
        tasks.append((int(r['row_global']), int(r['sid_t46']), r['ds'], int(r['cond_id']),
                      float(r['delta']), m_steps, r['group'], P))
    return rows, tasks


def main(repo_root, save_dir, smoke=False, n_workers=8, timeout_s=1800, selftest=False):
    print('sfdx start: workers=%d timeout=%ss smoke=%s selftest=%s'
          % (n_workers, timeout_s, smoke, selftest), flush=True)
    from t47_common import manifest_path, smoke_subset

    os.makedirs(save_dir, exist_ok=True)
    with open(manifest_path()) as f:
        man = list(csv.DictReader(f))
    assert len(man) == 7168
    if smoke:
        man = smoke_subset(man)

    suffix = '_smoke' if smoke else ''
    all_ledger = []
    meta = dict(workers=n_workers, timeout_s=timeout_s, thread_limits='all=1',
                seeds='n/a (deterministic statistical fits)', per_method={})

    if selftest:
        timeout_s = 5
        tasks = [(i, 0, 'SELFTEST', 0, 30.0 if i == 3 else 0.01, 0, 'A', -1) for i in range(8)]
        pool = TimeoutPool(4, 'SELFTEST', timeout_s, save_dir)
        res = pool.run(tasks, all_ledger)
        pool.close()
        statuses = [res[i][0] for i in sorted(res)]
        assert statuses[3] == 'timeout', statuses
        assert all(s == 'ok' for i, s in enumerate(statuses) if i != 3), statuses
        print('selftest PASS: timeout killed + pool continued:', statuses)
        return

    import inspect
    from statsforecast.models import AutoARIMA, AutoETS
    import statsforecast
    meta.update(statsforecast_version=getattr(statsforecast, '__version__', '?'),
                AutoETS=str(inspect.signature(AutoETS.__init__)),
                AutoARIMA=str(inspect.signature(AutoARIMA.__init__)))

    for method in ('AutoETS', 'AutoARIMA'):
        rows, tasks = build_tasks(man, method, smoke)
        pool = TimeoutPool(n_workers, method, timeout_s, save_dir)
        t0 = time.time()
        res = pool.run(tasks, all_ledger)
        elapsed = time.time() - t0
        pool.close()
        assert set(res) == {int(r['row_global']) for r in rows}, method
        n = len(rows)
        pred = np.full((n, H), np.nan, np.float32)
        true_p = np.full((n, H), np.nan, np.float32)
        status_col = []
        base_cache = {}   # ds -> (y_f64, sid_row)
        for j, r in enumerate(rows):
            st, p, el = res[int(r['row_global'])]
            status_col.append(st)
            if st == 'ok':
                pred[j] = p.astype(np.float32)
            ds = r['ds']
            if ds not in base_cache:
                bz = np.load(os.path.join(save_dir, 'baseline_%s.npz' % ds))
                base_cache[ds] = (bz['y'], {int(s): i for i, s in enumerate(bz['sid_t46'])})
            y_f64, sid_row = base_cache[ds]
            true_p[j] = (y_f64[sid_row[int(r['sid_t46'])]] + float(r['delta'])).astype(np.float32)
        np.savez_compressed(os.path.join(save_dir, 'pred_%s_t47%s.npz' % (method.lower(), suffix)),
                            pred=pred, true_p=true_p,
                            status=np.array(status_col),
                            item_ids=np.array([int(r['sid_t46']) for r in rows], dtype=np.int64),
                            cond_ids=np.array([int(r['cond_id']) for r in rows], dtype=np.int64),
                            row_globals=np.array([int(r['row_global']) for r in rows], dtype=np.int64))
        n_ok = sum(1 for x in status_col if x == 'ok')
        per = dict(n=n, ok=n_ok,
                   timeout=sum(1 for x in status_col if x == 'timeout'),
                   nonfinite=sum(1 for x in status_col if x == 'nonfinite'),
                   other_fail=sum(1 for x in status_col if x not in ('ok', 'timeout', 'nonfinite')),
                   wall_seconds=round(elapsed, 1),
                   row_globals=[int(r['row_global']) for r in rows])
        per['pool_events'] = dict(pool.events)
        meta['per_method'][method] = per
        print(method, {k: v for k, v in per.items() if k != 'row_globals'}, flush=True)

    with open(os.path.join(save_dir, 'sfdx_meta_t47%s.json' % suffix), 'w') as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(save_dir, 'ledger_t47%s.csv' % suffix), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(LEDGER_FIELDS)
        w.writerows(all_ledger)
    print('sfdx done; ledger rows =', len(all_ledger))


def _arg_val(name, default):
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return default


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0], args[1] if len(args) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_t47_levelshift'),
         smoke='--smoke' in sys.argv,
         n_workers=int(_arg_val('--workers', 8)),
         timeout_s=int(_arg_val('--timeout', 1800)),
         selftest='--selftest' in sys.argv)
