"""EXP-046:StatsForecast AutoETS / AutoARIMA 窗口内统计拟合对照。

- 每样本仅用该输入窗口拟合(标准化历史全长 L),season_length=P,其余默认;
  安装版本的构造签名快照进 meta;锁定 statsforecast 版本;
- 并发:持久 worker 进程池(默认 8,数值库线程=1/进程);真超时:join 到期后
  terminate()+kill() 杀死在跑 worker,该样本记 timeout,随后池自动补位继续
  (--selftest 验证杀进程后续任务可继续);
- 失败分类:timeout / nonfinite / error(单样本进台账,不中止);管线级异常才中止;
- Moirai 风格无 20s 硬门;逐样本耗时记录。

用法: python infer_sfdx_t46.py <repo_root> [save_dir] [--smoke] [--workers N] [--timeout S] [--selftest]
"""
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np

H = 96
LEDGER_FIELDS = ['sample_id', 'ds', 'method', 'status', 'elapsed_s', 'note']


def worker_main(task_q, result_q, method):
    model_cls = None  # 懒加载:selftest 不依赖 statsforecast;库缺失时正式跑在 main() 先行中止
    while True:
        task = task_q.get()
        if task is None:
            break
        sid, y, P = task
        t0 = time.time()
        try:
            if P == -1:  # selftest 慢任务
                time.sleep(y[0])
                result_q.put((sid, 'ok', np.arange(H, dtype=np.float64), time.time() - t0))
                continue
            if model_cls is None:
                from statsforecast.models import AutoARIMA, AutoETS
                model_cls = (AutoETS, AutoARIMA)
            m = model_cls[0](season_length=P) if method == 'AutoETS' else model_cls[1](season_length=P)
            m.fit(y)
            f = m.predict(h=H)
            pred = np.asarray(f['mean'], dtype=np.float64)
            assert pred.shape == (H,)
            if np.isfinite(pred).all():
                result_q.put((sid, 'ok', pred, time.time() - t0))
            else:
                result_q.put((sid, 'nonfinite', None, time.time() - t0))
        except Exception as e:  # 单样本失败,入台账不中止
            result_q.put((sid, 'error:' + str(e)[:180], None, time.time() - t0))


class TimeoutPool:
    """每 worker 单在跑任务;到期真杀;worker_died 视为单样本失败。"""

    def __init__(self, n_workers, method, timeout_s):
        self.method, self.timeout = method, timeout_s
        self.task_qs, self.result_qs, self.procs = [], [], []
        for _ in range(n_workers):
            self._spawn()

    def _spawn(self):
        w = len(self.task_qs)
        tq, rq = mp.Queue(), mp.Queue()
        p = mp.Process(target=worker_main, args=(tq, rq, self.method), daemon=True)
        p.start()
        self.task_qs.append(tq)
        self.result_qs.append(rq)
        self.procs.append(p)

    def _kill(self, w):
        try:
            self.procs[w].terminate()
            self.procs[w].join(5)
        except Exception:
            pass
        if self.procs[w].is_alive():
            self.procs[w].kill()
            self.procs[w].join(5)
        # 换新队列,避免残留任务串扰
        self.task_qs[w], self.result_qs[w] = mp.Queue(), mp.Queue()
        p = mp.Process(target=worker_main, args=(self.task_qs[w], self.result_qs[w], self.method),
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
        """tasks: [(sid, y, P)];返回 {sid: (status, pred, elapsed)}。"""
        results = {}
        inflight = {}  # worker -> (sid, deadline, ds)
        nxt = 0
        while nxt < len(tasks) or inflight:
            for w in range(len(self.procs)):
                if w not in inflight and nxt < len(tasks):
                    sid, y, P, ds = tasks[nxt]
                    if self._send(w, (sid, y, P)):
                        inflight[w] = (sid, time.time() + self.timeout, ds)
                        nxt += 1
                    else:
                        self._kill(w)
            for w in list(inflight):
                sid, deadline, ds = inflight[w]
                got = None
                try:
                    got = self.result_qs[w].get(timeout=0.05)
                except queue.Empty:
                    pass
                if got is not None:
                    sid2, status, pred, el = got
                    results[sid2] = (status, pred, el)
                    ledger.append([sid2, ds, self.method, status, '%.3f' % el, ''])
                    del inflight[w]
                elif not self.procs[w].is_alive():
                    results[sid] = ('worker_died', None, float('nan'))
                    ledger.append([sid, ds, self.method, 'worker_died', '', 'process died'])
                    self._kill(w)
                    del inflight[w]
                elif time.time() > deadline:
                    self._kill(w)
                    results[sid] = ('timeout', None, float('nan'))
                    ledger.append([sid, ds, self.method, 'timeout', '>%d' % self.timeout, 'killed'])
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


def main(repo_root, save_dir, smoke=False, n_workers=8, timeout_s=300, selftest=False):
    from t46_common import DS_CFG, DS_ORDER, load_csv, sample_list_path, std_all, train_scaler

    os.makedirs(save_dir, exist_ok=True)
    samples = list(csv.DictReader(open(sample_list_path())))
    if smoke:
        samples = [s for s in samples if s['ds'] in ('ETTh1', 'ETTm1')]
        groups = ['ETTh1', 'ETTm1']
    else:
        groups = list(DS_ORDER)

    suffix = '_smoke' if smoke else ''
    all_ledger = []
    meta = dict(workers=n_workers, timeout_s=timeout_s, thread_limits='all=1',
                seeds='n/a (deterministic statistical fits)', per_ds={})

    if selftest:
        # 慢任务(>timeout)应被杀且记 timeout;其后任务继续成功 → 验证管线存活
        timeout_s = 5
        tasks = [(i, (float(30 if i == 3 else 0.01),), -1, 'SELFTEST') for i in range(8)]
        pool = TimeoutPool(4, 'SELFTEST', timeout_s)
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
                AutoARIMA=str(inspect.signature(AutoARIMA.__init__)),
                AutoETS_defaults={k: repr(v) for k, v in vars(AutoETS(season_length=24)).items()
                                  if not k.startswith('_')})

    for method in ('AutoETS', 'AutoARIMA'):
        per_method = {}
        pool = TimeoutPool(n_workers, method, timeout_s)
        for ds in groups:
            cfg = DS_CFG[ds]
            ss = [s for s in samples if s['ds'] == ds]
            if smoke:
                assert len(ss) >= 8, (ds, len(ss))
                ss = ss[:8]
            data = load_csv(os.path.join(repo_root, cfg['csv']))
            mean, scale = train_scaler(data, cfg['K'])
            std = std_all(data, mean, scale)
            tasks = []
            truth_rows = []
            for s in ss:
                start, ch = int(s['start_row']), int(s['channel'])
                y = std[start - cfg['L']:start, ch].astype(np.float64)
                assert len(y) == cfg['L']
                tasks.append((int(s['sample_id']), y, cfg['P'], ds))
                truth_rows.append(std[start:start + H, ch])
            t0 = time.time()
            res = pool.run(tasks, all_ledger)
            elapsed = time.time() - t0
            assert set(res) == {int(s['sample_id']) for s in ss}
            pred = np.full((len(ss), H), np.nan, np.float32)
            status_col = []
            for j, s in enumerate(ss):
                st, p, el = res[int(s['sample_id'])]
                status_col.append(st)
                if st == 'ok':
                    pred[j] = p.astype(np.float32)
            true = np.stack(truth_rows).astype(np.float32)
            np.savez_compressed(os.path.join(save_dir, f'{method}{suffix}_{ds}.npz'),
                                pred=pred, true=true,
                                status=np.array(status_col),
                                item_ids=np.array([int(s['sample_id']) for s in ss], dtype=np.int64))
            n_ok = sum(1 for x in status_col if x == 'ok')
            per_method[ds] = dict(n=len(ss), ok=n_ok,
                                  timeout=sum(1 for x in status_col if x == 'timeout'),
                                  nonfinite=sum(1 for x in status_col if x == 'nonfinite'),
                                  other_fail=sum(1 for x in status_col if x not in ('ok', 'timeout', 'nonfinite')),
                                  wall_seconds=round(elapsed, 1),
                                  sample_order=[int(s['sample_id']) for s in ss])
            print(method, ds, {k: v for k, v in per_method[ds].items() if k != 'sample_order'})
        pool.close()
        meta[method] = per_method

    with open(os.path.join(save_dir, f'sfdx_meta{suffix}.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(save_dir, f'failure_ledger{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(LEDGER_FIELDS)
        w.writerows(all_ledger)
    print('sfdx done; ledger rows =', len(all_ledger))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0], args[1] if len(args) > 1 else 'save_t46_histdiag',
         smoke='--smoke' in sys.argv,
         n_workers=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--workers=')), 8)),
         timeout_s=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--timeout=')), 300)),
         selftest='--selftest' in sys.argv)
