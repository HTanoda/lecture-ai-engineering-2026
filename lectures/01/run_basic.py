# Copyright (c) 2026 HIROKI TANODA (TND). All rights reserved.
"""第1回・基礎: 2モデル、2質問、各3回。既存結果を保持して再開する。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'lectures/01/basic_results.jsonl'
MODELS = ('gemini-3.5-flash-lite', 'gemini-3.8-flash')
QUESTIONS = ('LoRAとは何ですか？', 'AIE2026の講座の宿題にある「3レベル」とは何ですか？')

def main():
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise SystemExit('GEMINI_API_KEY を設定してください（値は表示しません）。')
    rows = [json.loads(s) for s in OUTPUT.read_text(encoding='utf-8').splitlines()] if OUTPUT.exists() else []
    done = {(r['model'], r['question'], r['trial']) for r in rows}
    jobs = [(m, q, n) for q in QUESTIONS for n in range(1, 4) for m in MODELS if (m, q, n) not in done]
    lock = Lock()
    def run(job):
        model, question, trial = job
        record = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=model, question=question, trial=trial)
        try:
            result = subprocess.run([sys.executable, '-m', 'app.ask', question], cwd=ROOT, env=dict(os.environ, AIE_MODEL=model), capture_output=True, text=True, timeout=120)
            record.update(returncode=result.returncode, stdout=result.stdout.replace(key, '[REDACTED]'), stderr=result.stderr.replace(key, '[REDACTED]'))
        except subprocess.TimeoutExpired:
            record.update(returncode=None, stdout='', stderr='120秒でタイムアウト。回答は未取得。')
        with lock:
            with OUTPUT.open('a', encoding='utf-8') as output:
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(model, QUESTIONS.index(question)+1, trial, 'exit=', record['returncode'], flush=True)
    # 開始は全体で15秒以上の間隔。最大4件の応答待ちを重ねる。
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for i, job in enumerate(jobs):
            futures.append(pool.submit(run, job))
            if i + 1 < len(jobs):
                time.sleep(15)
        for future in futures:
            future.result()
    print('DONE', len(done) + len(jobs), flush=True)

if __name__ == '__main__':
    main()
