import argparse
import asyncio
import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from benchtop import study
from benchtop.cache import file_hash, runtime_identity
from benchtop.env import ROOT
from benchtop.sandbox import Pool

PROFILES = {
    '2610-36935-gaussian-process-bandits-for': (
        ['epsilon_greedy', 'gp_ucb_matern_linear'],
        [{'environment': 'stationary', 'noise_sd': 0.1, 'delay': 0},
         {'environment': 'drift', 'noise_sd': 0.3, 'delay': 3},
         {'environment': 'reactive', 'noise_sd': 0.3, 'delay': 3}]),
    '2610-36941-algorithmic-monoculture-in-machine': (
        ['monoculture', 'portfolio', 'uniform'],
        [{'kappa': 0.0, 'gap': 'small', 'cap': 0.05},
         {'kappa': 0.95, 'gap': 'large', 'cap': 0.25}]),
    '2610-36983-advancement-or-acceleration-how': (
        ['uniform', 'target_select', 'random_select'],
        [{'gap_deg': 0, 'n_pool': 200, 'label_noise': 0.0},
         {'gap_deg': 60, 'n_pool': 2000, 'label_noise': 0.2}]),
}
OBSERVER = '''

_probe_optimizer_rows = []
_probe_original_minimize = minimize
_probe_original_run = run

def _probe_minimize(*args, **kwargs):
    result = _probe_original_minimize(*args, **kwargs)
    _probe_optimizer_rows.append({"success": bool(result.success), "iterations": int(result.nit), "finite": bool(np.all(np.isfinite(result.x)) and np.isfinite(result.fun))})
    return result

minimize = _probe_minimize

def run(config, seed):
    _probe_optimizer_rows.clear()
    result = _probe_original_run(config, seed)
    result["probe_optimizer_calls"] = float(len(_probe_optimizer_rows))
    result["probe_optimizer_failures"] = float(sum(not row["success"] for row in _probe_optimizer_rows))
    result["probe_nonfinite_optimizer_results"] = float(sum(not row["finite"] for row in _probe_optimizer_rows))
    result["probe_max_optimizer_iterations"] = float(max((row["iterations"] for row in _probe_optimizer_rows), default=0))
    return result
'''


def equal(left, right):
    return set(left) == set(right) and all(math.isclose(left[key], right[key], rel_tol=1e-12, abs_tol=1e-12) for key in left)


async def execute(folder, name, source, jobs):
    path = folder / name
    path.mkdir()
    lab = path / 'lab.py'
    lab.write_text(source)
    rows = []
    async with Pool(lab, workers=1, timeout=90) as pool:
        for config, seed in jobs:
            result = await pool.run(config, seed)
            if not result.get('ok'):
                raise RuntimeError(name + ': ' + result.get('error', 'missing result'))
            rows.append({'config': config, 'seed': seed, 'metrics': result['metrics'], 'reported_job_seconds': result['secs']})
    return {'executed_source_sha256': file_hash(lab), 'rows': rows}


async def main():
    parser = argparse.ArgumentParser(description='Run the fixed bounded visible-seed runtime review; no models or stored-data changes.')
    parser.add_argument('--output', default=str(ROOT / 'artifacts/reference_runtime_review.json'))
    output = Path(parser.parse_args().output).resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('write a new diagnostic under artifacts/ or outside the checkout')
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    members = []
    with tempfile.TemporaryDirectory(prefix='benchtop-reference-review-') as temporary:
        folder = Path(temporary)
        for env_id, (methods, contexts) in PROFILES.items():
            source_path = ROOT / 'envs' / env_id
            snapshot = study._snapshot(source_path)
            source = (source_path / 'lab.py').read_text()
            jobs = [({**context, 'method': method}, seed) for seed in (0, 1) for context in contexts for method in methods]
            first = await execute(folder, env_id + '-pristine', source, jobs)
            repeated_jobs = [job for job in jobs if job[1] == 0]
            repeated = await execute(folder, env_id + '-fresh-worker', source, repeated_jobs)
            by_key = {study.canonical_hash([row['config'], row['seed']]): row for row in first['rows']}
            repeatable = all(equal(row['metrics'], by_key[study.canonical_hash([row['config'], row['seed']])]['metrics']) for row in repeated['rows'])
            item = {'env_id': env_id, 'source_files': snapshot['files'], 'visible_seed_ids': [0, 1],
                'pristine': first, 'fresh_worker': repeated, 'fresh_worker_numeric_repeat': repeatable,
                'observed_contexts': contexts, 'scope': 'Bounded existing visible seeds and declared contexts only; no qualification or confirmation runs and no change to stored caches.'}
            if env_id != '2610-36935-gaussian-process-bandits-for':
                observed = await execute(folder, env_id + '-observed', source + OBSERVER, repeated_jobs)
                item['optimizer_observer'] = observed
                item['observer_transparency'] = all(equal({key: value for key, value in row['metrics'].items() if not key.startswith('probe_')},
                    by_key[study.canonical_hash([row['config'], row['seed']])]['metrics']) for row in observed['rows'])
                item['optimizer_failures'] = sum(row['metrics']['probe_optimizer_failures'] for row in observed['rows'])
                item['nonfinite_optimizer_results'] = sum(row['metrics']['probe_nonfinite_optimizer_results'] for row in observed['rows'])
            if study._snapshot(source_path) != snapshot:
                raise RuntimeError('Original source changed during runtime review')
            members.append(item)
            print(json.dumps({'env_id': env_id, 'fresh_worker_repeatable': repeatable,
                'observer_transparency': item.get('observer_transparency'), 'optimizer_failures': item.get('optimizer_failures')}), flush=True)
    result = {'schema_version': 1, 'review_kind': 'bounded_visible_reference_runtime_review',
        'recorded_at_utc': datetime.now(UTC).isoformat(), 'members': members,
        'sandbox_jobs': sum(len(row[key]['rows']) for row in members for key in ('pristine', 'fresh_worker', 'optimizer_observer') if key in row),
        'wall_seconds': time.perf_counter() - started, 'runtime': runtime_identity(),
        'source_script_sha256': file_hash(Path(__file__)), 'model_sessions': 0, 'stored_results_modified': False,
        'human_admissions_created': 0, 'scope': 'Measured repeatability and optimizer diagnostics on unchanged originals/explicit observer copies. No paper replication, full-grid validity, margin approval or scientific admission follows automatically.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    study._publish(output, result)
    print(json.dumps({'saved': str(output), 'sandbox_jobs': result['sandbox_jobs'], 'seconds': result['wall_seconds'], 'model_sessions': 0}), flush=True)


asyncio.run(main())
