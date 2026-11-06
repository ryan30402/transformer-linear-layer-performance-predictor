"""Self-contained Transformer Linear Layer Performance Predictor launcher.

Built from a hashed source ZIP.

--self-check validates the payload and Python syntax without network or CUDA.
No PyTorch/CUDA installation, automatic GPU rental, or automatic retry occurs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile

VERSION = '1.1.0'
SOURCE_SHA256 = '__SOURCE_SHA256__'
SOURCE_B64 = '__SOURCE_B64__'
ANALYSIS_LOCK = {'numpy': '2.3.5', 'scipy': '1.17.0', 'matplotlib': '3.10.8'}


def payload():
    raw = base64.b64decode(SOURCE_B64, validate=True)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Embedded source archive SHA256 mismatch')
    return raw


def checked_members(raw, *, limit=80_000_000):
    """Reject traversal, links, duplicate members and excessive expanded size."""
    archive = zipfile.ZipFile(io.BytesIO(raw))
    members, names, total = [], set(), 0
    if len(archive.infolist()) > 6000:
        raise ValueError('Archive has too many members')
    for entry in archive.infolist():
        name = entry.filename
        path = PurePosixPath(name)
        mode = entry.external_attr >> 16
        if (not name or '\\' in name or '\x00' in name or path.is_absolute()
                or any(p in ('..', '.', '') for p in name.rstrip('/').split('/'))
                or ':' in name or name in names or entry.flag_bits & 1
                or stat.S_ISLNK(mode)
                or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
            raise ValueError('Unsafe archive member: ' + name)
        names.add(name)
        total += entry.file_size
        if total > limit:
            raise ValueError('Archive exceeds expanded-size budget')
        members.append(entry)
    return archive, members


def safe_extract(raw, destination, *, limit=80_000_000):
    root = Path(destination).resolve()
    if root.exists():
        raise ValueError('Extraction destination already exists: ' + str(root))
    archive, members = checked_members(raw, limit=limit)
    root.mkdir(parents=True)
    with archive:
        for entry in members:
            target = root.joinpath(*PurePosixPath(entry.filename).parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError('Archive escapes destination')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as handle:
                    handle.write(archive.read(entry))
    return root


def self_check():
    raw = payload()
    archive, members = checked_members(raw)
    compiled = 0
    with archive:
        for entry in members:
            if entry.filename.endswith('.py'):
                compile(archive.read(entry), entry.filename, 'exec')
                compiled += 1
        root = f'transformer-linear-layer-performance-predictor-{VERSION}/'
        if any(entry.filename.startswith(root + 'results/') or entry.filename.startswith(root + 'analysis/') for entry in members):
            raise ValueError('The fresh measurement payload must not contain historical result datasets')
        for name in ('workloads.json', 'workloads_smoke.json'):
            plan = json.loads(archive.read(root + 'configs/' + name))
            if (plan.get('protocol_version') != VERSION or not plan.get('cases')
                    or not {'train', 'test', 'challenge', 'stress'} <= {c.get('split') for c in plan['cases']}):
                raise ValueError('Unexpected embedded workload manifest')
    return {'status': 'self_check_passed', 'version': VERSION,
            'source_sha256': SOURCE_SHA256, 'python_files_compiled': compiled,
            'network_used': False, 'cuda_executed': False}


def prepare_source(workspace):
    root = Path(workspace).resolve()
    if root.exists():
        raise ValueError('Workspace already exists; preserve it and choose a new session/review ID')
    root.mkdir(parents=True)
    raw = payload()
    (root / 'source_archive.zip').write_bytes(raw)
    (root / 'source_archive.sha256').write_text(SOURCE_SHA256 + '\n', encoding='ascii')
    safe_extract(raw, root / 'source')
    source = root / 'source' / f'transformer-linear-layer-performance-predictor-{VERSION}'
    if not (source / 'linear_perf' / 'cli.py').is_file():
        raise ValueError('Embedded source layout is invalid')
    return source


def dependency_versions():
    result = {}
    for name in ANALYSIS_LOCK:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'unavailable'
    return result


def ensure_dependencies(source, *, pin=True):
    if pin and sys.version_info < (3, 11):
        raise RuntimeError('The tested analysis lock requires Python 3.11+. Select a compatible Colab runtime.')
    before = dependency_versions()
    if pin and before != ANALYSIS_LOCK:
        print('Installing only the recorded NumPy/SciPy/Matplotlib stack; retaining runtime PyTorch/CUDA.', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '-q', '-r', str(Path(source) / 'requirements-analysis-lock.txt')], check=True, timeout=600)
    elif not pin and 'unavailable' in before.values():
        raise RuntimeError('Missing analysis dependencies. Enable PIN_ANALYSIS_DEPS or install requirements first.')
    after = dependency_versions()
    if pin and after != ANALYSIS_LOCK:
        raise RuntimeError('Analysis versions differ from the release lock after installation')
    return {'before': before, 'actual': after, 'release_lock_used': pin,
            'pytorch_reinstalled': False, 'cuda_reinstalled': False}


def worker(command, *, source, log_path, timeout=1920):
    """Supervise one worker. Only the child writes files while it is active.

    stdout goes directly to a child-owned log descriptor. The parent only
    tails it for notebook progress: no competing parent log writes, background
    filesystem mutations, or retry after failure.
    """
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(source)
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('OMP_NUM_THREADS', '1')
    environment.setdefault('OPENBLAS_NUM_THREADS', '1')
    environment.setdefault('MPLCONFIGDIR', str(Path(log_path).parent / 'matplotlib_cache'))
    started = time.monotonic()
    log_handle = Path(log_path).open('xb', buffering=0)
    try:
        proc = subprocess.Popen(command, cwd=source, env=environment, stdout=log_handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=os.name == 'posix')
    except BaseException:
        log_handle.close()
        raise
    timed_out, interrupted = False, False

    def stop():
        if proc.poll() is not None:
            return
        try:
            if os.name == 'posix':
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == 'posix':
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    with log_handle as log, Path(log_path).open('r', encoding='utf-8', errors='replace') as reader:
        def show_new_output():
            chunk = reader.read()
            if chunk:
                print(chunk, end='', flush=True)

        try:
            while proc.poll() is None:
                show_new_output()
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    stop()
                    break
                time.sleep(.15)
        except KeyboardInterrupt:
            interrupted = True
            stop()
        finally:
            if proc.poll() is None:
                stop()
            show_new_output()
            log.flush(); os.fsync(log.fileno())
    return {'returncode': proc.returncode, 'timed_out': timed_out, 'interrupted': interrupted,
            'elapsed_seconds': time.monotonic() - started, 'automatic_retries': 0,
            'timeout_seconds': timeout}


def write_status(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def checkpoint_module(source):
    """Load the transfer helper from the verified embedded source, not an input ZIP."""
    path = Path(source) / 'linear_perf' / 'checkpoints.py'
    name = '_linear_perf_checkpoints_' + SOURCE_SHA256[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('The embedded checkpoint helper is unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def export_results(workspace, archive_path, *, extra_directories=()):
    root, destination = Path(workspace).resolve(), Path(archive_path).resolve()
    if destination.exists():
        raise ValueError('Result archive already exists; refusing to overwrite it')
    sidecar = destination.with_name(destination.name + '.sha256')
    if sidecar.exists():
        raise ValueError('Result archive checksum already exists; refusing to overwrite it')
    top_files = ('source_archive.zip', 'source_archive.sha256', 'launcher_status.json', 'worker.log')
    files = [root / name for name in top_files if (root / name).is_file()]
    for directory in ('run',) + tuple(extra_directories):
        base = root / directory
        if base.exists():
            files.extend(p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts)
    # Keep any failure envelopes with the run.
    files.extend(sorted(root.glob('run-*.json')))
    inventory = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Cannot export linked/outside evidence')
        inventory[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_status(root / 'export_manifest.json', {'source_sha256': SOURCE_SHA256, 'files': inventory,
                                               'scope': 'Internal archive integrity; no hardware attestation'})
    files.append(root / 'export_manifest.json')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    archive_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    with sidecar.open('x', encoding='ascii') as handle:
        handle.write(archive_sha + '  ' + destination.name + '\n')
    return str(destination)


def run(*, session_id='paired_s01', dtypes=('float16', 'bfloat16'), action='run',
        profile='standard', simulate=False, repeat_subset=False, skip_profiles=False,
        max_wall_seconds=7200, l2_cache_mib=None, pin_analysis_deps=True, base_directory=None,
        include_repeat=True, include_layer=False, backup_directory=None):
    """Collect all splits, then optionally perform ordinary offline modeling."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', session_id):
        raise ValueError('Invalid session ID')
    if (not dtypes or len(set(dtypes)) != len(dtypes) or
            any(d not in ('float16', 'bfloat16') for d in dtypes) or
            action not in ('run', 'collect', 'experiment') or profile not in ('standard', 'smoke')):
        raise ValueError('Invalid precision, action or workload profile')
    if type(max_wall_seconds) is not int or max_wall_seconds <= 0:
        raise ValueError('Measurement budget must be a positive number of seconds')
    if type(include_repeat) is not bool or type(include_layer) is not bool:
        raise ValueError('Experiment stage options must be boolean')
    if action == 'experiment' and (len(session_id) > 48 or repeat_subset):
        raise ValueError('Use an experiment ID of at most 48 characters; full experiments select their own repeat subset')
    base = Path(base_directory or Path.cwd()).resolve()
    workspace = base / ('linear-perf_' + session_id + '_workspace')
    archive = base / ('linear-perf_' + session_id + '_RESULTS.zip')
    backup_session = (Path(backup_directory).expanduser().resolve() / ('linear-perf_' + session_id)
                      if backup_directory is not None else None)
    if backup_session is not None:
        if backup_session.is_relative_to(workspace) or workspace.is_relative_to(backup_session):
            raise ValueError('Backup storage must be outside the active experiment workspace')
        if backup_session.exists():
            raise ValueError('A backup folder already exists for this session; preserve it and choose a new ID')
    if archive.exists():
        raise ValueError('Preserve the existing result ZIP; choose a new session ID')
    source = prepare_source(workspace)
    status = {'version': VERSION, 'session_id': session_id, 'action': action, 'dtypes': list(dtypes),
              'profile': profile, 'repeat_subset': repeat_subset, 'simulate': bool(simulate),
              'source_sha256': SOURCE_SHA256, 'status': 'preparing', 'python_version': sys.version,
              'automatic_retries': 0, 'hardware_accuracy_validated': False,
              'gpu_allocation_performed': False, 'workflow': 'collect_all_then_offline_cv_refit_evaluate'}
    status['backup'] = {'requested': backup_session is not None,
                        'directory': str(backup_session) if backup_session is not None else None,
                        'scope': 'Verified checkpoint copies; final archive copy is reported separately',
                        'background_transfers': False, 'automatic_resume': False}
    if action == 'experiment':
        status.update(workflow='fresh_primary_repeat_comparison', include_repeat=include_repeat,
                      include_layer=include_layer, historical_results_used=False,
                      per_collection_budget_seconds=max_wall_seconds)
    write_status(workspace / 'launcher_status.json', status)
    try:
        if backup_session is not None:
            backup_session.mkdir(parents=True, exist_ok=False)
            print('Backup folder:', backup_session, flush=True)
            print('A verified initial checkpoint is required before GPU measurement starts.', flush=True)
        status['dependencies'] = ensure_dependencies(source, pin=pin_analysis_deps)
        config = 'workloads_smoke.json' if profile == 'smoke' else 'workloads.json'
        command = [sys.executable, '-u', '-m', 'linear_perf', action,
                   '--config', str(source / 'configs' / config), '--out', str(workspace / 'run'),
                   '--session-id', session_id, '--max-wall-seconds', str(max_wall_seconds),
                   '--dtypes', *dtypes]
        if simulate:
            command.append('--simulate')
        if repeat_subset:
            command.append('--repeat-subset')
        if action == 'experiment':
            if not include_repeat:
                command.append('--skip-repeat')
            if include_layer:
                command.append('--include-layer')
        if skip_profiles:
            command.append('--skip-profiles')
        if l2_cache_mib is not None:
            command += ['--l2-cache-mib', str(l2_cache_mib)]
        if backup_session is not None:
            command += ['--checkpoint-directory', str(backup_session),
                        '--checkpoint-source-sha256', SOURCE_SHA256]
        status['command'] = command
        timeout = (max_wall_seconds * (2 if include_repeat else 1) + 900 + (900 if include_layer else 0)
                   if action == 'experiment' else max_wall_seconds + 900)
        status['worker_timeout_seconds'] = timeout
        status['status'] = 'worker_running'
        write_status(workspace / 'launcher_status.json', status)
        status['worker'] = worker(command, source=source, log_path=workspace / 'worker.log', timeout=timeout)
        completed = (status['worker']['returncode'] == 0 and not status['worker']['timed_out']
                     and not status['worker']['interrupted'])
        status['status'] = 'worker_completed' if completed else 'worker_failed_evidence_retained'
    except BaseException as exc:
        status.update(status='launcher_failed_evidence_retained', error_type=type(exc).__name__, error=str(exc))
    finally:
        write_status(workspace / 'launcher_status.json', status)
        result_zip = export_results(workspace, archive)
    result = {'status': status['status'], 'archive': result_zip, 'workspace': str(workspace),
              'source_sha256': SOURCE_SHA256, 'simulate': bool(simulate),
              'result_archive_sha256': hashlib.sha256(Path(result_zip).read_bytes()).hexdigest(),
              'backup_status': 'not_requested'}
    if backup_session is not None:
        try:
            published = checkpoint_module(source).publish_verified_file(result_zip, backup_session)
            result.update(backup_status='verified', backup_archive=published['path'],
                          backup_sha256=published['sha256'])
            print('Verified final archive backup:', published['path'], flush=True)
        except Exception as exc:
            result.update(backup_status='failed', backup_error_type=type(exc).__name__, backup_error=str(exc))
            print('Final backup failed. Download the retained local archive now:', result_zip, flush=True)
            print(type(exc).__name__ + ': ' + str(exc), flush=True)
    report = workspace / 'run' / ('report.html' if action == 'experiment' else 'analysis/report.html')
    if report.is_file():
        result['report'] = str(report)
    if action == 'experiment' and (workspace / 'run/summary.json').is_file():
        summary = json.loads((workspace / 'run/summary.json').read_text(encoding='utf-8'))
        result['experiment_status'] = summary.get('status')
        result['primary_evaluation_available'] = summary.get('primary_evaluation_available', False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Download the complete ZIP, including partial results if the worker failed.', flush=True)
    return result


def run_experiment(*, experiment_id, dtypes=('float16', 'bfloat16'), profile='standard',
                   simulate=False, include_repeat=True, include_layer=False, skip_profiles=False,
                   max_wall_seconds=7200, l2_cache_mib=None, pin_analysis_deps=True, base_directory=None,
                   backup_directory=None):
    """One fresh dataset experiment and one complete downloadable result archive."""
    return run(session_id=experiment_id, dtypes=dtypes, action='experiment', profile=profile,
               simulate=simulate, include_repeat=include_repeat, include_layer=include_layer,
               skip_profiles=skip_profiles, max_wall_seconds=max_wall_seconds,
               l2_cache_mib=l2_cache_mib, pin_analysis_deps=pin_analysis_deps, base_directory=base_directory,
               backup_directory=backup_directory)


def review(uploaded, *, review_mode='analyze', result_order=None,
           review_id='review_01', pin_analysis_deps=True, base_directory=None):
    """Analyze existing data or compare a source run with one repeat/GPU run."""
    if review_mode not in ('analyze', 'repeat', 'gpu'):
        raise ValueError('Review mode must be analyze, repeat or gpu')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', review_id):
        raise ValueError('Invalid review ID')
    expected = 1 if review_mode == 'analyze' else 2
    order = list(result_order or sorted(uploaded))
    if len(order) != expected or len(set(order)) != expected or set(order) != set(uploaded):
        raise ValueError(f'{review_mode} requires exactly {expected} distinct result ZIPs; RESULT_ORDER must cover them all')
    print('Input order (first is the source for comparisons): ' + ', '.join(order), flush=True)
    base = Path(base_directory or Path.cwd()).resolve()
    workspace = base / ('linear-perf_' + review_id + '_workspace')
    archive_path = base / ('linear-perf_' + review_id + '_REVIEW.zip')
    if archive_path.exists():
        raise ValueError('Review ZIP already exists; refusing overwrite')
    source = prepare_source(workspace)
    status = {'version': VERSION, 'status': 'review_preparing', 'review_mode': review_mode,
              'result_order': order, 'inputs': [], 'source_sha256': SOURCE_SHA256,
              'gpu_executed_by_review': False, 'scope': 'Offline analysis of previously measured data'}
    try:
        inputs = []
        for index, name in enumerate(order):
            data = bytes(uploaded[name])
            if len(data) > 400_000_000:
                raise ValueError('Input ZIP exceeds the 400 MB compressed-size limit')
            folder = safe_extract(data, workspace / 'inputs' / f'input_{index + 1:02d}', limit=700_000_000)
            manifest_path = folder / 'export_manifest.json'
            export = json.loads(manifest_path.read_text(encoding='utf-8'))
            actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                            if p.is_file() and p != manifest_path}
            if actual_files != set(export['files']):
                raise ValueError('Result ZIP inventory differs from its export manifest')
            for member, digest in export['files'].items():
                item = folder.joinpath(*PurePosixPath(member).parts)
                if not item.resolve().is_relative_to(folder) or hashlib.sha256(item.read_bytes()).hexdigest() != digest:
                    raise ValueError('Input ZIP checksum mismatch: ' + member)
            # Validate each input against its own recorded source. Compatible data
            # may be analyzed with later code; input source is never executed.
            source_digest = hashlib.sha256((folder / 'source_archive.zip').read_bytes()).hexdigest()
            if source_digest != export.get('source_sha256'):
                raise ValueError('Input source archive checksum differs from its recorded source')
            source_sidecar = (folder / 'source_archive.sha256').read_text(encoding='ascii').split()
            if not source_sidecar or source_sidecar[0] != source_digest:
                raise ValueError('Input source checksum sidecar is inconsistent')
            launch = json.loads((folder / 'launcher_status.json').read_text(encoding='utf-8'))
            primary_status_file = folder / 'run/primary/dataset/status.json'
            completed_primary = False
            if launch.get('action') == 'experiment' and primary_status_file.is_file():
                primary_status = json.loads(primary_status_file.read_text(encoding='utf-8'))
                completed_primary = primary_status.get('status') in ('completed', 'simulated_complete')
            primary_recovery = launch.get('status') != 'worker_completed' and completed_primary
            if ((launch.get('status') != 'worker_completed' and not primary_recovery)
                    or launch.get('action') not in ('run', 'collect', 'experiment')
                    or launch.get('source_sha256') != source_digest):
                raise ValueError('Only completed, internally consistent datasets can enter analysis')
            dataset_run = folder / 'run' / 'primary' if launch.get('action') == 'experiment' else folder / 'run'
            inputs.append(dataset_run)
            status['inputs'].append({'uploaded_filename': name, 'zip_sha256': hashlib.sha256(data).hexdigest(),
                                     'source_sha256': source_digest, 'run_directory': str(dataset_run),
                                     'full_experiment_primary_selected': launch.get('action') == 'experiment',
                                     'original_launcher_status': launch.get('status'),
                                     'primary_recovered_from_incomplete_experiment': primary_recovery})
        status['dependencies'] = ensure_dependencies(source, pin=pin_analysis_deps)
        command = [sys.executable, '-u', '-m', 'linear_perf']
        if review_mode == 'analyze':
            command += ['analyze', '--dataset', str(inputs[0])]
        else:
            command += ['compare', '--source', str(inputs[0]), '--target', str(inputs[1]), '--mode', review_mode]
        command += ['--out', str(workspace / 'review')]
        status['command'] = command
        status['worker'] = worker(command, source=source, log_path=workspace / 'worker.log', timeout=900)
        completed = (status['worker']['returncode'] == 0 and not status['worker']['timed_out']
                     and not status['worker']['interrupted'])
        status['status'] = 'worker_completed' if completed else 'review_failed_evidence_retained'
    except BaseException as exc:
        status.update(status='review_failed_evidence_retained', error_type=type(exc).__name__, error=str(exc))
    finally:
        write_status(workspace / 'launcher_status.json', status)
        archive = export_results(workspace, archive_path, extra_directories=('review',))
    result = {'status': status['status'], 'archive': archive, 'workspace': str(workspace),
              'source_sha256': SOURCE_SHA256, 'gpu_executed_by_review': False,
              'result_archive_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest()}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Keep the original input ZIPs with this derived review.', flush=True)
    return result


def layer(source_result, *, model_id='qwen2_5_0_5b', dtype='float16', batch_size=1,
          sequence_length=128, pin_analysis_deps=True):
    """Run an optional random-weight layer check in the source GPU runtime."""
    workspace = Path(source_result['workspace']).resolve()
    source = workspace / 'source' / f'transformer-linear-layer-performance-predictor-{VERSION}'
    if (not source.is_dir() or source_result.get('simulate')
            or source_result.get('status') != 'worker_completed'):
        raise ValueError('Layer validation requires a completed measured source run in this runtime')
    layer_output = workspace / 'layer'
    if layer_output.exists():
        raise ValueError('Layer output already exists; preserve it')
    source_run = workspace / 'run/primary' if (workspace / 'run/primary/dataset').is_dir() else workspace / 'run'
    command = [sys.executable, '-u', '-m', 'linear_perf', 'layer', '--source', str(source_run),
               '--model-id', model_id, '--dtype', dtype, '--batch-size', str(batch_size),
               '--sequence-length', str(sequence_length), '--out', str(layer_output)]
    archive = workspace.parent / (workspace.name.removesuffix('_workspace') + '_LAYER.zip')
    if archive.exists():
        raise ValueError('Layer archive already exists; preserve it')
    status = {'status': 'preparing', 'command': command, 'source_sha256': SOURCE_SHA256}
    try:
        status['dependencies'] = ensure_dependencies(source, pin=pin_analysis_deps)
        result = worker(command, source=source, log_path=workspace / 'layer_worker.log', timeout=900)
        status['worker'] = result
        status['status'] = ('worker_completed' if result['returncode'] == 0 and
                            not result['timed_out'] and not result['interrupted'] else 'failed_evidence_retained')
    except BaseException as exc:
        status.update(status='failed_evidence_retained', error_type=type(exc).__name__, error=str(exc))
    finally:
        layer_output.mkdir(exist_ok=True)
        write_status(layer_output / 'layer_launcher_status.json', status)
        if (workspace / 'layer_worker.log').is_file():
            (layer_output / 'layer_worker.log').write_bytes((workspace / 'layer_worker.log').read_bytes())
        path = export_results(workspace, archive, extra_directories=('layer',))
    return {'archive': path, 'status': status['status']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--session-id', default='paired_s01')
    parser.add_argument('--dtypes', nargs='+', choices=('float16', 'bfloat16'), default=['float16', 'bfloat16'])
    parser.add_argument('--action', choices=('run', 'collect', 'experiment'), default='experiment')
    parser.add_argument('--profile', choices=('standard', 'smoke'), default='standard')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--repeat-subset', action='store_true')
    parser.add_argument('--skip-profiles', action='store_true')
    parser.add_argument('--skip-repeat', action='store_true')
    parser.add_argument('--include-layer', action='store_true')
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    parser.add_argument('--keep-analysis-versions', action='store_true')
    parser.add_argument('--base-directory')
    parser.add_argument('--backup-directory', help='Parent directory for verified per-session checkpoint and final archive copies')
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps(self_check(), indent=2)); return 0
    result = run(session_id=args.session_id, dtypes=args.dtypes, action=args.action,
                 profile=args.profile, simulate=args.simulate, repeat_subset=args.repeat_subset,
                 skip_profiles=args.skip_profiles, max_wall_seconds=args.max_wall_seconds,
                 include_repeat=not args.skip_repeat, include_layer=args.include_layer,
                 pin_analysis_deps=not args.keep_analysis_versions, base_directory=args.base_directory,
                 backup_directory=args.backup_directory)
    return 0 if result['status'] == 'worker_completed' and result.get('backup_status') != 'failed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
