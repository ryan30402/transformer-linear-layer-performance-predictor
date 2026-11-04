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

VERSION = '1.0.0'
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
        root = 'transformer-linear-layer-performance-predictor-1.0.0/'
        for dtype in ('fp16', 'bf16'):
            plan = json.loads(archive.read(root + 'configs/' + dtype + '_graph.json'))
            if plan.get('protocol_version') != VERSION or len(plan['cases']) != 104:
                raise ValueError('Unexpected embedded experiment plan')
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
    source = root / 'source' / 'transformer-linear-layer-performance-predictor-1.0.0'
    if not (source / 'linear_perf' / 'campaign.py').is_file():
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
    # Failure envelopes are siblings of the sealed run, and must travel with it.
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


def run(*, session_id='fp16_s01', dtype='float16', action='run', simulate=False,
        pin_analysis_deps=True, base_directory=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', session_id):
        raise ValueError('Invalid session ID')
    if dtype not in ('float16', 'bfloat16') or action not in ('run', 'diagnose'):
        raise ValueError('Invalid dtype/action')
    base = Path(base_directory or Path.cwd()).resolve()
    workspace = base / ('linear-perf_' + session_id + '_workspace')
    archive = base / ('linear-perf_' + session_id + '_RESULTS.zip')
    if archive.exists():
        raise ValueError('Preserve the existing result ZIP; choose a new registered session ID')
    source = prepare_source(workspace)
    status = {'version': VERSION, 'session_id': session_id, 'action': action, 'dtype': dtype,
              'simulate': bool(simulate), 'source_sha256': SOURCE_SHA256, 'status': 'preparing',
              'python_version': sys.version, 'automatic_retries': 0,
              'hardware_accuracy_validated': False, 'gpu_allocation_performed': False}
    write_status(workspace / 'launcher_status.json', status)
    try:
        status['dependencies'] = ensure_dependencies(source, pin=pin_analysis_deps)
        config = 'bf16_graph.json' if dtype == 'bfloat16' else 'fp16_graph.json'
        command = [sys.executable, '-u', '-m', 'linear_perf.campaign', action,
                   '--config', str(source / 'configs' / config), '--out', str(workspace / 'run'),
                   '--session-id', session_id]
        if simulate:
            command.append('--simulate')
        status['command'] = command
        status['worker'] = worker(command, source=source, log_path=workspace / 'worker.log')
        completed = status['worker']['returncode'] == 0 and not status['worker']['timed_out'] and not status['worker']['interrupted']
        status['status'] = 'worker_completed' if completed else 'worker_failed_evidence_retained'
    except BaseException as exc:
        status.update(status='launcher_failed_evidence_retained', error_type=type(exc).__name__, error=str(exc))
    finally:
        write_status(workspace / 'launcher_status.json', status)
        result_zip = export_results(workspace, archive)
    result = {'status': status['status'], 'archive': result_zip, 'workspace': str(workspace),
              'source_sha256': SOURCE_SHA256, 'simulate': bool(simulate),
              'result_archive_sha256': hashlib.sha256(Path(result_zip).read_bytes()).hexdigest()}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Download the full ZIP, including failures. No automatic retry was performed.', flush=True)
    return result


def review(uploaded, *, review_mode='study', transfer_mode='gpu', result_order=None,
           review_id='review_01', pin_analysis_deps=True, base_directory=None):
    """Audit known-release result ZIPs and compare them with a CPU-only worker."""
    if review_mode not in ('study', 'transfer', 'compare-dtypes') or transfer_mode not in ('session', 'gpu'):
        raise ValueError('Invalid review mode')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', review_id):
        raise ValueError('Invalid review ID')
    expected = 3 if review_mode == 'study' else 2
    order = list(result_order or sorted(uploaded))
    if len(order) != expected or len(set(order)) != expected or set(order) != set(uploaded):
        raise ValueError(f'{review_mode} requires exactly {expected} distinct result ZIPs; RESULT_ORDER must cover them all')
    print('Fixed input order (first is transfer source): ' + ', '.join(order), flush=True)
    base = Path(base_directory or Path.cwd()).resolve()
    workspace = base / ('linear-perf_' + review_id + '_workspace')
    archive_path = base / ('linear-perf_' + review_id + '_REVIEW.zip')
    if archive_path.exists():
        raise ValueError('Review ZIP already exists; refusing overwrite')
    source = prepare_source(workspace)
    status = {'version': VERSION, 'status': 'review_preparing', 'review_mode': review_mode,
              'transfer_mode': transfer_mode, 'result_order': order, 'inputs': [],
              'source_sha256': SOURCE_SHA256, 'gpu_executed_by_review': False,
              'scope': 'Offline analysis of already sealed sessions; no new measurements'}
    try:
        inputs = []
        for index, name in enumerate(order):
            data = bytes(uploaded[name])
            if len(data) > 400_000_000:
                raise ValueError('Input ZIP exceeds the 400 MB compressed-size limit')
            folder = safe_extract(data, workspace / 'inputs' / f'input_{index + 1:02d}', limit=700_000_000)
            manifest_path = folder / 'export_manifest.json'
            export = json.loads(manifest_path.read_text(encoding='utf-8'))
            if export.get('source_sha256') != SOURCE_SHA256:
                raise ValueError('Result uses a different release source; do not mix code versions')
            actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob('*')
                            if p.is_file() and p != manifest_path}
            if actual_files != set(export['files']):
                raise ValueError('Result ZIP inventory differs from its export manifest')
            for member, digest in export['files'].items():
                item = folder.joinpath(*PurePosixPath(member).parts)
                if not item.resolve().is_relative_to(folder) or hashlib.sha256(item.read_bytes()).hexdigest() != digest:
                    raise ValueError('Input ZIP checksum mismatch: ' + member)
            if hashlib.sha256((folder / 'source_archive.zip').read_bytes()).hexdigest() != SOURCE_SHA256:
                raise ValueError('Input source archive differs from this release')
            launch = json.loads((folder / 'launcher_status.json').read_text(encoding='utf-8'))
            if launch.get('status') != 'worker_completed' or launch.get('action') != 'run':
                raise ValueError('Only completed primary sessions can enter comparison; retain failed/diagnostic evidence separately')
            inputs.append(folder / 'run')
            status['inputs'].append({'uploaded_filename': name, 'zip_sha256': hashlib.sha256(data).hexdigest(),
                                     'run_directory': str(folder / 'run')})
        status['dependencies'] = ensure_dependencies(source, pin=pin_analysis_deps)
        command = [sys.executable, '-u', '-m', 'linear_perf.campaign', review_mode]
        if review_mode == 'transfer':
            command += ['--source', str(inputs[0]), '--target', str(inputs[1]), '--mode', transfer_mode]
        else:
            command += ['--runs'] + [str(path) for path in inputs]
        command += ['--out', str(workspace / 'review')]
        status['command'] = command
        status['worker'] = worker(command, source=source, log_path=workspace / 'worker.log', timeout=600)
        completed = status['worker']['returncode'] == 0 and not status['worker']['timed_out'] and not status['worker']['interrupted']
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
    print('Keep the original input ZIPs with this derived review. No source run was modified.', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--session-id', default='fp16_s01')
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='float16')
    parser.add_argument('--action', choices=('run', 'diagnose'), default='run')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--keep-analysis-versions', action='store_true')
    parser.add_argument('--base-directory')
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps(self_check(), indent=2)); return 0
    result = run(session_id=args.session_id, dtype=args.dtype, action=args.action, simulate=args.simulate,
                 pin_analysis_deps=not args.keep_analysis_versions, base_directory=args.base_directory)
    return 0 if result['status'] == 'worker_completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
