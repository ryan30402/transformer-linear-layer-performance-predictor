"""Human-readable, offline evidence report. Rendering never fits a model."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

from .common import LabError, read_json, write_json


def _text(value):
    return html.escape(str(value), quote=True)


def _number(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + _text(x) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + _text(x) + '</td>' for x in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def _details(title, value):
    return '<details><summary>' + _text(title) + '</summary><pre>' + _text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    ) + '</pre></details>'


def _image(figure, path, *, simulated):
    if simulated:
        figure.suptitle('SIMULATED — SOFTWARE CHECK ONLY — NO GPU USED',
                        fontsize=10, color='#a33a16', weight='bold')
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    path.write_bytes(buffer.getvalue())
    return '<img alt="' + _text(path.stem) + '" src="data:image/png;base64,' + base64.b64encode(
        buffer.getvalue()).decode('ascii') + '">'


def render_session_report(run_dir, destination=None):
    """Render a new report, preserving an already sealed run when rerendering."""
    run = Path(run_dir).resolve()
    out = Path(destination).resolve() if destination is not None else run
    if out == run and (run / 'manifest.json').exists():
        raise LabError('A sealed run is immutable; render to a separate new directory')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('report.html', 'report_summary.json', 'group_metrics.csv'):
        if (out / name).exists():
            raise LabError('Refusing to overwrite an existing report artifact: ' + name)
    results = read_json(run / 'results.json')
    status = read_json(run / 'status.json')
    environment = read_json(run / 'environment.json')
    plan = read_json(run / 'plan.json')
    suite = read_json(run / 'model_suite.json')
    quality = read_json(run / 'quality.json')
    acquisition = read_json(run / 'acquisition_results.json')
    diagnostics = read_json(run / 'diagnostics_summary.json')
    controls = read_json(run / 'controls.json')
    simulated = results['measurement_kind'] == 'simulated'
    selected = results['selected_model_id']
    figure_dir = out / 'figures'
    if figure_dir.exists():
        raise LabError('Refusing to overwrite an existing figures directory')
    figure_dir.mkdir()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})

    sections = []
    badge = 'SIMULATED · NO GPU USED · SOFTWARE VALIDATION ONLY' if simulated else 'CUDA MEASURED · ONE SESSION · REPLICATION REQUIRED'
    sections.append('<div class="badge">' + badge + '</div><h1>Transformer Linear Layer Performance Predictor</h1>'
                    '<p class="subtitle">1.0.0 · ' + _text(results['session_id']) + ' · ' + _text(plan['dtype']) + '</p>')
    sections.append('<p>Prediction target: preallocated <code>Y = X @ W.T</code> tensors under warm-reuse CUDA Graph execution. '
                    'CUDA-event graph replay elapsed time is divided by 50 operations. Each visit contains nine replay samples; '
                    'the final label is the median of three block medians. The interval includes graph execution gaps and internal GPU memory accesses. '
                    'It excludes allocation, CPU-to-GPU transfers, graph capture, bias, activation, serving queues and full-model inference.</p>')
    device = environment.get('identity', {}).get('device', {})
    sections.append(_table(['Field', 'Recorded value'], [
        ['GPU', device.get('name', 'unknown')], ['GPU UUID', device.get('uuid', 'unavailable')],
        ['Environment identity', environment.get('environment_id')], ['Status', status['status']],
        ['Model selection', selected + ' (development median APE; declared tie-breakers)'],
        ['Timing quality criteria', 'All passed' if quality['passed'] else 'Qualifications recorded; all observations retained'],
        ['Data', '24 calibration / 12 development / 24 test / 12 challenge / 24 candidate / 8 diagnostic'],
    ]))
    sections.append('<h2>1. Decisions fixed before held-out evaluation</h2><p>Cases, groups, timing, quality thresholds, '
                    'candidate models and acquisition budgets / seeds are registered in advance. Calibration and development data determine models '
                    'and hyperparameters. Model artifacts, acquisition checkpoints and concrete held-out predictions are saved before held-out labels '
                    'are collected. Evaluation never reselects the winner, removes slow samples or automatically reruns a session.</p>')
    sections.append(_details('Development-set model selection record', {
        'selected_model_id': selected, 'selection_rule': suite['selection_rule'],
        'development_scores': {k: {x: v[x] for x in ('median_ape_pct', 'p90_ape_pct', 'rmse_ms')}
                               for k, v in suite['development_metrics'].items()},
        'hyperparameter_search': suite['hyperparameter_search'],
    }))

    headline = {}
    group_export = []
    for split in ('test', 'challenge'):
        evaluation = results['evaluations'][split]
        names = suite['model_order']
        rows = []
        for model_id in names:
            metrics = evaluation['models'][model_id]
            rows.append([model_id + (' [development-selected]' if model_id == selected else ''), metrics['n'],
                         _number(metrics['median_ape_pct']), _number(metrics['p90_ape_pct']),
                         _number(metrics['max_ape_pct']), _number(metrics['rmse_us']),
                         _number(metrics['signed_bias_us']), _number(metrics['pairwise_ordering_accuracy'])])
        title = 'new M values in known families, including extrapolation' if split == 'test' else 'N/K families absent from calibration, development and acquisition'
        sections.append('<h2>2' + ('a' if split == 'test' else 'b') + '. ' + split + ': ' + title + '</h2>')
        sections.append(_table(['Model', 'Cases', 'Median APE %', 'P90 APE %', 'Worst APE %',
                                'RMSE μs', 'Bias μs', 'Pairwise ordering accuracy'], rows))
        comparison = evaluation['selected_vs_roofline_3p']
        delta = comparison['observed_difference_pp']
        headline[split] = {'selected_model': selected, 'median_ape_pct': evaluation['models'][selected]['median_ape_pct'],
                           'difference_to_same_data_roofline_3p_pp': delta,
                           'descriptive_family_bootstrap': comparison,
                           'gpu_accuracy_claim_permitted_from_simulation': False,
                           'generalization_established': False}
        sections.append('<p>The selected model differs from the three-parameter Roofline baseline by <b>' + _number(delta) +
                        ' percentage points</b> in median APE on the same cases (negative is lower). Every candidate is reported; '
                        'held-out scores do not change the development-selected model. RMSE emphasizes large absolute timing errors, '
                        'while APE measures relative errors, including errors on short operations. Pairwise ordering excludes observed near-ties '
                        'within 1%; this does not remove all measurement uncertainty.</p>')
        sections.append(_details('Paired family bootstrap: descriptive uncertainty and limitations', comparison))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        predicted = evaluation['models'][selected]['predictions']
        actual = np.array([x['actual_ms'] for x in predicted]) * 1000
        prediction = np.array([x['predicted_ms'] for x in predicted]) * 1000
        axes[0].scatter(actual, prediction, c='#137e75', edgecolor='white', s=48)
        bounds = [min(actual.min(), prediction.min()) * .75, max(actual.max(), prediction.max()) * 1.3]
        axes[0].plot(bounds, bounds, '--', color='#637084', label='Perfect prediction')
        axes[0].set(xscale='log', yscale='log', xlim=bounds, ylim=bounds,
                    xlabel='Observed time (microseconds)', ylabel='Predicted time (microseconds)',
                    title=split + ': development-selected model')
        axes[0].legend(fontsize=8)
        for name in names:
            ape = sorted(x['ape_pct'] for x in evaluation['models'][name]['predictions'])
            axes[1].step(ape, np.arange(1, len(ape) + 1) / len(ape), where='post', label=name,
                         linewidth=2.7 if name == selected else 1.2)
        axes[1].set(xlabel='Absolute percentage error (%)', ylabel='Fraction of distinct cases',
                    title='Error distribution (all cases retained)', ylim=(0, 1.02))
        axes[1].legend(fontsize=7)
        sections.append(_image(fig, figure_dir / (split + '_predictions.png'), simulated=simulated))
        plt.close(fig)
        sections.append('<h3>Group results for the selected model</h3>')
        group_rows = []
        for dimension, values in evaluation['groups'][selected].items():
            if dimension == 'role':
                continue
            for label, metrics in sorted(values.items()):
                group_rows.append([dimension, label, metrics['n'], _number(metrics['median_ape_pct']),
                                   _number(metrics['p90_ape_pct']), _number(metrics['rmse_us'])])
        sections.append(_table(['Dimension', 'Group', 'n', 'Median APE %', 'P90 APE %', 'RMSE μs'], group_rows))
        for model_id, dimensions in evaluation['groups'].items():
            for dimension, values in dimensions.items():
                for label, metrics in values.items():
                    group_export.append({'split': split, 'model': model_id, 'dimension': dimension,
                                         'group': label, 'n': metrics['n'],
                                         'median_ape_pct': metrics['median_ape_pct'], 'p90_ape_pct': metrics['p90_ape_pct'],
                                         'rmse_us': metrics['rmse_us']})
        worst = sorted(predicted, key=lambda x: x['ape_pct'], reverse=True)[:8]
        sections.append(_details('Eight largest errors, with coverage flags and observed block ranges', worst))

    sections.append('<h2>3. Timing quality and session drift</h2><p>The 10% CV, 1.2 block ratio and 1.2 control ratio '
                    'are predeclared engineering qualification thresholds, not industry certification or confidence intervals. '
                    'Quality warnings never remove observations. Event repeats, collection blocks, random seeds and separate sessions '
                    'describe different levels of variation.</p>')
    sections.append(_details('All quality criteria, cases and qualifications', quality))
    if controls:
        fig, ax = plt.subplots(figsize=(10.5, 4.1), constrained_layout=True)
        for cid in sorted({r['case_id'] for r in controls}):
            rows = [r for r in controls if r['case_id'] == cid]
            times = [r.get('visit_started_monotonic_seconds', i) for i, r in enumerate(rows)]
            start = min(r.get('visit_started_monotonic_seconds', 0) for r in controls)
            vals = [r['median_ms'] / rows[0]['median_ms'] for r in rows]
            ax.plot((np.array(times) - start) / 60, vals, '.-', label=cid)
        ax.axhline(1, color='#8791a1', linestyle='--')
        ax.set(xlabel='Minutes since first control visit', ylabel='Time / first observed control time',
               title='Controls across collection stages (raw visit medians)')
        ax.legend()
        sections.append(_image(fig, figure_dir / 'control_drift.png', simulated=simulated))
        plt.close(fig)
    sections.append('<p>GPU state is sampled at visit boundaries, no more than once per second, to keep nvidia-smi calls '
                    'outside primary event intervals. These coarse observations cannot identify clocks inside each event or prove the absence '
                    'of throttling. Unlocked clocks remain an experimental limitation.</p>')

    sections.append('<h2>4. Acquisition policies: ten seeds and four label budgets</h2><p>Heuristic and random selection '
                    'use the same two-parameter Roofline model to separate selection effects from model-family differences. All 24 candidate '
                    'cases were measured in advance; selectors reveal labels incrementally. The plotted budgets count additional labels '
                    'visible to a policy, not GPU measurement costs saved during this experiment. Every 2 / 4 / 8 / 12-label checkpoint '
                    'is frozen before held-out evaluation.</p>')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, split in zip(axes, ('test', 'challenge')):
        curves = acquisition[split]['curves']
        for policy, color in (('heuristic', '#137e75'), ('random', '#bd672d')):
            values = sorted([x for x in curves if x['policy'] == policy and x['split'] == split], key=lambda x: x['budget'])
            x = [v['budget'] for v in values]
            ax.plot(x, [v['median_median_ape_pct'] for v in values], 'o-', color=color, label=policy)
            ax.fill_between(x, [v['min_median_ape_pct'] for v in values],
                            [v['max_median_ape_pct'] for v in values], color=color, alpha=.14)
        baseline = acquisition[split]['baseline'][split]['median_ape_pct']
        ax.axhline(baseline, linestyle='--', color='#637084', label='Initial 2p baseline')
        ax.set(xlabel='Newly revealed distinct labels', ylabel='Heldout median APE (%)', title=split)
        ax.set_xticks([2, 4, 8, 12]); ax.legend(fontsize=8)
    sections.append(_image(fig, figure_dir / 'acquisition_curves.png', simulated=simulated)); plt.close(fig)
    sections.append('<p>Lines show medians across seeds; shaded bands show minimum to maximum values, not confidence intervals. '
                    'Repeated identical heuristic sequences are not independent evidence. The best seed or budget observed after evaluation '
                    'cannot be substituted for a policy selected before evaluation.</p>')

    sections.append('<h2>5. Diagnostics, numerical witnesses and hardware evidence</h2><p>Eager / graph timing, '
                    '25 / 50 / 100-operation sensitivity, one / four-buffer rotation and graph profiling are recorded separately. '
                    'They never replace primary model labels. Buffer rotation does not flush the cache; without hardware counters, '
                    'these records do not measure HBM traffic or L2 hit rates.</p>')
    sections.append(_details('Diagnostic summary and profiler availability', {
        key: diagnostics.get(key) for key in ('status', 'sensitivity', 'profiles', 'rotation_records', 'numerical_witnesses',
                                             'main_timing_policy_changed_from_diagnostic_results')}))
    sections.append('<p>Small numerical witnesses preserve the represented low-precision inputs, GPU outputs and FP32 references. '
                    'A CPU audit recomputes them with NumPy float64. The four witnesses do not preserve every large primary matrix output '
                    'and are not independent hardware attestation. Witness collection is explicitly skipped in simulation.</p>')
    sections.append(_details('Fixed-workload batching, separate from model evaluation', read_json(run / 'batching.json')))

    sections.append('<h2>6. Interpretation and limits</h2><p>' +
                    ('This report uses synthetic data to validate software behavior. None of its chart values measures NVIDIA GPU performance '
                     'or establishes prediction accuracy on hardware. '
                     if simulated else 'This report describes held-out cases from one session. Errors, variability and model differences can be '
                     'reported for this session; they do not establish transfer to other GPUs, software stacks, precisions, cache states or inference services. ') +
                    'After the three declared sessions are complete, compare both per-session fits and frozen transfer from the first session. '
                    'GPU and precision comparisons use explicit comparison modes. Environment hashes and manifests check internal consistency, '
                    'not hardware attestation. The methods reference public measurement guidance without claiming TensorRT, NVBench or MLPerf certification.</p>')
    sections.append('<h3>Reproduction and audit</h3><p><code>python -m linear_perf audit --run PATH</code> '
                    'checks raw evidence, frozen predictions, chronology and results. JSONL records, controls, telemetry, numerical witnesses, '
                    'graph traces, model coefficients and predictions.csv are retained in the session package. Preserve existing evidence '
                    'and register a new protocol and unseen holdout when changing the experimental design.</p>')
    sections.append('<h3>Primary method references</h3><ul>' + ''.join(
        '<li><a href="' + _text(url) + '">' + _text(name) + '</a></li>' for name, url in (
            ('PyTorch CUDA semantics', 'https://docs.pytorch.org/docs/2.11/notes/cuda.html'),
            ('NVIDIA TensorRT performance benchmarking', 'https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html'),
            ('NVIDIA NVBench measurement scope', 'https://github.com/NVIDIA/nvbench/blob/main/README.md'),
            ('NVIDIA matrix multiplication guide', 'https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html'),
        )) + '</ul>')
    summary = {'protocol_version': '1.0.0', 'measurement_kind': results['measurement_kind'],
               'session_id': results['session_id'], 'selected_model_id': selected, 'heldout': headline,
               'timing_quality_passed': quality['passed'], 'hardware_accuracy_validated': False,
               'cross_session_validated': False, 'industry_certification': False,
               'source_result_directory': str(run), 'report_does_not_refit_or_select_models': True}
    write_json(out / 'report_summary.json', summary, overwrite=False)
    with (out / 'group_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_export[0]))
        writer.writeheader(); writer.writerows(group_export)
    css = '''body{font:16px/1.7 system-ui,-apple-system,sans-serif;color:#17283b;background:#f3f6f9;margin:0}
main{max-width:1150px;margin:auto;padding:45px 35px 70px;background:#fff}h1{font-size:36px;line-height:1.25;margin:18px 0 8px}
h2{font-size:25px;color:#124e59;border-top:1px solid #d9e2ea;padding-top:32px;margin-top:40px}h3{font-size:19px}
.badge{display:inline-block;padding:7px 12px;background:#fff0df;color:#874013;border-radius:5px;font-weight:700}.subtitle{color:#53677e}
table{border-collapse:collapse;min-width:650px;width:100%;font-size:14px}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dfe7ee;vertical-align:top}
th{background:#eaf1f7;color:#263f58}tr:nth-child(even){background:#f8fafc}.scroll{overflow:auto;margin:20px 0}
img{display:block;max-width:100%;height:auto;margin:24px auto}code{font-size:13px;background:#edf2f6;padding:3px 6px;border-radius:3px}
pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:16px}details{margin:18px 0;border:1px solid #d9e2ea;padding:10px 15px;border-radius:5px}summary{cursor:pointer;color:#254d68}a{color:#126887}
@media(max-width:650px){main{padding:24px 16px}h1{font-size:28px}}@media print{body{background:white}main{padding:0}details{break-inside:avoid}h2{break-after:avoid}img{break-inside:avoid}}'''
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + \
               '<title>Transformer Linear Layer Performance Predictor — ' + _text(results['session_id']) + '</title><style>' + css + '</style></head><body><main>' + \
               '\n'.join(sections) + '</main></body></html>'
    (out / 'report.html').write_text(document, encoding='utf-8')
    return summary
