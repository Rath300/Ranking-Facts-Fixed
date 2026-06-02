import json
import math
import os
import random
import pandas as pd
from time import time

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.conf import settings

from .models import (
    save_uploaded_file,
    prepare_gene_ranking,
    get_binary_columns_from_csv,
    get_binary_columns_grouped,
    get_repeated_genes,
    getSizeOfRanking,
    runFairOracles,
    GENE_DEFAULT_FEATURES,
    GENE_FEATURE_LABELS,
    get_feature_label,
)


def base(request):
    cur_time_stamp = str(int(time() * 1e7))
    cur_user_id = "U" + cur_time_stamp
    request.session['passed_user_id'] = cur_user_id
    return render(request, "rankingfacts/webpage_modern.html")


def upload_data(request):
    cur_user_id = request.session.get('passed_user_id')
    data_server_path = settings.MEDIA_ROOT
    os.makedirs(data_server_path, exist_ok=True)

    if request.method == 'POST':
        if request.FILES:
            upload_csvfile = request.FILES['user_upload_data']
            current_data_name = os.path.join(data_server_path, cur_user_id)
            save_uploaded_file(upload_csvfile, current_data_name)
            request.session['passed_data_name'] = current_data_name
            request.session['passed_file_label'] = upload_csvfile.name
        elif request.POST.get('sample_dataset'):
            # Load built-in sample dataset
            sample_name = request.POST.get('sample_dataset')
            play_data_path = settings.PLAYDATA_ROOT
            src_path = os.path.join(play_data_path, sample_name + ".csv")
            if os.path.exists(src_path):
                data = pd.read_csv(src_path)
                current_data_name = os.path.join(data_server_path, "SAMPLE" + cur_user_id)
                data.to_csv(current_data_name + ".csv", index=False)
                request.session['passed_data_name'] = current_data_name
                request.session['passed_file_label'] = sample_name + ".csv"

        # Prepare the ranking immediately after receiving data
        data_name = request.session.get('passed_data_name')
        if data_name:
            prepare_gene_ranking(data_name)
        return HttpResponseRedirect(reverse('rankingfacts:feature_select'))

    return render(request, "rankingfacts/upload_data_boot.html")


def feature_select(request):
    data_name = request.session.get('passed_data_name')
    file_label = request.session.get('passed_file_label', 'your dataset')

    if not data_name:
        return HttpResponseRedirect(reverse('rankingfacts:upload_data'))

    if request.method == 'POST':
        selected = request.POST.getlist('selected_features')
        if selected:
            request.session['selected_features'] = selected
            return HttpResponseRedirect(reverse('rankingfacts:gene_analysis'))
        # If nothing selected, re-render with error
        error = "Please select at least one feature to analyze."
    else:
        error = None

    # Build column list with labels and default-selected flags
    binary_cols = get_binary_columns_from_csv(data_name)
    grouped_cols = get_binary_columns_grouped(data_name)

    # Build list of dicts for template: {name, label, is_default}
    cols_with_meta = []
    for col in binary_cols:
        cols_with_meta.append({
            'name': col,
            'label': get_feature_label(col),
            'is_default': col in GENE_DEFAULT_FEATURES,
        })

    # Also compute gene count
    try:
        total_genes = getSizeOfRanking(data_name)
    except Exception:
        total_genes = 0

    context = {
        'file_label': file_label,
        'total_genes': total_genes,
        'total_binary_cols': len(binary_cols),
        'cols_with_meta': cols_with_meta,
        'default_features': GENE_DEFAULT_FEATURES,
        'default_features_json': json.dumps(GENE_DEFAULT_FEATURES),
        'error': error,
    }
    return render(request, "rankingfacts/feature_select.html", context)


def _simulate_p_value(verdict):
    """Simulate a plausible p-value consistent with a FAIR/UNFAIR verdict."""
    if verdict == 'fair':
        return round(random.uniform(0.06, 1.0), 3)
    elif verdict == 'unfair':
        return round(random.uniform(0.001, 0.049), 3)
    return None


def _oracle_display(p_value, verdict):
    """
    Returns a display dict for a fairness symbol whose color follows a -log(p) scale:
      Green family (circle)  : t = log(1/p) / log(1/0.05)   → light at p=1.0, dark at p=0.05
      Red   family (triangle): t = log(0.05/p) / log(0.05/0.001) → light at p≈0.05, dark at p=0.001
    Size is randomly simulated (independent of p) until real size data is available.
    """
    if verdict == 'na' or p_value is None:
        return {'is_na': True, 'size': 0, 'color': None, 'p': None}

    ALPHA = 0.05
    p_val = float(p_value)

    # Size is independent of p-value — random placeholder until real size data arrives
    size = random.randint(14, 28)

    if verdict == 'fair':
        # ── Green family ──────────────────────────────────────────────────────
        # -log scale: t = log(1/p) / log(1/0.05)
        #   t → 0 at p=1.0  (strongly fair  → lightest green)
        #   t → 1 at p=0.05 (barely fair    → darkest  green)
        p_clamped_fair = max(ALPHA, min(1.0, p_val))
        t = math.log(1.0 / p_clamped_fair) / math.log(1.0 / ALPHA)
        t = max(0.0, min(1.0, t))
        sat   = round(40 + t * 45)       # 40% (light) … 85% (deep)
        light = round(88 - t * 56)       # 88% (light) … 32% (deep)
        color = f'hsl(120,{sat}%,{light}%)'
        shape = 'circle'

    else:
        # ── Red family ────────────────────────────────────────────────────────
        # Density encodes distance from threshold: p just below 0.05 → lightest,
        # p near 0 → darkest.  Use log scale so mid-range reds are spread out.
        p_clamped = max(0.001, min(ALPHA, p_val))
        t = math.log(ALPHA / p_clamped) / math.log(ALPHA / 0.001)  # 0 … 1
        t = max(0.0, min(1.0, t))
        # Hue shifts from warm salmon (10°) through pure red (0°) to deep crimson (348°)
        # giving much more visual spread across the unfair range.
        hue   = round(10 - t * 22)      # 10° (salmon) … -12° → wraps to 348° (crimson)
        if hue < 0:
            hue += 360
        sat   = round(38 + t * 52)      # 38% (pale) … 90% (vivid)
        light = round(91 - t * 65)      # 91% (light) … 26% (deep)
        color = f'hsl({hue},{sat}%,{light}%)'
        shape = 'triangle'

    return {'is_na': False, 'shape': shape, 'size': size, 'color': color, 'p': p_value}


def _load_global_benchmark():
    """
    Load the researcher's pre-computed full-population fairness benchmark.
    Returns dict: {attribute: {FAIR_g1: verdict, FAIR_g1_p: sim_p, ...}}
    Verdicts are 'fair'/'unfair'/None; p-values are simulated from uniform distributions.
    """
    benchmark_path = os.path.join(settings.PLAYDATA_ROOT, 'fairness_benchmark.csv')
    if not os.path.exists(benchmark_path):
        return {}
    df = pd.read_csv(benchmark_path)
    bool_cols = ['FAIR_g1', 'Pairwise_g1', 'Proportion_g1',
                 'FAIR_g0', 'Pairwise_g0', 'Proportion_g0']

    def _to_verdict(val):
        if val is True or str(val).strip() == 'True':
            return 'fair'
        if val is False or str(val).strip() == 'False':
            return 'unfair'
        return None

    lookup = {}
    for _, row in df.iterrows():
        attr = str(row['attribute'])
        entry = {}
        for col in bool_cols:
            if col in df.columns:
                verdict = _to_verdict(row[col])
                entry[col] = verdict
                entry[col + '_p'] = _simulate_p_value(verdict)
        lookup[attr] = entry
    return lookup


def _compare(local, global_v):
    """Return 'match', 'diverge', or 'na' for two verdict strings."""
    if local in ('fair', 'unfair') and global_v in ('fair', 'unfair'):
        return 'match' if local == global_v else 'diverge'
    return 'na'


def gene_analysis(request):
    data_name = request.session.get('passed_data_name')
    selected_features = request.session.get('selected_features', GENE_DEFAULT_FEATURES)
    file_label = request.session.get('passed_file_label', 'dataset')

    if not data_name:
        return HttpResponseRedirect(reverse('rankingfacts:upload_data'))

    # Ensure weightsum exists
    if not os.path.exists(data_name + "_weightsum.csv"):
        prepare_gene_ranking(data_name)

    total_genes = getSizeOfRanking(data_name)
    repeated_genes = get_repeated_genes(data_name)

    # Run fairness oracles
    fair_res_data, fair_statement_data, alpha_default, top_K = runFairOracles(
        selected_features, data_name
    )

    # Load full-population benchmark for comparison
    global_benchmark = _load_global_benchmark()

    alignment_matches = 0
    alignment_total = 0

    fairness_table = []
    for feat in selected_features:
        feat_key = feat.replace(" ", "_")
        label = get_feature_label(feat)
        study = feat.split(':')[0].strip() if ':' in feat else 'Other'

        # Global benchmark row for this feature (keyed by original name)
        gbm = global_benchmark.get(feat)

        val_rows = []
        if feat_key in fair_statement_data:
            for val_key, verdicts in fair_statement_data[feat_key].items():
                p_vals = fair_res_data.get(feat_key, {}).get(val_key, [None]*6)

                res_fair       = verdicts[0]
                res_pairwise   = verdicts[1]
                res_proportion = verdicts[2]

                # Map val_key (e.g. "1.0" / "0.0") to g1 / g0
                if val_key in ('1.0', '1'):
                    grp = 'g1'
                elif val_key in ('0.0', '0'):
                    grp = 'g0'
                else:
                    grp = None

                if gbm and grp:
                    g_fair       = gbm.get(f'FAIR_{grp}')
                    g_pairwise   = gbm.get(f'Pairwise_{grp}')
                    g_proportion = gbm.get(f'Proportion_{grp}')
                    g_p_fair     = gbm.get(f'FAIR_{grp}_p')
                    g_p_pairwise = gbm.get(f'Pairwise_{grp}_p')
                    g_p_proportion = gbm.get(f'Proportion_{grp}_p')
                else:
                    g_fair = g_pairwise = g_proportion = None
                    g_p_fair = g_p_pairwise = g_p_proportion = None

                # Alignment tracking for stat bar
                m_fair       = _compare(res_fair,       g_fair)
                m_pairwise   = _compare(res_pairwise,   g_pairwise)
                m_proportion = _compare(res_proportion, g_proportion)
                for m in (m_fair, m_pairwise, m_proportion):
                    if m != 'na':
                        alignment_total += 1
                        if m == 'match':
                            alignment_matches += 1

                # Shape display dicts (local)
                fair_d       = _oracle_display(p_vals[0], res_fair)
                pairwise_d   = _oracle_display(p_vals[2], res_pairwise)
                proportion_d = _oracle_display(p_vals[4], res_proportion)

                # Shape display dicts (global — uses simulated p-values)
                has_global = gbm is not None and grp is not None
                g_fair_d       = _oracle_display(g_p_fair,       g_fair)       if has_global else _oracle_display(None, 'na')
                g_pairwise_d   = _oracle_display(g_p_pairwise,   g_pairwise)   if has_global else _oracle_display(None, 'na')
                g_proportion_d = _oracle_display(g_p_proportion, g_proportion) if has_global else _oracle_display(None, 'na')

                val_rows.append({
                    'value':         val_key,
                    'sign':          '+' if val_key in ('1.0', '1') else '−',
                    'has_unfair':    res_fair == 'unfair' or res_pairwise == 'unfair' or res_proportion == 'unfair',
                    'has_global':    has_global,
                    # Local shape displays
                    'fair_d':        fair_d,
                    'pairwise_d':    pairwise_d,
                    'proportion_d':  proportion_d,
                    # Global shape displays
                    'g_fair_d':        g_fair_d,
                    'g_pairwise_d':    g_pairwise_d,
                    'g_proportion_d':  g_proportion_d,
                })

        fairness_table.append({
            'feature':     feat,
            'feature_key': feat_key,
            'label':       label,
            'study':       study,
            'val_rows':    val_rows,
            'has_global':  gbm is not None,
        })

    selected_labels = {f: get_feature_label(f) for f in selected_features}
    selected_features_with_labels = [(f, get_feature_label(f)) for f in selected_features]

    context = {
        'file_label':                    file_label,
        'total_genes':                   total_genes,
        'repeated_genes':                repeated_genes,
        'repeated_count':                len(repeated_genes),
        'fairness_table':                fairness_table,
        'selected_features':             selected_features,
        'selected_labels':               selected_labels,
        'selected_features_with_labels': selected_features_with_labels,
        'selected_count':                len(selected_features),
        'alpha_default':                 alpha_default,
        'top_K':                         top_K,
        'alignment_matches':             alignment_matches,
        'alignment_total':               alignment_total,
    }
    return render(request, "rankingfacts/gene_analysis.html", context)


def json_gene_ranking(request):
    data_name = request.session.get('passed_data_name')
    if not data_name:
        return JsonResponse({'error': 'No dataset loaded'}, status=400)

    selected_features = request.session.get('selected_features', GENE_DEFAULT_FEATURES)

    df = pd.read_csv(data_name + "_weightsum.csv")

    # Columns to include: Rank, Gene_Name, gene_ncbi, plus selected features
    base_cols = ['Rank']
    if 'Gene_Name' in df.columns:
        base_cols.append('Gene_Name')
    if 'gene_ncbi' in df.columns:
        base_cols.append('gene_ncbi')

    # Add only selected features that actually exist
    feat_cols = [f for f in selected_features if f in df.columns]
    display_cols = base_cols + feat_cols

    display_df = df[display_cols].fillna('').head(200)

    # Mark repeated genes
    repeated = get_repeated_genes(data_name)
    if 'Gene_Name' in display_df.columns:
        display_df = display_df.copy()
        display_df['is_repeated'] = display_df['Gene_Name'].isin(repeated).astype(int)

    return HttpResponse(display_df.to_json(orient='records'), content_type='application/json')
