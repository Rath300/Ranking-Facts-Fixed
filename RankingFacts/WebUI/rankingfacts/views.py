import json
import os
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


def _load_global_benchmark():
    """
    Load the researcher's pre-computed full-population fairness benchmark.
    Returns a dict: {attribute_name: {FAIR_g0, Pairwise_g0, Proportion_g0,
                                       FAIR_g1, Pairwise_g1, Proportion_g1}}
    Values are 'fair', 'unfair', or None.
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
        lookup[attr] = {col: _to_verdict(row[col]) for col in bool_cols if col in df.columns}
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
                else:
                    g_fair = g_pairwise = g_proportion = None

                m_fair       = _compare(res_fair,       g_fair)
                m_pairwise   = _compare(res_pairwise,   g_pairwise)
                m_proportion = _compare(res_proportion, g_proportion)

                match_count      = sum(1 for m in (m_fair, m_pairwise, m_proportion) if m == 'match')
                comparable_count = sum(1 for m in (m_fair, m_pairwise, m_proportion) if m != 'na')

                for m in (m_fair, m_pairwise, m_proportion):
                    if m != 'na':
                        alignment_total += 1
                        if m == 'match':
                            alignment_matches += 1

                val_rows.append({
                    'value':          val_key,
                    'res_fair':       res_fair,
                    'res_pairwise':   res_pairwise,
                    'res_proportion': res_proportion,
                    'p_fair':         p_vals[0],
                    'alphac_fair':    p_vals[1],
                    'p_pairwise':     p_vals[2],
                    'alpha_pairwise': p_vals[3],
                    'p_proportion':   p_vals[4],
                    'alpha_proportion': p_vals[5],
                    # Global comparison
                    'g_fair':         g_fair,
                    'g_pairwise':     g_pairwise,
                    'g_proportion':   g_proportion,
                    'match_fair':        m_fair,
                    'match_pairwise':    m_pairwise,
                    'match_proportion':  m_proportion,
                    'match_count':       match_count,
                    'comparable_count':  comparable_count,
                    'has_global':        gbm is not None and grp is not None,
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
