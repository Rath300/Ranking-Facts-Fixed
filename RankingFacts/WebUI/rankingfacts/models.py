import json
import math
import os
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import linear_model
from math import sqrt
from django.conf import settings
from .utils import read_json_file
from FAIR.FairnessInRankings import FairnessInRankingsTester

# ---------------------------------------------------------------------------
# Gene feature defaults — pre-selected on the feature selection page
# ---------------------------------------------------------------------------
GENE_DEFAULT_FEATURES = [
    "Lek et al., 2016: exome variation, pLI",
    "Itzhak et al., 2016: protein localization, plasma membrane",
    "Hart et al., 2015: gene essentiality in HeLa",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: liver_a",
    "Tani et al., 2012: RNA halflife",
]

GENE_FEATURE_LABELS = {
    "Lek et al., 2016: exome variation, pLI":                       "LoF Intolerant (pLI)",
    "Lek et al., 2016: exome variation, syn_z":                     "Synonymous Z-score",
    "Lek et al., 2016: exome variation, mis_z":                     "Missense Z-score",
    "Lek et al., 2016: exome variation, lof_z":                     "LoF Z-score",
    "Lek et al., 2016: exome variation, pRec":                      "Prob. Recessive (pRec)",
    "Lek et al., 2016: exome variation, pNull":                     "Prob. Null (pNull)",
    "Itzhak et al., 2016: protein localization, plasma membrane":   "Plasma Membrane Protein",
    "Itzhak et al., 2016: protein localization, mitochondrion":     "Mitochondrial Protein",
    "Itzhak et al., 2016: protein localization, ER":                "ER-localized Protein",
    "Itzhak et al., 2016: protein localization, Golgi":             "Golgi-localized Protein",
    "Itzhak et al., 2016: protein localization, lysosome":          "Lysosomal Protein",
    "Itzhak et al., 2016: protein localization, endosome":          "Endosomal Protein",
    "Itzhak et al., 2016: protein localization, peroxisome":        "Peroxisomal Protein",
    "Itzhak et al., 2016: protein localization, cytosolic pool":    "Cytosolic Protein",
    "Itzhak et al., 2016: protein localization, nuclear pore complex": "Nuclear Pore Complex",
    "Itzhak et al., 2016: protein localization, large protein complex": "Large Protein Complex",
    "Hart et al., 2015: gene essentiality in HeLa":                 "Essential in HeLa (Cancer)",
    "Hart et al., 2015: gene essentiality in A375 GeCKo":           "Essential in A375 (Melanoma)",
    "Hart et al., 2015: gene essentiality in DLD1":                 "Essential in DLD1 (Colorectal)",
    "Hart et al., 2015: gene essentiality in GBM":                  "Essential in GBM (Glioblastoma)",
    "Hart et al., 2015: gene essentiality in HCT116":               "Essential in HCT116 (Colorectal)",
    "Hart et al., 2015: gene essentiality in Rpe1":                 "Essential in Rpe1 (Retinal)",
    "Wang et al., 2015: CRISPR score: KBM7":                        "CRISPR Score: KBM7 (Leukemia)",
    "Wang et al., 2015: CRISPR score: Jiyoye":                      "CRISPR Score: Jiyoye (Lymphoma)",
    "Wang et al., 2015: CRISPR score: Raji":                        "CRISPR Score: Raji (Lymphoma)",
    "Blomen et al., 2015: p value (KBM7 cells)":                    "Essentiality p-val KBM7",
    "Blomen et al., 2015: q value (KBM7 cells)":                    "Essentiality q-val KBM7",
    "Blomen et al., 2015: p value (HAP1 cells)":                    "Essentiality p-val HAP1",
    "Blomen et al., 2015: q value (HAP1 cells)":                    "Essentiality q-val HAP1",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: liver_a":      "Expressed in Liver",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: kidney_a":     "Expressed in Kidney",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: brain_a":      "Expressed in Brain",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: lung_3e":      "Expressed in Lung",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: heart_5a":     "Expressed in Heart",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: pancreas_6a":  "Expressed in Pancreas",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: prostate_4a":  "Expressed in Prostate",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: ovary_6a":     "Expressed in Ovary",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: testis_4a":    "Expressed in Testis",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: hela":         "Expressed in HeLa (Cell Line)",
    "Uhlen et al., 2015: RNA expression, log10 FPKM: hek293":       "Expressed in HEK293 (Cell Line)",
    "Uhlen et al., 2015: Fraction of cells with detected RNA (1 FPKM)": "Fraction of Cells Expressing",
    "Uhlen et al., 2015: Fraction of tissues with detected RNA (1 FPKM)": "Fraction of Tissues Expressing",
    "Tani et al., 2012: RNA halflife":                              "Long RNA Half-Life",
    "Leuenberger et al., 2017: Protein stability: Tm peptide":      "High Peptide Tm (Stability)",
    "Leuenberger et al., 2017: Protein stability: Tm protein":      "High Protein Tm (Stability)",
    "Leuenberger et al., 2017: Protein stability: Tm 90percentile": "High Stability (90th %ile)",
    "Protein: length":                                              "Long Protein",
    "Protein: molecular weight":                                    "High Molecular Weight",
    "Protein: isoelectric point":                                   "High Isoelectric Point",
    "Protein: GRAVY score of hydrophobicity":                       "High Hydrophobicity (GRAVY)",
    "Aminoacids_swiss_or_trembl: A":                                "Swiss/TrEMBL Entry",
}


def get_feature_label(col_name):
    """Return a short display label for a feature column, or a trimmed version of the name."""
    if col_name in GENE_FEATURE_LABELS:
        return GENE_FEATURE_LABELS[col_name]
    # Auto-shorten: take the part after the last colon
    if ':' in col_name:
        parts = col_name.split(':')
        return parts[-1].strip().title()
    return col_name


# ---------------------------------------------------------------------------
# Gene data helpers
# ---------------------------------------------------------------------------

def save_uploaded_file(file, current_file):
    """Save user-uploaded data on server. current_file has no .csv suffix."""
    with open(current_file + ".csv", 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


def prepare_gene_ranking(data_name):
    """
    Read the uploaded CSV (already ranked by row order), add an integer Rank
    column, and save as {data_name}_weightsum.csv so all fairness code can
    read it without modification.
    """
    df = pd.read_csv(data_name + ".csv")
    df.insert(0, "Rank", range(1, len(df) + 1))
    df.to_csv(data_name + "_weightsum.csv", index=False)
    return df


def get_binary_columns_from_csv(data_name):
    """
    Return all columns in the CSV that contain only 0 and 1 values (ignoring NaN).
    Returns a list of column names sorted alphabetically, grouped by study source.
    """
    df = pd.read_csv(data_name + ".csv")
    binary_cols = []
    for col in df.columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
    return sorted(binary_cols)


def get_binary_columns_grouped(data_name):
    """
    Return binary columns grouped by their study source prefix (text before first colon).
    Returns a dict: {group_name: [col_name, ...]}
    """
    cols = get_binary_columns_from_csv(data_name)
    groups = {}
    for col in cols:
        if ':' in col:
            group = col.split(':')[0].strip()
        else:
            group = 'Other'
        groups.setdefault(group, []).append(col)
    return groups


def get_repeated_genes(data_name):
    """
    Return a dict of {gene_name: count} for Gene_Name values appearing more than once.
    Returns empty dict if Gene_Name column doesn't exist.
    """
    df = pd.read_csv(data_name + "_weightsum.csv")
    if 'Gene_Name' not in df.columns:
        return {}
    counts = df['Gene_Name'].value_counts()
    return {gene: int(count) for gene, count in counts.items() if count > 1}


def getSizeOfRanking(current_file):
    """Return number of rows in the weightsum CSV."""
    data = pd.read_csv(current_file + "_weightsum.csv")
    return len(data)


# ---------------------------------------------------------------------------
# Fairness oracle helpers (unchanged from original)
# ---------------------------------------------------------------------------

def mergeUnfairRanking(_px, _sensitive_idx, _fprob):
    rx = [x for x in _px if x not in _sensitive_idx]
    qx = [x for x in _px if x in _sensitive_idx]
    rx.reverse()
    qx.reverse()
    res_list = []
    while len(qx) > 0 and len(rx) > 0:
        r_cur = random.random()
        if r_cur < _fprob:
            res_list.append(qx.pop())
        else:
            res_list.append(rx.pop())
    if len(qx) > 0:
        qx.reverse()
        res_list = res_list + qx
    if len(rx) > 0:
        rx.reverse()
        res_list = res_list + rx
    if len(res_list) < len(_px):
        print("Error!")
    return res_list


def runFairOracles(chosed_atts, current_file, alpha_default=0.05, k_threshold=200, k_percentage=0.5):
    """
    Run FA*IR, Pairwise, and Proportion fairness oracles for each selected feature.

    Returns (fair_res_data, fair_statement_data, alpha_default, top_K)
    where fair_res_data contains p-values and fair_statement_data contains
    'fair'/'unfair'/'NA' strings.
    """
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_n = len(data)
    if total_n > k_threshold:
        top_K = 10
    else:
        top_K = min(10, int(np.ceil(k_percentage * total_n)))

    fair_res_data = {}
    fair_statement_data = {}

    for si in chosed_atts:
        if si not in data.columns:
            continue
        values_si_att = list(data[si].dropna().unique())
        si_value_json = {}
        si_fair_json = {}

        for vi in values_si_att:
            p_value_fair, alphac_fair = computePvalueFAIR(si, vi, current_file, top_K)
            if alphac_fair == 0:
                res_fair = "NA"
            else:
                res_fair = "fair" if p_value_fair > alphac_fair else "unfair"

            p_value_pairwise = computePvaluePairwise(si, vi, current_file)
            if p_value_pairwise is None or p_value_pairwise == 0:
                res_pairwise = "NA"
            else:
                res_pairwise = "fair" if p_value_pairwise > alpha_default else "unfair"

            try:
                p_value_proportion = computePvalueProportion(si, vi, current_file, top_K)
                res_proportion = "fair" if p_value_proportion > alpha_default else "unfair"
            except (ZeroDivisionError, ValueError):
                p_value_proportion = None
                res_proportion = "NA"

            filled_vi = vi if isinstance(vi, str) else str(vi)
            filled_vi = filled_vi.replace(" ", "")

            si_value_json[filled_vi] = [p_value_fair, alphac_fair, p_value_pairwise,
                                         alpha_default, p_value_proportion, alpha_default]
            si_fair_json[filled_vi] = [res_fair, res_pairwise, res_proportion]

        filled_si = si if isinstance(si, str) else str(si)
        filled_si_key = filled_si.replace(" ", "_")
        fair_res_data[filled_si_key] = si_value_json
        fair_statement_data[filled_si_key] = si_fair_json

    return fair_res_data, fair_statement_data, alpha_default, top_K


def computePvalueFAIR(att_name, att_value, current_file, top_K, round_default=2):
    """Compute p-value using FA*IR oracle."""
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)

    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)
    fair_p_vi = size_vi / total_N

    generated_ranking = []
    for idx in range(total_N):
        row_val = data[att_name].iloc[idx]
        label = "pro" if row_val == att_value else "unpro"
        generated_ranking.append((idx, label))

    try:
        p_value, isFair, posAtFail, alpha_c, candidates_needed = computeFairRankingProbability(
            top_K, fair_p_vi, generated_ranking[:top_K]
        )
        return round(p_value, round_default), round(alpha_c, round_default)
    except Exception:
        return 0, 0


def computeFairRankingProbability(k, p, generated_ranking, default_alpha=0.05):
    gft = FairnessInRankingsTester(p, default_alpha, k, correctedAlpha=True)
    posAtFail, isFair = gft.ranked_group_fairness_condition(generated_ranking)
    p_value = gft.calculate_p_value_left_tail(k, generated_ranking)
    return p_value, isFair, posAtFail, gft.alpha_c, gft.candidates_needed


def computePvaluePairwise(att_name, att_value, current_file, run_time=1000, round_default=2):
    """
    Compute p-value using Pairwise oracle.

    First tries pre-computed simulation files for speed. If unavailable, falls
    back to running the Monte Carlo simulation live (fast for small datasets).
    """
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)

    # Try pre-computed simulation file first
    sim_file = os.path.join(
        settings.PLAYDATA_ROOT,
        "SimulationPairs_N" + str(total_N) + "_R1000.json"
    )
    pair_N_vi, estimated_fair_pair_vi, size_vi = computePairN(att_name, att_value, current_file)

    if size_vi == 0 or size_vi == total_N:
        return None

    if os.path.exists(sim_file):
        sim_data = read_json_file(sim_file)
        sample_pairs = sim_data.get(str(size_vi), None)
        if sample_pairs:
            return round(Cdf(sample_pairs, pair_N_vi), round_default)

    # Fall back to live Monte Carlo simulation
    seed_ranking = list(range(total_N))
    seed_protected_idx = set(data[data[att_name] == att_value].index.tolist())
    fair_p = size_vi / total_N

    sample_pairs = []
    for _ in range(run_time):
        sim_ranking = mergeUnfairRanking(seed_ranking, seed_protected_idx, fair_p)
        positions = [i for i, r in enumerate(sim_ranking) if r in seed_protected_idx]
        count = 0
        for i, pos in enumerate(positions):
            left_protected = size_vi - (i + 1)
            count += (total_N - pos - left_protected)
        sample_pairs.append(count)

    cdf_val = Cdf(sample_pairs, pair_N_vi)
    return round(cdf_val, round_default)


def computePvalueProportion(att_name, att_value, current_file, top_K, round_default=2):
    """Compute p-value using Proportion oracle (z-test)."""
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)
    top_data = data.iloc[0:top_K]

    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)
    size_other = total_N - size_vi

    if size_vi == 0 or size_other == 0:
        raise ValueError("Group size is zero — cannot compute proportion z-test.")

    size_vi_top = len(top_data[top_data[att_name] == att_value])
    size_other_top = top_K - size_vi_top

    p_vi_top = size_vi_top / size_vi
    p_other_top = size_other_top / size_other
    p_vi_rest = 1 - p_vi_top
    p_other_rest = 1 - p_other_top

    se_sq = (p_vi_top * p_vi_rest / size_vi) + (p_other_top * p_other_rest / size_other)
    if se_sq <= 0:
        raise ValueError("Pooled SE is zero — cannot compute proportion z-test.")

    pooledSE = sqrt(se_sq)
    z_test = (p_other_top - p_vi_top) / pooledSE
    p_value = norm.sf(z_test)
    return round(p_value, round_default)


def computePairN(att_name, att_value, current_file):
    """Compute number of pairs that att_value > * in the ranked list."""
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)

    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)
    count_vi_prefered_pairs = 0
    for i in range(len(position_lists_val)):
        cur_position = position_lists_val[i]
        left_vi = size_vi - (i + 1)
        count_vi_prefered_pairs += total_N - cur_position - left_vi

    total_pairs_vi = size_vi * (total_N - size_vi)
    estimated_vi_pair = math.ceil((size_vi / total_N) * total_pairs_vi)
    return int(count_vi_prefered_pairs), int(estimated_vi_pair), int(size_vi)


def Cdf(_input_array, x):
    """Compute left-tail CDF value."""
    count = sum(1.0 for vi in _input_array if vi <= x)
    return count / len(_input_array)


# ---------------------------------------------------------------------------
# Legacy stubs retained for compatibility — not called in new flow
# ---------------------------------------------------------------------------

class DataDescriberUI:
    def read_dataset_from_csv(self, filepath):
        self._df = pd.read_csv(filepath)

    def get_json_data(self):
        self.json_data = self._df.to_json(orient='records')
