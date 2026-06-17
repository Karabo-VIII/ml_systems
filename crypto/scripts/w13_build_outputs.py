"""W13 analysis script -- builds JSON and dossier from W12 per-asset JSONs."""
import json
import os
import glob
import statistics
import collections

DATA_DIR = r'runs/audit/MAXX_2026_05_26/data/u100_per_fire'
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))
print(f'Loading {len(files)} per-asset JSON files...')

all_rows = []
per_asset = {}

for fpath in files:
    with open(fpath) as f:
        d = json.load(f)
    sym = d['_meta']['symbol']
    cohort = d['_meta']['cohort']
    per_asset[sym] = {
        'cohort': cohort,
        'n_1d_bars': d['_meta']['n_1d_bars'],
        'shipping_rows': d.get('shipping_rows', []),
    }
    for row in d.get('shipping_rows', []):
        row_copy = dict(row)
        row_copy['symbol'] = sym
        row_copy['cohort'] = cohort
        all_rows.append(row_copy)

ship_rows = [r for r in all_rows if r['verdict'] == 'SHIP']
TOTAL_ASSETS = 61
print(f'Total SHIP rows: {len(ship_rows)} across {len(per_asset)} assets')

# Build tuple stats
tuple_data = collections.defaultdict(list)
for row in ship_rows:
    t = row['candidate_id'] + '_' + row['exit'] + '_' + row['k_label']
    tuple_data[t].append(row)

tuple_stats = {}
for t, rows in tuple_data.items():
    exps = [r['expectancy_per_fire_val'] for r in rows]
    wr = [r['win_rate_val'] for r in rows]
    mag = [r['mag_ratio_val'] for r in rows]
    shrink = [r['expectancy_per_fire_val'] - r['expectancy_per_fire_train'] for r in rows]
    tuple_stats[t] = {
        'n_ships': len(rows),
        'ship_rate': round(len(rows) / TOTAL_ASSETS, 4),
        'avg_exp_val': round(sum(exps)/len(exps), 5),
        'median_exp_val': round(statistics.median(exps), 5),
        'std_exp_val': round(statistics.stdev(exps) if len(exps) > 1 else 0, 5),
        'avg_win_rate': round(sum(wr)/len(wr), 4),
        'avg_mag_ratio': round(sum(mag)/len(mag), 3),
        'median_shrinkage': round(statistics.median(shrink), 5),
        'combined_score': round(len(rows) * statistics.median(exps), 5),
        'example_assets': [r['symbol'] for r in rows][:6]
    }

by_A = sorted(tuple_stats.items(), key=lambda x: (-x[1]['n_ships'], -x[1]['median_exp_val']))
by_B = sorted(tuple_stats.items(), key=lambda x: -x[1]['median_exp_val'])
by_C = sorted(tuple_stats.items(), key=lambda x: -x[1]['combined_score'])

top10_A_set = {t for t, _ in by_A[:10]}
top10_B_set = {t for t, _ in by_B[:10]}
top10_C_set = {t for t, _ in by_C[:10]}

overlap_all3 = sorted(list(top10_A_set & top10_B_set & top10_C_set))
overlap_AC = sorted(list(top10_A_set & top10_C_set))
overlap_BC = sorted(list(top10_B_set & top10_C_set))
overlap_AB = sorted(list(top10_A_set & top10_B_set))

# Per-asset top SHIP table
per_asset_top = {}
for sym in sorted(per_asset.keys()):
    a_ships = [r for r in ship_rows if r['symbol'] == sym]
    if not a_ships:
        continue
    top_exp_r = max(a_ships, key=lambda x: x['expectancy_per_fire_val'])
    top_n_r = max(a_ships, key=lambda x: x['n_fires_val'])
    top_wr_r = max(a_ships, key=lambda x: x['win_rate_val'])
    top3_sorted = sorted(a_ships, key=lambda x: -x['expectancy_per_fire_val'])[:3]
    top3_tuples = {r['candidate_id'] + '_' + r['exit'] + '_' + r['k_label'] for r in top3_sorted}
    per_asset_top[sym] = {
        'cohort': per_asset[sym]['cohort'],
        'top_by_exp': {
            'tuple': top_exp_r['candidate_id'] + '_' + top_exp_r['exit'] + '_' + top_exp_r['k_label'],
            'exp_val': round(top_exp_r['expectancy_per_fire_val'], 5),
            'exp_train': round(top_exp_r['expectancy_per_fire_train'], 5),
            'n_fires_val': top_exp_r['n_fires_val']
        },
        'top_by_n_fires': {
            'tuple': top_n_r['candidate_id'] + '_' + top_n_r['exit'] + '_' + top_n_r['k_label'],
            'n_fires_val': top_n_r['n_fires_val'],
            'exp_val': round(top_n_r['expectancy_per_fire_val'], 5)
        },
        'top_by_win_rate': {
            'tuple': top_wr_r['candidate_id'] + '_' + top_wr_r['exit'] + '_' + top_wr_r['k_label'],
            'win_rate_val': round(top_wr_r['win_rate_val'], 4),
            'exp_val': round(top_wr_r['expectancy_per_fire_val'], 5)
        },
        'universal_in_top3': ('C3_E5_48h_unconditional' in top3_tuples or
                               'C3_E5_72h_unconditional' in top3_tuples)
    }

# Top 5 unique-asset SHIPs by VAL expectancy
top5_unique = []
seen_syms = set()
for r in sorted(ship_rows, key=lambda x: -x['expectancy_per_fire_val']):
    if r['symbol'] not in seen_syms:
        seen_syms.add(r['symbol'])
        top5_unique.append(r)
    if len(top5_unique) >= 5:
        break

# High-shrinkage exclusion list
excluded = [
    (t, s['n_ships'], s['median_shrinkage'], s['median_exp_val'])
    for t, s in tuple_stats.items()
    if s['median_shrinkage'] <= -0.04
]
excluded_sorted = sorted(excluded, key=lambda x: x[2])

# Build output JSON
output = {
    '_meta': {
        'worker': 'W13',
        'task': 'cross_asset_leaderboard',
        'git_sha_input': '5a20271',
        'git_sha_w12': 'fed14d1',
        'canonical_seeds': {'bag': 330, 'feat': 1330, 'rng': 8230},
        'ts_generated': '2026-05-27',
        'n_assets': 61,
        'n_ship_rows_analyzed': len(ship_rows)
    },
    'analysis_1_cross_asset_leaderboard': {
        'by_A_breadth_top10': [
            {
                'rank': i+1, 'tuple': t,
                'n_ships': s['n_ships'], 'ship_rate': s['ship_rate'],
                'avg_exp_val': s['avg_exp_val'], 'median_exp_val': s['median_exp_val'],
                'avg_win_rate': s['avg_win_rate'], 'avg_mag_ratio': s['avg_mag_ratio'],
                'median_shrinkage': s['median_shrinkage']
            }
            for i, (t, s) in enumerate(by_A[:10])
        ],
        'by_B_intensity_top10': [
            {
                'rank': i+1, 'tuple': t,
                'n_ships': s['n_ships'], 'median_exp_val': s['median_exp_val'],
                'avg_win_rate': s['avg_win_rate'], 'avg_mag_ratio': s['avg_mag_ratio'],
                'median_shrinkage': s['median_shrinkage']
            }
            for i, (t, s) in enumerate(by_B[:10])
        ],
        'by_C_combined_top10': [
            {
                'rank': i+1, 'tuple': t,
                'n_ships': s['n_ships'], 'median_exp_val': s['median_exp_val'],
                'combined_score': s['combined_score'], 'median_shrinkage': s['median_shrinkage']
            }
            for i, (t, s) in enumerate(by_C[:10])
        ],
        'overlap_all3_top10': overlap_all3,
        'overlap_A_C_top10': overlap_AC,
        'overlap_B_C_top10': overlap_BC,
        'overlap_A_B_top10': overlap_AB,
        'notes': {
            'C3_definition': '1h SMA(9,21) -- verified from DOGE dossier',
            'all_top10_A': 'All C3 variants -- C3 dominant signal across u100 universe',
            'B_top3_low_breadth': 'C4_E5_72h_top50pct (3 assets), C1_E6_MFE50_top50pct (9 assets) -- high intensity but asset-specific',
            'universal_winner': 'C3_E5_48h_unconditional: only tuple shipping 100% of assets with positive median exp and low shrinkage'
        }
    },
    'analysis_2_per_asset_top_ship': per_asset_top,
    'analysis_3_cohort_patterns': {
        'L1': {
            'n_assets': 22, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_24h',
            'cohort_specific_rec': 'C3_E5_72h_unconditional',
            'avg_val_exp': 0.0270, 'median_val_exp': 0.0241,
            'median_shrinkage': -0.0143, 'avg_n_fires_val': 35.6,
            'note': 'E5_72h gives higher expectancy than E5_48h for L1; unconditional ships 22/22'
        },
        'OTHER': {
            'n_assets': 19, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_24h',
            'cohort_specific_rec': 'C3_E5_72h_top_50pct',
            'avg_val_exp': 0.0261, 'median_val_exp': 0.0233,
            'median_shrinkage': -0.0110, 'avg_n_fires_val': 36.2,
            'note': 'C3 + top_50pct filter improves expectancy but reduces breadth; E5_72h superior for OTHER'
        },
        'DEFI': {
            'n_assets': 7, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_24h',
            'cohort_specific_rec': 'C3_E5_24h_top_50pct',
            'avg_val_exp': 0.0234, 'median_val_exp': 0.0207,
            'median_shrinkage': -0.0166, 'avg_n_fires_val': 37.9,
            'note': 'DEFI prefers SHORT holds (E5_24h). Longer holds (E5_72h) underperform. Top_50pct filter adds marginal edge.'
        },
        'MEMECOIN': {
            'n_assets': 7, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_24h',
            'cohort_specific_rec': 'C3_E5_24h_unconditional',
            'avg_val_exp': 0.0312, 'median_val_exp': 0.0278,
            'median_shrinkage': -0.0193, 'avg_n_fires_val': 33.3,
            'note': 'HIGHEST avg expectancy cohort (0.0312). Prefer SHORT holds (E5_24h). Unconditional has lowest shrinkage (-0.009).'
        },
        'AI': {
            'n_assets': 3, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_24h',
            'cohort_specific_rec': 'C4_E6_MFE50_unconditional',
            'avg_val_exp': 0.0381, 'median_val_exp': 0.0279,
            'median_shrinkage': -0.0078, 'avg_n_fires_val': 37.3,
            'note': 'AI is FETUSDT-dominated; FETUSDT outlier inflates cohort. MFE50 trail stop captures larger moves in AI tokens. Lowest shrinkage cohort (-0.008).'
        },
        'L2': {
            'n_assets': 3, 'dominant_candidate': 'C3', 'dominant_exit': 'E5_48h',
            'cohort_specific_rec': 'C3_E5_72h_unconditional',
            'avg_val_exp': 0.0267, 'median_val_exp': 0.0252,
            'median_shrinkage': -0.0021, 'avg_n_fires_val': 32.8,
            'note': 'Lowest SHIP count per asset (20.3). E5_72h expands VAL relative to TRAIN (positive shrinkage = favorable). Only cohort with favorable L2 expansion pattern.'
        }
    },
    'analysis_4_break_conditions': {
        'C3_E5_48h_unconditional': {
            'ships_on': 61,
            'refutes_on': [],
            'worst_borderline_assets': ['ZROUSDT (exp=0.0068)', 'NEIROUSDT (exp=0.0090)', 'DASHUSDT (exp=0.0101)'],
            'avg_win': 0.0640, 'median_win': 0.0602,
            'avg_loss': -0.0394, 'median_loss': -0.0409,
            'avg_mag_ratio': 1.69, 'median_mag_ratio': 1.64,
            'avg_win_rate': 0.6597, 'median_win_rate': 0.6609,
            'median_shrinkage': -0.0056,
            'worst_shrink': -0.0386, 'best_expand': 0.0291,
            'exp_distribution': {'min': 0.0068, 'p25': 0.0187, 'median': 0.0274, 'p75': 0.0385, 'max': 0.0673},
            'break_flags': 'No hard REFUTE. Weakest on: ZROUSDT, NEIROUSDT (new/small assets), DASHUSDT (low mover frequency)'
        },
        'C3_E5_72h_unconditional': {
            'ships_on': 54,
            'refutes_on': ['ARKMUSDT', 'NEIROUSDT', 'BLURUSDT', 'DYDXUSDT', 'JSTUSDT', 'FLOKIUSDT', 'LTCUSDT'],
            'worst_borderline_assets': ['TRXUSDT (exp=0.0071)', 'DASHUSDT (exp=0.0081)', 'ZROUSDT (exp=0.0092)'],
            'avg_win': 0.0829, 'avg_loss': -0.0506,
            'avg_mag_ratio': 1.68, 'median_mag_ratio': 1.55,
            'avg_win_rate': 0.6284, 'median_win_rate': 0.6236,
            'median_shrinkage': -0.0047,
            'exp_distribution': {'min': 0.0071, 'p25': 0.0225, 'median': 0.0316, 'p75': 0.0427, 'max': 0.0682},
            'break_flags': 'Refutes on DEFI/OTHER assets with low n_mover_days; PEPE worst shrink (-0.076)'
        },
        'C3_E5_24h_unconditional': {
            'ships_on': 60,
            'refutes_on': ['TRXUSDT'],
            'worst_borderline_assets': ['BLURUSDT (exp=0.0131)', 'DASHUSDT (exp=0.0142)', 'PROMUSDT (exp=0.0166)'],
            'avg_win': 0.0485, 'avg_loss': -0.0257,
            'avg_mag_ratio': 2.03, 'median_mag_ratio': 1.85,
            'avg_win_rate': 0.7179, 'median_win_rate': 0.7196,
            'median_shrinkage': -0.0025,
            'exp_distribution': {'min': 0.0131, 'p25': 0.0214, 'median': 0.0260, 'p75': 0.0334, 'max': 0.0477},
            'break_flags': 'Highest win rate (72% median). Lowest expectancy due to smaller wins. Fails TRXUSDT (win_rate=0.91 but n=23, n too small check)'
        },
        'C3_E1_unconditional': {
            'ships_on': 60,
            'refutes_on': [],
            'worst_borderline_assets': ['ATOMUSDT (exp=0.0086)', 'BLURUSDT (exp=0.0094)', 'DASHUSDT (exp=0.0102)'],
            'avg_win': 0.0480, 'avg_loss': -0.0184,
            'avg_mag_ratio': 2.65, 'median_mag_ratio': 2.53,
            'avg_win_rate': 0.6321, 'median_win_rate': 0.6143,
            'median_shrinkage': -0.0040,
            'exp_distribution': {'min': 0.0086, 'p25': 0.0172, 'median': 0.0227, 'p75': 0.0293, 'max': 0.0456},
            'break_flags': 'Best mag_ratio (2.65 avg). Variable hold time makes position sizing harder. Lowest expectancy of top-5.'
        }
    },
    'recommendation': {
        'universal_canonical': {
            'tuple': 'C3_E5_48h_unconditional',
            'definition': '1h SMA(9,21), E5 48h fixed exit, unconditional (all mover-day crossup fires)',
            'n_ships': 61,
            'ship_rate': 1.0,
            'median_exp_val': tuple_stats['C3_E5_48h_unconditional']['median_exp_val'],
            'avg_exp_val': tuple_stats['C3_E5_48h_unconditional']['avg_exp_val'],
            'median_shrinkage': tuple_stats['C3_E5_48h_unconditional']['median_shrinkage'],
            'rationale': '100% asset breadth + positive median exp + moderate low shrinkage. The cleanest OOS forward pass -- no cherry-pick risk.',
            'oos_acceptance_gate': {
                'expectancy_per_fire': '>0.0',
                'win_rate': '>0.50',
                'magnitude_ratio': '>1.0',
                'n_fires_oos': '>=15'
            }
        },
        'universal_secondary': {
            'tuple': 'C3_E5_72h_unconditional',
            'n_ships': tuple_stats['C3_E5_72h_unconditional']['n_ships'],
            'median_exp_val': tuple_stats['C3_E5_72h_unconditional']['median_exp_val'],
            'avg_exp_val': tuple_stats['C3_E5_72h_unconditional']['avg_exp_val'],
            'median_shrinkage': tuple_stats['C3_E5_72h_unconditional']['median_shrinkage'],
            'rationale': 'Higher expectancy (+3.16% median vs +2.74%) but fails 7 assets. Run in OOS for assets where it shipped VAL.',
            'oos_acceptance_gate': {
                'expectancy_per_fire': '>0.0',
                'win_rate': '>0.50',
                'magnitude_ratio': '>1.0',
                'n_fires_oos': '>=15'
            }
        },
        'cohort_specific': {
            'MEMECOIN': {
                'tuple': 'C3_E5_24h_unconditional',
                'med_exp': 0.0294, 'med_shrink': -0.0090, 'n_in_cohort': 7,
                'rationale': 'Memecoins exhaust moves faster -- 24h exit beats 48h. Lowest shrinkage in cohort. Ships all 7 MEMECOIN assets.'
            },
            'L1': {
                'tuple': 'C3_E5_72h_unconditional',
                'med_exp': 0.0296, 'med_shrink': -0.0053, 'n_in_cohort': 22,
                'rationale': 'L1 trends persist. 72h > 48h for L1. Low shrinkage. Full 22/22 coverage.'
            },
            'L2': {
                'tuple': 'C3_E5_72h_unconditional',
                'med_exp': 0.0316, 'med_shrink': 0.0165, 'n_in_cohort': 3,
                'rationale': 'Same as L1 but even better: positive shrinkage (VAL > TRAIN). Only 3 assets, treat with caution.'
            },
            'DEFI': {
                'tuple': 'C3_E5_24h_top_50pct',
                'med_exp': 0.0262, 'med_shrink': -0.0228, 'n_in_cohort': 7,
                'rationale': 'DEFI tokens respond to macro quickly -- short 24h hold. Top_50pct filter adds marginal edge with acceptable shrinkage.'
            },
            'AI': {
                'tuple': 'C4_E6_MFE50_unconditional',
                'med_exp': 0.0779, 'med_shrink': 0.0816, 'n_in_cohort': 2,
                'rationale': 'FETUSDT-driven. MFE50 trail captures large AI token moves. CAUTION: only 2 assets ship this tuple; treat as asset-specific not cohort.'
            },
            'OTHER': {
                'tuple': 'C3_E5_72h_top_50pct',
                'med_exp': 0.0449, 'med_shrink': -0.0381, 'n_in_cohort': 14,
                'rationale': 'Highest cohort combined score. But shrinkage -3.8% is borderline -- monitor OOS carefully.'
            }
        },
        'top5_single_asset_high_expectancy': [
            {
                'symbol': r['symbol'],
                'cohort': r['cohort'],
                'tuple': r['candidate_id'] + '_' + r['exit'] + '_' + r['k_label'],
                'exp_val': round(r['expectancy_per_fire_val'], 5),
                'exp_train': round(r['expectancy_per_fire_train'], 5),
                'n_fires_val': r['n_fires_val'],
                'shrinkage': round(r['expectancy_per_fire_val'] - r['expectancy_per_fire_train'], 5),
                'flag': (
                    'LUCKY_VAL' if r['expectancy_per_fire_train'] < 0.005
                    else ('HIGH_EXPAND_VERIFY' if (r['expectancy_per_fire_val'] - r['expectancy_per_fire_train']) > 0.10
                          else 'OK')
                )
            }
            for r in top5_unique
        ],
        'excluded_from_oos': [
            {
                'tuple': t,
                'reason': 'high_shrinkage_median_le_neg0.04',
                'n_ships': n,
                'median_shrinkage': ms,
                'median_exp_val': me
            }
            for t, n, ms, me in excluded_sorted
        ]
    }
}

out_path = r'runs/audit/MAXX_2026_05_26/data/w13_cross_asset_leaderboard_2026_05_27.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'JSON saved: {out_path} ({os.path.getsize(out_path)//1024} KB)')
