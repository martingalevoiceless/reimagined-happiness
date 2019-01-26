
minview = 500
explore_target = 30
high_quality_ratio = 0.0
#neighborhood = 0.07
max_neighborhood = 1000
inversion_ratio = 6
inversion_max_count = 90

recent_wins = 20
recent_wins_curve = 8
pref_skip = 1
nopref_skip = 0.3

min_clean_compares = 8
min_clean_wins = 1
goat_window = 200

def precision_func(x):
    #(opts.min_target_precision + (pos ** opts.target_precision_curve) * opts.target_precision_top)
    #min_target_precision = 4
    J = 6.0
    z = 2 ** 13
    j = 1+(J/z)
    m = 3
    return m+j**(z**x)-j
    #return 2+x*20
    #target_precision_curve = 40
    #target_precision_top = 20

def neighborhood_func(x):
    #(opts.min_target_precision + (pos ** opts.target_precision_curve) * opts.target_precision_top)
    #min_target_precision = 4
    J = 6.0
    z = 2 ** 13
    j = 1+(J/z)
    m = 14
    return m+j**(z**x)-j
    #return 2+x*20
    #target_precision_curve = 40
    #target_precision_top = 20

def inversion_precision_func(x):
    #(opts.min_target_precision + (pos ** opts.target_precision_curve) * opts.target_precision_top)
    #min_target_precision = 4
    J = 6.0
    z = 2 ** 13
    j = 1+(J/z)
    m = 100
    return m+j**(z**x)-j
    #return 2+x*20
    #target_precision_curve = 40
    #target_precision_top = 20

drop_ratio = 20
drop_min = 7
model_drop_min = 4
max_delay = 10
softmin_falloff_per_unit = 10.0
inversion_neighborhood = 0.5
inversion_max = 4
seen_suppression_max = 60*60*24*3
seen_suppression_min = 120
seen_suppression_rate = 2
fix_inversions = True
#inversion_ratio = 100
too_close_boost = 3
initial_mag = 8
min_mag = 0.3
last_winner_prob = 0.0
last_winner_last_winner_prob = 0.0

min_frag_length = 10

seen_noise_lmean = 0
seen_noise_lstd = 1

comparison_half_life = 60 * 60 * 24 * 15
comparison_min = 0.1
def comparison_decay_func(age_seconds):
    return max(comparison_min, 1/(1+age_seconds / comparison_half_life))

inversion_compare_boost = 1.2
inversion_compare_relboost = 0.2

initial_tolerance = 2e-7
ongoing_tolerance = 1e-5

alpha = lambda count: 0.3/count

min_to_rank = 3

weighted_softmin_sharpness = 4

ambiguity_threshold = 0.6

