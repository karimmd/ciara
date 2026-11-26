# Column dictionary of `data/`

One section per file. The files are tidy: one row per node-block, request, arm and outcome, or per quantity of an analysis. Files named `*_blocks.csv` hold node-blocks (one row per node and block); files whose columns are `path` and `value` (or `key` and `value`) hold the key paths of a nested result flattened to rows.

**Outcome names:** `Y_Q` quality, `Y_L_f1` and `Y_L_f2` tail exceedance of family 1 and family 2, `Y_D` deadline misses, `Y_E_A`, `Y_E_B`, `Y_E_C` device energy of class A, B, C, `Y_C` cloud share. Family 1 is device control by tool calls; family 2 is sensor interpretation (the escalable family). Classes A, B, C are NB-IoT sensors, LTE-M actuators and 5G cameras.

**Arms:** `I1-E` is the incumbent; `q35-4b-ih-w8a8|b8|c32|f1prio` and `q35-4b-ih-w8a8|b8|c32|fifo` are the priority arm and the FIFO arm of CIARA (4B model at INT8, batch cap 8, token-budget escalation of 32 tokens, priority or FIFO admission). `I1-S`, `I1-oracle`, `I2`, `I3`, `RecServe`, `RecServe-ft`, `CB` are the baselines.

**Units:** Energy columns are in constant wire-overhead units unless the name ends in `_measured`. Times are in seconds unless the name ends in `_ms`. Shares and means of indicators are fractions in [0, 1].

**Carryover bands:** The column `band` holds `planned` (the planned carryover band, t_bar = 5, which decides), `tbar8` (t_bar = 8, sensitivity) or `measured` (95 percent upper limit of the carryover measured on the switches of the look, sensitivity). `look_day` is the look: day 4 (230 rounds) or day 7 (403 rounds).

**Runs:** `campaign` is the certification campaign, `e4` the held-out deployment, `e1e5` the comparison run, `e2` the reference levels for the validity analysis (hold-one-configuration runs), `e8v9` the load contexts outside the certified one, `e9` the second model family, `e5off` the offline baselines.

**Flag columns:** A column `mock` is 0 in every row. Empty cells are missing values.

### `ablation_rows.csv`

Source of every bar of Figure 5(b): the worst level over budget of CIARA and of the variants that remove one part of it. **Rows:** 5.

| column | meaning |
|---|---|
| `variant` | CIARA or a variant that removes one part of it |
| `source` | comparison run, or offline at the reference-run levels |
| `worst_level_over_budget` | Largest (mean + 2 SE) / budget over the seven cost rows (offline FIFO only: the incumbent's reference level, no standard error) |

### `campaign_a7_stationarity.csv`

Within-cell stationarity tests of condition A7, per arm and outcome, on the 403 complete rounds of the campaign. **Rows:** 24.

| column | meaning |
|---|---|
| `arm` | Arm |
| `outcome` | Outcome |
| `n_rounds` | Rounds used |
| `slope_per_100` | Slope of the round values per 100 rounds (Newey-West) |
| `slope_se_per_100` | Newey-West standard error of the slope |
| `slope_t` | t statistic of the slope |
| `half_diff` | Second-half mean minus first-half mean |
| `half_se` | Standard error of the difference |
| `half_t` | t statistic |

### `campaign_arm_estimates.csv`

Arm estimates of the campaign: centers, radii and upper bounds at the looks of day 4 and day 7, under the three carryover bands. **Rows:** 93.

| column | meaning |
|---|---|
| `look_day` | Look (4 or 7); the campaign stopped at day 7 |
| `band` | `planned` (planned band, t_bar = 5), `tbar8` (t_bar = 8, sensitivity), `measured` (95 percent upper limit of the carryover measured on the switches of the look, sensitivity; day 7 only) |
| `arm` | Arm id |
| `outcome` | Outcome name |
| `rounds` | Rounds with a node-block of the arm |
| `propensity` | Arm probability under the balanced design |
| `center` | Round-weighted arm mean (center of the interval) |
| `lower` | center minus radius |
| `radius` | Interval radius: anytime-valid radius plus the carryover band of the look (the radius of the look is in the look file; tbar8 and measured radii are upper minus center) |
| `upper` | center plus radius (the one-sided bound used by the certificate for cost rows) |
| `band_t` | Carryover band used in the radius (`planned`: the arm band; tbar8: the t_bar = 8 arm band; measured: the band of the outcome, in outcome units) |
| `radius_derived` | 1 when the radius is upper minus center because the look file stores the upper bound only |

### `campaign_budget_sweep.csv`

The day-7 certified program solved again with one budget row (or the three energy rows together) scaled from 0.50 to 1.60 of its planned value. **Rows:** 555.

| column | meaning |
|---|---|
| `scope` | Budget row scaled: Y_C, Y_L_f1, Y_L_f2, Y_D, or energy (the three energy rows together) |
| `factor` | Scale factor of the planned budget (0.50 to 1.60) |
| `budget` | Scaled budget of the (first) row |
| `switch` | 1 when a certified policy exists |
| `lp_status` | Status |
| `applied` | Program that decided |
| `w_<arm>` | Policy weights |
| `quality_cost_bound` | Certified quality cost bound |
| `nonincumbent_mass` | Mass outside the incumbent |

### `campaign_certificate_rows.csv`

Certificate per budget row at each look and carryover band. **Rows:** 35.

| column | meaning |
|---|---|
| `look_day` | Look (4 or 7) |
| `band` | `planned`, `tbar8` or `measured` (the three carryover bands) |
| `row` | Constraint row (outcome) |
| `budget` | Budget c of the row |
| `policy_upper` | Certified upper bound of the policy on the row (`planned`: upper_bounds; other bands: budget minus slack); empty without a certified policy |
| `policy_slack` | Budget minus the policy upper bound (0 on a binding row) |
| `incumbent_upper` | Upper bound of the incumbent a0 on the row |
| `incumbent_margin` | Budget minus the incumbent upper bound (negative: the incumbent breaks the budget at its upper bound) |
| `binding` | 1 when the policy slack is zero (the row binds) |

### `campaign_coupling_tests.csv`

Slope tests of the monotone coupling A5'' at each look, flattened to key paths. **Rows:** 1,083.

| column | meaning |
|---|---|
| `look_day` | Look (4 or 7) |
| `path` | Key path in the look file (coupling_tests/<part>/..., coupling/..., share_constraint/...) |
| `value` | Value (slope, standard error, counts, p-values, margins, zeta terms) |

### `campaign_looks.csv`

One row per look and carryover band: decision, status of the pessimistic program and quality cost bound. **Rows:** 5.

| column | meaning |
|---|---|
| `look_day` | Look (4 or 7) |
| `band` | `planned`, `tbar8` or `measured` (the three carryover bands) |
| `complete_rounds` | Complete rounds at the look |
| `branch` | Decision branch (budget_compliance) |
| `lp_status` | Status of the certified program (optimal or infeasible) |
| `switch` | 1 when a certified policy that differs from the incumbent exists and the campaign stops (planned band decides; the others are sensitivities) |
| `quality_cost_bound` | Certified upper bound on J_Q(a0) minus J_Q(pi) |
| `gate_value` | Gate value of the look (negative when the cost bound is within the gate) |
| `nonincumbent_mass` | Mass of the policy outside the incumbent |
| `incumbent_feasible` | 1 when the incumbent meets every budget at its upper bound |
| `eta_dep` | Deployment-law term of the certificate |
| `t_bar` | Carryover horizon in epochs (planned 5) |
| `written` | Time the look was written |
| `reason` | Decision reason (first 240 characters) |

### `campaign_node_blocks.csv`

Campaign node-block outcomes (measurement window, one row per node and round). The campaign stopped at the day-7 look. The certificate driver reads this file. **Rows:** 1,616.

| column | meaning |
|---|---|
| `round` | Round index (a balanced round: one block per edge node, all nodes in the same wall-clock slot); equals block_id |
| `node` | Edge node E1 to E4 |
| `block_id` | Block index of the node, equals round |
| `arm_id` | Arm played by the node in the block (I1-E is the incumbent; the other two are the members c32 (token-budget escalation of 32 tokens) with priority or FIFO admission) |
| `kind` | incumbent or static |
| `propensity` | Probability of the arm for the block under the balanced design |
| `complete_round` | 1 when the round has four complete node-blocks (the rounds counted by the looks), 0 for the partial round at the stop |
| `in_day4_look` | 1 when the round is part of the day-4 look (first 230 rounds) |
| `epoch_lo` | First epoch of the measurement window of the block |
| `epoch_hi` | Last epoch (exclusive) |
| `context_id` | Policy context of the block (pooled in the campaign) |
| `design` | balanced |
| `epsilon` |  |
| `rng_draw` | Uniform draw that fixed the arm |
| `cfg_profile` | Model profile of the arm |
| `cfg_batch_cap` | Batch cap of the arm |
| `cfg_escalate_family2` | Cascade escalation of family 2 requests |
| `cfg_router` | Router |
| `profile_switched` | 1 when the block began with a profile switch |
| `switch_time_s` | Switch time (s) when profile_switched |
| `block_status` | complete or partial |
| `block_switches` | Switches in the block |
| `block_switch_time_total_s` | Seconds spent switching in the block |
| `block_actuation_failures` | Failed actuations in the block |
| `epochs_logged_window` | Epochs logged in the measurement window |
| `profiles_in_window` |  |
| `batch_caps_in_window` |  |
| `routers_in_window` |  |
| `n_switch_epochs_window` |  |
| `switch_time_window_s` |  |
| `n_config_not_ok_window` |  |
| `n_epoch_errors_window` |  |
| `n` | Requests in the window |
| `Y_Q` | Quality (task score mean over requests) |
| `Y_D` | Deadline-miss share |
| `Y_C` | Cloud share (requests answered by the cloud) |
| `n_f1` | Family 1 requests |
| `Y_L_f1` | Tail exceedance, family 1 (share above the latency threshold) |
| `Y_Q_f1` | Quality, family 1 |
| `n_f2` | Family 2 requests |
| `Y_L_f2` | Tail exceedance, family 2 |
| `Y_Q_f2` | Quality, family 2 |
| `n_A` | Class A requests |
| `Y_E_A` | Device energy outcome of class A, mean of min(E/(2 E_d), 1), constant wire overhead 1.0456 |
| `n_B` | Class B requests |
| `Y_E_B` | Device energy outcome of class B, same definition |
| `n_C` | Class C requests |
| `Y_E_C` | Device energy outcome of class C, same definition |
| `lat_p50_s` | Median latency at the device (s) |
| `lat_p95_s` | 95th percentile latency (s) |
| `n_failed` | Failed requests |
| `n_cloud` | Requests served by the cloud |
| `n_compact_fallback` |  |
| `n_truncated` |  |
| `n_no_gateway_record` |  |
| `n_admitted_earlier_block` |  |
| `n_arrived_earlier_block` |  |
| `gpu_energy_j` | Edge GPU energy in the window (J) |
| `gpu_power_w_mean` | Mean edge GPU power (W) |
| `gpu_util_mean` | Mean edge GPU utilization (percent) |
| `gpu_throttled_frac` | Share of epochs with thermal throttling |
| `cloud_gpu_energy_j` | Cloud GPU energy in the window (J) |
| `Y_D_logged` | Deadline-miss share as logged by the client (check of Y_D) |

### `campaign_node_tests.csv`

Node-by-arm interaction tests of the campaign, flattened to key paths. **Rows:** 136.

| column | meaning |
|---|---|
| `outcome` | Outcome |
| `kind` | node_mean (arm mean at the node), contrast_max_change_in_se (leave-one-out change of the arm contrast, in standard errors), F_node and F_interaction (F statistic; p is the Holm p), throttle_slope |
| `arm` | Arm (node_mean and contrast rows) |
| `node` | Node (node_mean rows) |
| `value` | Value of the quantity named in kind |
| `n` | Observations |
| `p` | p-value (Holm-adjusted for the F tests; unadjusted for the throttle slope) |

### `campaign_nodeadj_arms.csv`

Node-adjusted sensitivity of the arm estimates, 403 rounds. **Rows:** 24.

| column | meaning |
|---|---|
| `outcome` | Outcome |
| `arm` | Arm |
| `adjusted` | Node-adjusted arm mean |
| `arm_only` | Arm-only mean with the same rounds |
| `look_center` | Center of the look (estimator of the campaign) |
| `radius` | Planned radius |
| `se` | Standard error of the adjusted mean |
| `shift_in_se` | Shift of the adjusted center from the look center in standard errors |

### `campaign_nodeadj_contrasts.csv`

Node-adjusted sensitivity: contrasts of each arm against the incumbent. **Rows:** 16.

| column | meaning |
|---|---|
| `outcome` | Outcome |
| `arm` | Arm contrasted with the incumbent (arm minus I1-E) |
| `adjusted` | Node-adjusted contrast |
| `adjusted_lo` | Lower end of the adjusted interval |
| `adjusted_hi` | Upper end |
| `arm_only` | Arm-only contrast |
| `look_contrast` | Contrast at the look centers |
| `planned_lo` | Interval of the look, lower end |
| `planned_hi` | Interval of the look, upper end |
| `se` | Standard error |
| `excludes_zero_adjusted` | 1 when the adjusted interval excludes zero |

### `campaign_nodeadj_program.csv`

Node-adjusted sensitivity: the pessimistic program with the adjusted estimates, flattened to key paths. **Rows:** 84.

| column | meaning |
|---|---|
| `path` | Key path in NODEADJ.json |
| `value` | Value: planned and adjusted program (switch, pi, t*, quality cost bound, slack), flags |

### `campaign_policies.csv`

Certified policy weights per look, carryover band and arm. **Rows:** 15.

| column | meaning |
|---|---|
| `look_day` | Look (4 or 7) |
| `band` | `planned`, `tbar8` or `measured` (the three carryover bands) |
| `arm` | Arm id |
| `mass` | Policy mass of the arm (empty when no certified policy exists: LP infeasible) |
| `certified_policy` | 1 when a certified policy exists under the band |

### `campaign_requests_reduced.csv.gz`

Requests of the measurement windows of the campaign (epochs 25 to 49 of each block), one row per request. `e_norm_const` equals `e_norm`: the campaign is in constant units. **Rows:** 520,953.

| column | meaning |
|---|---|
| `node` | Edge node (E1 to E4) |
| `round` | Round (a balanced round has one block per edge node); in the held-out deployment, the round of 1,200 epochs |
| `epoch` | Epoch of the request (30 s epochs) |
| `arm` | Arm of the block |
| `class` | Device class A, B or C (NB-IoT sensors, LTE-M actuators, 5G cameras) |
| `family` | Request family: 1 device control by tool calls, 2 sensor interpretation |
| `link` | Radio link profile of the device |
| `deadline_s` | Deadline of the request in seconds |
| `ok` | 1 when the request completed |
| `latency_s` | Latency at the device in seconds |
| `deadline_violation` | 1 when the request missed its deadline |
| `exceed` | 1 when the latency exceeds the target of the family (tail exceedance flag) |
| `q` | Quality score of the reply |
| `e_norm` | Normalized device energy of the request as logged |
| `e_norm_const` | Normalized device energy with the constant wire overhead |
| `served_by` | Serving tier (edge or cloud) |
| `escalated` | 1 when the request was escalated to the cloud after an edge attempt |
| `cloud_used` | 1 when the cloud served the request |

### `campaign_settling.csv`

Settling after a configuration switch, per switch direction and outcome. **Rows:** 54.

| column | meaning |
|---|---|
| `group` | Switch group: arm entered (into) or left (out of) |
| `series` | Series: cloud_queue, cloud_share, deadline_miss, mean_latency_s, tail_f1, tail_f2 |
| `n_blocks` | Node-blocks that begin with a switch in the group |
| `n_rounds` | Rounds with such a block |
| `amplitude` | Fitted exponential amplitude of the transient |
| `chi2_gain` | Chi-square gain of the exponential fit over the null |
| `max_abs_t_epochs_1_to_29` | Largest absolute t of the deviation from the plateau over epochs 1 to 29 |
| `joint_chi2_gain` | Gain of the joint fit of the group |
| `joint_identified` | 1 when the t_bar interval is identified |
| `tbar_ci_lo` | Lower end of the 95 percent interval of the fitted t_bar (epochs) |
| `tbar_ci_hi` | Upper end (60 means no upper limit identified) |
| `tbar_ci_hi_conservative` | Conservative upper end |

### `campaign_settling_curves.csv`

Deviation from the plateau against the epoch after a switch, pooled over the node-blocks that begin with a switch. **Rows:** 1,620.

| column | meaning |
|---|---|
| `group` | Switch group |
| `series` | Series |
| `epoch` | Epoch since the switch (0 to 29) |
| `mean_deviation` | Mean deviation from the block plateau at the epoch |
| `n_blocks` | Blocks contributing |
| `se` | Standard error when stored |

### `campaign_settling_summary.csv`

Summary of the settling analysis, one field per row. **Rows:** 11.

| column | meaning |
|---|---|
| `key` | Field of SETTLING.json |
| `value` | Value (JSON for structured fields) |

### `campaign_sufficiency.csv`

State sufficiency of the arm for the node-block window mean, per arm and outcome. **Rows:** 24.

| column | meaning |
|---|---|
| `arm` | Arm |
| `outcome` | Outcome |
| `rounds` | Rounds (clusters) |
| `backlog_coef` | Coefficient of the cloud backlog at block start |
| `backlog_se` | Round-clustered standard error |
| `backlog_p` | p-value |
| `backlog_p_holm` | Holm-adjusted p-value over outcomes |
| `composition_stat` | Wald statistic of the previous-round arm columns (composition) |
| `composition_p` | p-value |
| `composition_p_holm` | Holm-adjusted p-value |
| `sufficient` | 1 when neither Holm p-value is below alpha = 0.05 |

### `campaign_trajectory_best.csv`

Certificate over the campaign: the worst certified bound over budget of the best certifiable mixture of S, of the incumbent and of each arm, at every 5 rounds. **Rows:** 77.

| column | meaning |
|---|---|
| `k` | Completed rounds of the prefix |
| `best_mixture_worst_ratio` | Smallest common scale f of all budgets at which the A5'' program of the look has a solution (bisection): the worst certified bound over budget that the best mixture of S attains; at most 1 means a policy is certified at the planned budgets; empty when no mixture is feasible below 4 times the budgets |
| `incumbent_worst_ratio` | Largest (upper + eta_dep) / budget of the incumbent over the seven cost rows |
| `worst_ratio_<arm>` | The same for each arm of S alone |
| `switch_planned` | 1 when the gated rule certifies a policy at the planned budgets |

### `campaign_trajectory_bounds.csv`

Certificate over the campaign: certified bounds of every arm and outcome at every 5 rounds (the program re-run on round prefixes). **Rows:** 3,696.

| column | meaning |
|---|---|
| `k` | Completed rounds of the prefix (20 to 395 in steps of 5, plus 230 and 403) |
| `band` | `planned` (planned band, t_bar = 5, decides) or tbar8 (sensitivity) |
| `arm` | Arm of S |
| `outcome` | Outcome |
| `center` | anytime-valid center |
| `radius` | anytime-valid radius including the carryover band |
| `upper` | center plus radius |
| `eta_dep` | Deployment-law term of the look |
| `upper_cert` | upper plus eta_dep: the certified bound that enters the program (cost rows; under A5'' the coupling slack of the cost rows is zero) |
| `budget` | Budget of the row |

### `campaign_trajectory_decision.csv`

Certificate over the campaign: decision, status of the program, weights and quality cost bound at every 5 rounds. **Rows:** 154.

| column | meaning |
|---|---|
| `k` | Completed rounds of the prefix |
| `band` | `planned` or `tbar8` (carryover band) |
| `switch` | 1 when the gated rule finds a certified policy |
| `lp_status` | Status of the program |
| `applied` | Program that decided (A5'' or a5pp_strict) |
| `w_<arm>` | Weight of each arm of S in the certified policy |
| `quality_cost_bound` | Certified quality cost bound when a policy exists |
| `nonincumbent_mass` | Mass outside the incumbent |
| `eta_dep` | Deployment-law term |
| `t_star` | Uniform margin of the A5'' program (planned band only): largest t with every budget lowered by t still feasible |
| `binding` | Binding rows at t_star |

### `cascade_edge_share.csv`

Share of family 2 requests of the campaign served at the edge, per arm and device class. **Rows:** 12.

| column | meaning |
|---|---|
| `arm` | Campaign arm |
| `class` | Device class or all |
| `requests_f2` | Family 2 requests |
| `share_edge` | Served at the edge |
| `share_escalated_after_edge` | Escalated after an edge attempt |

### `classc_rows.csv`

Class C deadline rows added to the certified program (offline sensitivity): per-arm center, radius and upper bound at 403 rounds. **Rows:** 6.

| column | meaning |
|---|---|
| `row` | Y_D_C (class C deadline misses) or Y_D_C1 (class C device control, 5 s) |
| `meaning` | Plain name |
| `arm` | Arm of S |
| `center, radius, upper` | anytime-valid center, radius and center plus radius at 403 rounds |
| `upper_cert` | upper plus eta_dep |
| `eta_dep, level_L` | Deployment term and the one-sided level for 9 outcomes |

### `classc_sweep.csv`

Class C deadline rows added to the certified program (offline sensitivity): certified and plug-in decisions per row and budget. **Rows:** 94.

| column | meaning |
|---|---|
| `row` | Added row |
| `budget` | Budget of the added row (the other budgets keep their planned values) |
| `certified` | 1 when the gated rule certifies a policy |
| `applied, lp_status` | Program that decided and its status |
| `w_<arm>` | Certified weights |
| `quality_cost_bound` | Certified quality cost bound |
| `cloud_share_center, new_row_center` | Cloud share and the added row of the certified policy at the centers |
| `binding` | Rows at zero slack |
| `plugin_found, plugin_w_<arm>, plugin_quality, plugin_cloud_share` | Plug-in program at the means |

### `context_members.csv`

Member means of S in each load context outside the certified one, with the standard error across node-blocks. **Rows:** 72.

| column | meaning |
|---|---|
| `level, context` | Run (hi = capacity minus 25 percent, lo = plus 25 percent, mps) and its name |
| `arm` | Member of S |
| `outcome` | Outcome |
| `mean, se` | Mean over the usable node-blocks of the member and its standard error across node-blocks |
| `node_blocks` | Usable node-blocks |
| `budget, mean_over_budget` | Budget and the mean over it |

### `context_programs.csv`

The plug-in program and the certified program applied to the node-blocks of each load context. **Rows:** 6.

| column | meaning |
|---|---|
| `level, context` | Run and its name |
| `program` | plug-in (point estimates) or certified (own radii) |
| `rounds, usable_node_blocks` | Rounds and usable node-blocks |
| `lp_status, policy_found` | Status of the program; 1 when a policy is found (plug-in) or certified (certified) |
| `w_<arm>` | Weights of the policy |
| `level_<outcome>` | Levels of the plug-in policy at the means |
| `best_mixture_scale_at_means` | Smallest common scale of all budgets at which a mixture of S meets every budget at the means (at most 1: a mixture meets the planned budgets) |
| `applied, eta_dep, radius_<outcome>_max` | Program that decided, deployment term, largest radius of the cloud share, family 1 tail and quality |
| `reason` | Decision reason |

### `e1e5_ciara_members.csv`

The members of S that the arm CIARA drew in the comparison run. **Rows:** 3.

| column | meaning |
|---|---|
| `member` | Controller arm that the CIARA arm ran in a node-block (CIARA\|I1-E is the incumbent member) |
| `weight` | Certified weight of the member (day-7 look) |
| `node_blocks` | Usable node-blocks in which the member ran |
| `mock` | Flag column, 0 in every row |

### `e1e5_ciara_vs_baselines.csv`

CIARA minus each baseline in the comparison run, per outcome. **Rows:** 80.

| column | meaning |
|---|---|
| `baseline` | Baseline arm of the comparison run |
| `outcome` | Outcome |
| `diff` | Mean of the CIARA arm minus mean of the baseline (same run; negative is better for a cost row, positive for quality) |
| `se` | Standard error of the difference (arms treated as independent) |
| `mock` | Flag column, 0 in every row |

### `e1e5_estimates.csv`

Arm estimates of the comparison run: mean and standard error per arm and outcome. **Rows:** 88.

| column | meaning |
|---|---|
| `arm` | Controller or baseline arm of the comparison run (I1-E, I1-S, I1-oracle, I2, I3, RecServe, RecServe-ft, CB, the attribution arms I1-E\|c32, I1-E\|f1prio and CIARA, the certified policy as an arm of the same run) |
| `outcome` | Outcome |
| `mean` | Mean over node-blocks (headline energy rows in constant units) |
| `se` | Standard error (round means) |
| `node_blocks` | Node-blocks of the arm |
| `budget` | Budget of the row |
| `within_budget` | 1 when the mean is within the budget |
| `mock` | Flag column, 0 in every row |

### `e1e5_h1_inversions.csv`

Pairs of controllers ranked differently by quality alone and by the budgets. **Rows:** 8.

| column | meaning |
|---|---|
| `quantity` | table_vs_certified or log_vs_certified: items, pairs, inversions, tau |
| `value` | Value |

### `e1e5_h1_members.csv`

Members used for the ranking analysis of the comparison run. **Rows:** 2.

| column | meaning |
|---|---|
| `member` | Static member of S |
| `i1_table` | Quality predicted by the I1 profile table |
| `incumbent_logs` | Quality estimated from incumbent logs (empty: no log observation, an instance of Proposition 1) |
| `randomized` | Certified campaign center |
| `mock` | Flag column, 0 in every row |

### `e1e5_h1_rows.csv`

Ranking rows: level of each controller or certified arm per outcome. **Rows:** 15.

| column | meaning |
|---|---|
| `name` | Controller, baseline or certified row |
| `source` | comparison run (measured) or certified (campaign estimate) |
| `quality` | Quality level |
| `se` | Standard error (measured rows) |
| `lower` | Certified interval lower end (certified rows) |
| `upper` | Certified interval upper end |
| `feasible` | 1 when the row meets every budget at the certificate standard |
| `mock` | Flag column, 0 in every row |

### `e1e5_latency_parts.csv`

Mean and median latency components per arm and family of the comparison run (completed requests). **Rows:** 22.

| column | meaning |
|---|---|
| `arm, family` | Arm and request family |
| `requests_ok` | Completed requests |
| `mean_<part>, median_<part>` | Mean and median of each latency component (dev, up, edge_wait, edge_serve, cloud) and of latency_s, in seconds |

### `e1e5_node_blocks.csv`

Node-block outcomes of the comparison run (one row per node-block). **Rows:** 476.

| column | meaning |
|---|---|
| (all columns of `campaign_node_blocks.csv` except `round`, `complete_round` and `in_day4_look`) | Same meaning as in `campaign_node_blocks.csv` |
| `run_id` | Run of the node-block |
| `node_index` | Index of the node |
| `window` | Window of the block: `measure` |
| `t_lo_mono_ns`, `t_hi_mono_ns` | Start and end of the window on the monotonic clock (ns) |
| `seed` | Seed of the allocation |
| `Y_E_A_measured`, `Y_E_B_measured`, `Y_E_C_measured` | Device energy with the measured wire overhead |

### `e1e5_other.csv`

Remaining results of the comparison run, flattened to key paths (contrasts with the incumbent, ablations, tuning of the bandit, table of the oracle selector). **Rows:** 280.

| column | meaning |
|---|---|
| `path` | Key path in result.json |
| `value` | Value (Proposition 1 instance, contrasts with a0, ablations, CB tuning, oracle table, ranking) |
| `mock` | Flag column, 0 in every row |

### `e1e5_requests_reduced.csv.gz`

**Rows:** 138,676.

| column | meaning |
|---|---|
| `node, round, epoch` | Node, block (round) and epoch of the request |
| `arm` | Arm of the block; CIARA blocks carry the member of S they drew (q35-... or CIARA\|I1-E) |
| `member` | Arm id logged by the gateway |
| `class, family, link, deadline_s` | Device class, request family, radio link profile, deadline |
| `ok, timeout` | Completed; reached the client limit |
| `latency_s, deadline_violation, exceed, q` | Latency at the device, deadline miss, tail exceedance flag, quality score |
| `served_by, escalated, cloud_used` | Serving tier, escalated after an edge attempt, cloud used |
| `completion_tokens, edge_attempt_tokens` | Tokens of the reply and of the edge attempt |
| `dev, up, edge_wait, edge_serve, cloud` | Latency components in seconds (empty when the stage was not reached) |

### `e1e5_switching.csv`

Configuration switches and switch pause per arm of the comparison run, with the timeout share. **Rows:** 11.

| column | meaning |
|---|---|
| `arm` | Arm of the comparison run (CIARA members mapped to CIARA) |
| `node_blocks` | Node-blocks |
| `measured_hours` | Hours of measurement windows |
| `switches_per_hour` | Configuration switches within measurement windows per hour |
| `paused_share` | Share of measured time paused by switches |
| `timeout_share` | Client timeouts over the arm's requests |

### `e1e5_timeouts.csv`

Client timeouts per arm of the comparison run. **Rows:** 11.

| column | meaning |
|---|---|
| `arm` | Arm of the comparison run (CIARA counts its members together) |
| `requests` | Requests of the run that complete in the arm's blocks |
| `timeouts` | Client timeouts at the 60 s limit, kept in the outcomes as measured |
| `share` | timeouts / requests (a defect above 0.10) |
| `other_failures` | Failed requests that are not client timeouts |
| `mock` | Flag column, 0 in every row |

### `e2_cells.csv`

Bias and coverage of the campaign intervals against the hold-one-configuration reference levels, per arm and outcome. **Rows:** 48.

| column | meaning |
|---|---|
| `look_day` | Look whose estimate is checked |
| `arm` | Arm |
| `outcome` | Outcome |
| `bias` | Campaign estimate minus the hold-one-configuration reference level |
| `radius` | Planned radius |
| `width` | Interval width |
| `zeta` | Coupling term |
| `ref` | Reference level |
| `ref_se` | Standard error of the reference |
| `covered_radius` | 1 when \|bias\| is within the radius |
| `covered_radius_zeta` | Within radius plus zeta |
| `covered_radius_2se` | Within radius plus two reference standard errors |
| `mock` | Flag column, 0 in every row |

### `e2_coverage.csv`

Number of covered cells at the nominal level. **Rows:** 2.

| column | meaning |
|---|---|
| `look_day` | Look |
| `cells` | Cells checked |
| `coverage_radius` | Share covered by the radius |
| `coverage_radius_zeta` | Share covered by radius plus zeta |
| `coverage_radius_2se` | Share covered by radius plus two reference standard errors |
| `mean_abs_bias` | Mean absolute bias |
| `mean_width` | Mean interval width |
| `mock` | Flag column, 0 in every row |

### `e2_coverage_by_level.csv`

Share of the 24 cells covered against the nominal simultaneous level (the radius recomputed at each level). **Rows:** 14.

| column | meaning |
|---|---|
| `delta` | Simultaneous miss level |
| `nominal_level` | 1 - delta |
| `level_L` | Log level of the radius |
| `cells` | Cells checked (24) |
| `covered_planned` | Cells with \|center - reference-run level\| <= radius |
| `share_planned` | Share covered |
| `covered_standardized` | Same with the link-standardized energy references |
| `share_standardized` | Share covered, link-standardized |

### `e2_coverage_standardized.csv`

Coverage of the cells with the reference levels and with the link-standardized reference levels of the device energy rows. **Rows:** 2.

| column | meaning |
|---|---|
| `look_day` | Look |
| `cells` | Cells of the comparison (3 members x 8 outcomes) |
| `energy_cells` | Of them the device energy cells (3 members x classes A, B, C) |
| `covered_radius` | Cells covered by the radius, reference |
| `covered_radius_std` | Same with the link-standardized reference on the energy cells (the other cells unchanged) |
| `coverage_radius` | Share covered (equals e2_coverage.csv) |
| `coverage_radius_std` | Share covered, energy cells link-standardized |
| `covered_radius_2se` | Cells covered by radius plus two reference standard errors |
| `covered_radius_2se_std` | Same, energy cells standardized (with the standard error of the standardized level) |
| `coverage_radius_2se` | Share |
| `coverage_radius_2se_std` | Share, energy cells standardized |

### `e2_energy_a5pp_check.csv`

Within-link differences of device energy between the reference runs and the campaign, against the sign of condition A5''. **Rows:** 18.

| column | meaning |
|---|---|
| `arm` | Arm |
| `class` | Device class |
| `link` | Link |
| `campaign` | Mean e_norm of the arm on the link in the campaign (siblings follow the balanced design) |
| `e2` | Same in the reference runs (all four nodes on the arm) |
| `diff` | e2 minus campaign |
| `se` | Bootstrap standard error |
| `z` | diff / se |
| `sibling_cloud_design` | Cloud share of the siblings under the design (mean of the three siblings' cloud-share centers) |
| `sibling_cloud_e2` | Cloud share of the siblings in the reference runs (the arm's own cloud share) |
| `predicted_sign` | Sign of diff under A5'' (cost does not fall when the siblings' load rises) |
| `sign_as_predicted` | 1 when diff has the predicted sign |
| `contradiction_2se` | 1 when diff has the opposite sign and \|z\| > 2 |

### `e2_energy_decomposition.csv`

Decomposition of the gap in device energy between the reference runs and the campaign (constant units), by member and class. **Rows:** 144.

| column | meaning |
|---|---|
| `arm` | Member held on all four nodes in the reference runs |
| `class` | Device class |
| `quantity` | campaign_level and e2_level (mean over node-blocks of the class mean of e_norm in constant units: the estimator of the campaign), campaign_at_design_mix and e2_at_design_mix (link means weighted with the configured link mix 0.70/0.30, 0.75/0.25, 0.60/0.40), gap (reference runs minus campaign), mix_devices (device-to-link mix of the schedule), mix_requests (request intensity per device and link: rate multipliers, time slice of the arrival process), mix_total, within_total, within_payload (bytes of the exchange), within_lat_queue (edge queue wait), within_lat_edge (edge attempt), within_lat_cloud (cloud service), within_lat_other (rest of the device latency: radio transfer and link delay), within_cpu, within_clip_failed (clipping at 2 E_d and requests without a latency) |
| `value` | Contribution to the gap (Kitagawa two-way decomposition at the mean weights and means: exact, no order effect) |
| `se` | Standard error (bootstrap over rounds of the campaign and blocks of the hold, 1000 resamples) |
| `share_of_gap` | value / gap |

### `e2_energy_link_cells.csv`

Per link request statistics of the campaign and of the reference runs. **Rows:** 36.

| column | meaning |
|---|---|
| `source` | campaign (day 7, complete rounds, arm's own node-blocks) or e2 (hold of the arm without its first block) |
| `arm` | Arm |
| `class` | Device class |
| `link` | Link profile |
| `requests` | Requests in the measurement window |
| `request_share` | Share of the class's requests of the arm on the link |
| `design_share` | Share of the arm's exposure on the link by the schedule (sum of device rate multipliers) |
| `requests_per_device_epoch` | Requests per device and measured epoch on the link |
| `e_norm` | Mean e_norm (constant units) |
| `energy_mj` | Mean device energy per request (mJ, rebuilt with the model) |
| `radio_mj` | Radio term (mJ) |
| `wait_mj` | Wait term (mJ) |
| `bytes_up` | Application bytes up |
| `bytes_down` | Application bytes down |
| `device_latency_s` | L = last byte minus send at the device (s), the latency of the energy model |
| `queue_s` | Edge queue wait inside L (s, mean over all requests) |
| `edge_s` | Edge attempt inside L (s) |
| `cloud_s` | Cloud service inside L (s, mean over all requests) |
| `cloud_service_s` | Cloud service time per cloud-served request (s) |
| `cloud_served_share` | Share served by the cloud |
| `escalated_share` | Share escalated after an edge attempt |
| `latency_outcome_s` | Latency of the outcomes, from the arrival (s) |
| `pre_send_s` | Device preparation and connection set-up before the send (s) |

### `e2_energy_standardized.csv`

Reference levels of the device energy rows and their link-standardized form. **Rows:** 18.

| column | meaning |
|---|---|
| `look_day` | Look |
| `arm` | Arm |
| `outcome` | Energy outcome |
| `center` | Campaign center (planned band) |
| `radius` | Planned radius |
| `ref` | Reference level of the validity analysis (node-block mean) |
| `ref_se` | Its standard error over rounds |
| `ref_std` | Link-standardized reference: reference plus (reference runs link means at the campaign request mix minus the reference runs request-weighted mean); labeled sensitivity |
| `ref_std_se` | Standard error of the standardized level over the blocks of the hold |
| `bias` | center minus ref |
| `bias_std` | center minus ref_std |
| `covered_radius` | Planned coverage by the radius |
| `covered_std_radius` | Coverage by the radius with the standardized reference |
| `covered_radius_2se` | Planned coverage by radius plus two se |
| `covered_std_radius_2se` | Standardized coverage by radius plus two se |
| `adjustment` | ref_std minus ref |
| `campaign_mix_poor_share` | Campaign request share on the poor link (nbiot_cl1, ltem_mid, nr_typical) |
| `e2_mix_poor_share` | reference runs request share on the poor link for the arm |

### `e4_contrast.csv`

Held-out deployment: quality contrast of the certified policy against the incumbent and its certified cost bound. **Rows:** 1.

| column | meaning |
|---|---|
| `quantity` | quality_C_minus_I |
| `value` | Certified minus incumbent quality in held-out deployment |
| `se` | Standard error |
| `certified_cost_bound` | Certified upper bound on J_Q(a0) minus J_Q(pi) at the final look |
| `measured_cost` | Measured J_Q(a0) minus J_Q(pi) |
| `cost_within_bound` | 1 when the measured cost is within the bound (plus one standard error) |

### `e4_energy_cells.csv`

Held-out deployment: device energy per round and class under the logged, measured and constant wire overhead. **Rows:** 48.

| column | meaning |
|---|---|
| `node` | Edge node |
| `round` | Round of the held-out deployment |
| `window` | measure or burn_in |
| `class` | Device class |
| `n` | Requests with an energy row |
| `e_logged` | Mean e_norm as logged (measured overhead) |
| `e_measured_rescored` | Rescored with the measured overhead (check: equals e_logged to 1e-8) |
| `e_constant` | Rescored with the constant overhead 1.0456 |

### `e4_energy_decomposition.csv`

Held-out deployment: decomposition of the gap in device energy against the campaign into timing, link draw and wire overhead. **Rows:** 16.

| column | meaning |
|---|---|
| `class` | Device class (A or B) |
| `arm` | Arm (incumbent I1-E) |
| `quantity` | Sequential steps from the campaign to held-out deployment measured (values), the gap, and its parts: timing, link draw, wire overhead |
| `value` | Energy outcome or energy difference |

### `e4_energy_link_adjustment.csv`

Held-out deployment: device energy re-weighted to the request mix of the campaign. **Rows:** 6.

| column | meaning |
|---|---|
| `policy` | C certified policy, I incumbent |
| `class` | Device class |
| `unadjusted_pooled` | Constant-unit mean pooled over requests |
| `adjusted` | Constant-unit mean re-weighted to the (class, link) request mix of the campaign (descriptive sensitivity, not a planned rule) |
| `certified_upper` | Certified upper bound |
| `adjusted_within` | 1 when the adjusted value is within the bound |

### `e4_estimates.csv`

Held-out deployment: level of each policy and outcome (mean over node-blocks) in constant and measured wire-overhead units. **Rows:** 22.

| column | meaning |
|---|---|
| `policy` | C certified policy, I incumbent |
| `outcome` | Outcome |
| `unit` | constant (headline) or measured overhead (energy rows only; other outcomes have one unit) |
| `mean` | Mean over the node-blocks of the policy |
| `se` | Standard error from the round means (one degree of freedom with two rounds) |
| `rounds` | Rounds of the policy |
| `node_blocks` | Node-blocks of the policy |
| `budget` | Budget of the row |
| `within_budget` | 1 when the mean is within the budget |
| `certified_upper` | Certified upper bound of the policy (campaign look, constant units); empty for the incumbent |
| `within_certified_upper` | 1 when the mean is within the certified upper bound |

### `e4_node_blocks.csv`

Held-out deployment: node-block outcomes (4 rounds of 1,200 epochs, contexts I, C, C, I). **Rows:** 16.

| column | meaning |
|---|---|
| `round` | Round of the held-out deployment (1,200 epochs, four nodes in parallel) |
| `node` | Edge node |
| `block_id` | Block index (equals round) |
| `arm_id` | Arm played (the certified policy draws arms per node-block) |
| `policy` | Policy context of the round: C (certified policy) or I (incumbent) |
| `propensity` | Arm probability under the context |
| `block_status` | complete or partial |
| `n` | Requests |
| `Y_Q` | Quality |
| `Y_L_f1` | Tail exceedance, family 1 |
| `Y_L_f2` | Tail exceedance, family 2 |
| `Y_D` | Deadline-miss share |
| `Y_C` | Cloud share |
| `Y_E_A` | Energy outcome of class A in constant wire-overhead units (headline; units of E_d and of the certificate) |
| `Y_E_B` | Class B, constant units |
| `Y_E_C` | Class C, constant units |
| `Y_E_A_measured` | Class A with the overhead measured by the tc counters of the run (sensitivity) |
| `Y_E_B_measured` | Class B, measured overhead |
| `Y_E_C_measured` | Class C, measured overhead |
| `n_failed` | Failed requests |
| `n_cloud` | Cloud-served requests |
| `lat_p50_s` | Median latency (s) |
| `lat_p95_s` | 95th percentile latency (s) |
| `gpu_throttled_frac` | Share of throttled epochs |

### `e4_requests_reduced.csv.gz`

Requests of the measurement windows of the held-out deployment, one row per request, with the same columns as `campaign_requests_reduced.csv.gz`. **Rows:** 247,388.

| column | meaning |
|---|---|
| `node` | Edge node (E1 to E4) |
| `round` | Round of the held-out deployment (1,200 epochs) |
| `epoch` | Epoch of the request (30 s epochs) |
| `arm` | Arm of the block |
| `class` | Device class A, B or C (NB-IoT sensors, LTE-M actuators, 5G cameras) |
| `family` | Request family: 1 device control by tool calls, 2 sensor interpretation |
| `link` | Radio link profile of the device |
| `deadline_s` | Deadline of the request in seconds |
| `ok` | 1 when the request completed |
| `latency_s` | Latency at the device in seconds |
| `deadline_violation` | 1 when the request missed its deadline |
| `exceed` | 1 when the latency exceeds the target of the family (tail exceedance flag) |
| `q` | Quality score of the reply |
| `e_norm` | Normalized device energy of the request as logged (measured wire overhead) |
| `e_norm_const` | Normalized device energy rescored with the constant wire overhead |
| `served_by` | Serving tier (edge or cloud) |
| `escalated` | 1 when the request was escalated to the cloud after an edge attempt |
| `cloud_used` | 1 when the cloud served the request |

### `e4_timeseries.csv`

Held-out deployment hour by hour over the 40 hours (four rounds of 10 hours: incumbent, certified, certified, incumbent). **Rows:** 40.

| column | meaning |
|---|---|
| `policy` | incumbent (rounds 0 and 3) or certified (rounds 1 and 2) |
| `round` | Round of the held-out deployment |
| `hour` | Hour of the 40-hour deployment |
| `hour_mid` | Hour midpoint |
| `requests` | Requests of the hour |
| `Y_C` | Cloud share |
| `requests_f1` | Family 1 requests |
| `Y_L_f1` | Family 1 tail exceedance |
| `Y_D` | Deadline misses |
| `requests_q` | Requests of the hour with a quality score |
| `Y_Q` | Mean quality of the hour (input of `trace_quality`) |

### `e4_wire_overhead.csv`

Wire overhead measured by the traffic counters of the held-out deployment. **Rows:** 3.

| column | meaning |
|---|---|
| `class` | Device class |
| `up` | Mean measured wire overhead uplink (layer 2 bytes per application byte, tc counters of the run) |
| `down` | Downlink |
| `devices` | Devices with counters |
| `constant` | Constant overhead of the energy table (headline) |

### `e5off_policies.csv`

Offline baselines (plug-in program, CIARA without the energy row): policies and status of their programs. **Rows:** 3.

| column | meaning |
|---|---|
| `policy` | Offline policy (ciara as decided, plug_in_lp, ciara_without_energy) |
| `lp_status` | Status of its program on the final look |
| `realized_defined` | 1 when the realized levels exist (needs the hold-one-configuration levels) |
| `mass_I1_E` | Mass on I1-E |
| `mass_c32_prio` | Mass on c32/priority |
| `mass_c32_FIFO` | Mass on c32/FIFO |

### `e5off_realized.csv`

Offline baselines: policy-weighted levels at the reference levels. **Rows:** 24.

| column | meaning |
|---|---|
| `policy` | ciara, plug_in_lp or ciara_without_energy |
| `outcome` | Outcome |
| `level` | Policy-weighted hold-one-configuration level |
| `se` | Standard error |
| `violated` | 1 when the level exceeds the budget |
| `violated_with_margin` | 1 when it exceeds the budget by more than two standard errors |
| `mock` | Flag column, 0 in every row |

### `e8v9_contrast.csv`

Load contexts outside the certified one (capacity minus and plus 25 percent, GPU sharing mode): levels and contrast of the certified policy against the incumbent. **Rows:** 32.

| column | meaning |
|---|---|
| `level` | lo (capacity +25 percent), hi (-25 percent), mps (MPS cross-check) or campaign |
| `outcome` | Outcome |
| `J_pi` | Level of the certified policy |
| `J_a0` | Level of the incumbent |
| `diff` | Policy minus incumbent |
| `se_diff` | Standard error of the difference |
| `node_blocks` | Node-blocks |
| `quality_sign_as_campaign` | 1 when the sign of the quality contrast equals the campaign sign |
| `same_budget_status_as_campaign` | 1 when the budget status equals the campaign status |
| `mock` | Flag column, 0 in every row |

### `e8v9_node_blocks.csv`

Node-block outcomes of the three load-context runs (one row per node-block). A node-block with status `excluded` did not pass the run-validity check fixed before the run and does not enter the analysis; the row stays in the file. **Rows:** 220.

| column | meaning |
|---|---|
| `level` | Run: `hi` (capacity plus 25 percent), `lo` (capacity minus 25 percent) or `mps` (GPU sharing mode) |
| (all columns of `e1e5_node_blocks.csv` except `Y_*_measured`) | Same meaning as in `campaign_node_blocks.csv` and `e1e5_node_blocks.csv` |
| `block_status` | `complete` or `excluded` |
| `block_status_before_exclusion` | Status of an excluded node-block before the exclusion |
| `excluded` | `True` for a node-block that did not pass the run-validity check fixed before the run; the node-block does not enter the analysis |

### `e9_contrast.csv`

Second model family (Ministral 3): deployment contrast. **Rows:** 8.

| column | meaning |
|---|---|
| `outcome` | Outcome |
| `candidate` | Candidate level |
| `candidate_se` | Standard error |
| `incumbent` | Incumbent level |
| `incumbent_se` | Standard error |
| `diff` | Candidate minus incumbent |
| `diff_se` | Standard error of the difference |
| `budget` | Budget of the row |
| `mock` | Flag column, 0 in every row |

### `e9_inversion.csv`

Second model family: pairs of configurations ordered differently by the isolated profile and by the randomized measurement. **Rows:** 1.

| column | meaning |
|---|---|
| `inversions` | Inverted pairs |
| `pairs` | Pairs |
| `share` | Inversion share |
| `ci95_lo` | 95 percent interval of the share, lower |
| `ci95_hi` | Upper |
| `reps` | Bootstrap replicates |
| `edge_set` | Edge set of the replicate (int8-int4) |
| `mock` | Flag column, 0 in every row |

### `e9_ranks.csv`

Second model family: rank of each configuration by the isolated profile and by the randomized measurement. **Rows:** 6.

| column | meaning |
|---|---|
| `item` | Ministral 3 configuration |
| `i1_rank` | Rank under the I1 table |
| `measured_rank` | Measured rank |
| `mock` | Flag column, 0 in every row |

### `energy_device_by_class.csv`

Device energy by class with the coefficient band and the wire-overhead sensitivities. **Rows:** 612.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |
| `class` | Device class A (NB-IoT), B (LTE-M), C (5G) |
| `variant` | `nom` nominal coefficients, constant wire overhead (headline); `low`, `high` all-low and all-high coefficients; `cls` wire overhead measured per class (held-out deployment tc counters); `dev` wire overhead measured per device (held-out deployment and reference runs only); `mix_design`, `mix_campaign`, `mix_good`, `mix_poor` nominal coefficients recombined with the configured link mix, the campaign mix, all devices on the good or on the poor link; `own_poor_share` share of requests on the poor link |
| `level` | Device energy Y_E of the class in units of the certificate (share of 2 E_d); mean over blocks of the node-block class means |
| `se` | Standard error over blocks (descriptive; two rounds in held-out deployment) |
| `blocks` | Blocks (rounds) of the unit |
| `n_requests` | Requests of the unit |
| `mj_per_request` | Mean device energy per request in mJ (nominal, low or high coefficients) |
| `poor_share` | Share of the poor link in the target mix |
| `planned_center` | Estimate of the same cell (campaign look center, reference-run level, held-out deployment estimate) |
| `radius` | anytime-valid radius of the campaign interval (planned band, day 7) |
| `certified_upper` | Certified upper bound (campaign day 7 arm upper bound or policy upper bound; held-out deployment policy C) |
| `budget` | Budget of the class (0.5) |

### `energy_device_cells.csv.gz`

Device energy per node-block and class (base of `energy_device_by_class.csv`). **Rows:** 13,800.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `node` | Node E1 to E4 |
| `block` | Block (round) index |
| `arm` | Arm id (`*` for scalars) |
| `class` | Device class A (NB-IoT), B (LTE-M), C (5G) |
| `link` | Link profile of the device |
| `n` | Requests |
| `log_norm` | Sum of e_norm as logged |
| `bytes_up` | Application bytes up |
| `bytes_down` | Application bytes down |
| `radio_mj` | Nominal radio energy, sum (mJ) |
| `wait_mj` | Nominal wait energy, sum (mJ) |
| `cpu_mj` | Nominal device CPU energy, sum (mJ) |
| `nom_norm` |  |
| `nom_mj` |  |
| `nom_n` |  |
| `low_norm` |  |
| `low_mj` |  |
| `low_n` |  |
| `high_norm` |  |
| `high_mj` |  |
| `high_n` |  |
| `cls_norm` |  |
| `cls_mj` |  |
| `cls_n` |  |
| `dev_norm` |  |
| `dev_mj` |  |
| `dev_n` |  |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |

### `energy_device_check.csv`

Reproduction of the estimates of the campaign, the reference runs, the held-out deployment and the comparison run by the device-energy model. **Rows:** 24.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |
| `class` | Device class A (NB-IoT), B (LTE-M), C (5G) |
| `level` | Device energy Y_E of the class in units of the certificate (share of 2 E_d); mean over blocks of the node-block class means |
| `planned_center` | Estimate of the same cell (campaign look center, reference-run level, held-out deployment estimate) |
| `diff` | level minus planned_center |
| `blocks` | Blocks (rounds) of the unit |

### `energy_e2e.csv`

End-to-end energy per request and quality, per arm. **Rows:** 21.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |
| `Y_Q` | Quality of the node-block or the unit |
| `Y_Q_se` | Standard error of the quality over blocks |
| `device_j_per_req` | Device energy per request (J, nominal, all classes, request-weighted) |
| `edge_j_per_req` | Edge joules per request |
| `cloud_j_per_req` | Cloud joules per request |
| `edge_idle_j_per_req` | Edge joules per request, idle part |
| `cloud_idle_j_per_req` | Cloud joules per request, idle part |
| `total_j_per_req` | Edge plus cloud joules per request (serving table) or device plus edge plus cloud (end-to-end table) |
| `total_j_per_req_se` | Bootstrap standard error of the serving total |
| `serving_j_per_req` | edge plus cloud joules per request |
| `active_serving_j_per_req` | serving joules per request without the idle parts |
| `device_share` | Device share of the total |

### `energy_frontier.csv`

Certified energy-budget frontier: the day-7 program re-solved offline with the device-energy budgets multiplied by a factor. **Rows:** 82.

| column | meaning |
|---|---|
| `scope` | `all`: the budgets of classes A, B and C scaled together; `A`: class A alone |
| `factor` | Factor on the device energy budgets (nominal 0.5) |
| `budget_Y_E_A` | Budget of Y_E_A in this row |
| `budget_Y_E_B` | Budget of Y_E_B |
| `budget_Y_E_C` | Budget of Y_E_C |
| `program` | Program that decided: `A5''`, or `a5pp_strict` when rho(pi_hat) > 0 (rule a5pp_gated) |
| `lp_status` | optimal or infeasible |
| `branch` | gate or budget_compliance (incumbent exceeds a budget at its upper bound) |
| `policy_exists` | 1 when a certified policy exists and deploys |
| `incumbent_violates` | Rows on which the incumbent exceeds a budget at its upper bound |
| `rows_no_arm_meets` | Rows on which no single arm meets the budget at its upper bound |
| `w_I1-E` | Certified weight of the incumbent |
| `w_q35-4b-ih-w8a8` | b8\|c32\|f1prio` \| Certified weight of c32/priority |
| `w_q35-4b-ih-w8a8` | b8\|c32\|fifo` \| Certified weight of c32/FIFO |
| `nonincumbent_mass` | Weight outside the incumbent |
| `quality_cost_bound` | Certified upper bound on the quality cost against the incumbent |
| `objective_lower_bound` | Lower bound on the certified quality objective |
| `gate_value` | Value of the paired gate |
| `ub_Y_E_A` | Certified upper bound of the policy on Y_E_A |
| `ub_Y_E_B` | Same, Y_E_B |
| `ub_Y_E_C` | Same, Y_E_C |
| `slack_Y_E_A` | Budget minus the upper bound, Y_E_A |
| `slack_Y_E_B` | Same, Y_E_B |
| `slack_Y_E_C` | Same, Y_E_C |
| `binding` | Rows with zero slack at the optimum (`share:Y_C` is the escalation share constraint of A5'') |
| `reason` | Decision reason of the program |

### `energy_frontier_inputs.csv`

Inputs of the offline re-solve of the day-7 program. **Rows:** 59.

| column | meaning |
|---|---|
| `row_kind` | center, radius, scalar or stored_weight |
| `arm` | Arm id (`*` for scalars) |
| `name` | Outcome or scalar name |
| `value` | Value |

### `energy_frontier_threshold.csv`

Smallest budget factor with a certified policy. **Rows:** 2.

| column | meaning |
|---|---|
| `scope` | `all`: the budgets of classes A, B and C scaled together; `A`: class A alone |
| `factor_min` | Smallest factor with a certified policy (bisection) |
| `budget_Y_E_A` | Budget of Y_E_A in this row |
| `budget_Y_E_B` | Budget of Y_E_B |
| `budget_Y_E_C` | Budget of Y_E_C |
| `binding_at_threshold` | Binding rows at the threshold |
| `quality_cost_bound_at_threshold` | Certified quality cost at the threshold |
| `w_incumbent_at_threshold` | Incumbent weight at the threshold |
| `lp_status_just_below` | Status just below the threshold |

### `energy_gpu_audit.csv`

Audit of the GPU energy counters and idle baseline, per run and role (edge or cloud); one row per counter. **Rows:** 24.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `role` | Edge node or cloud pair |
| `n_samples` | NVML samples |
| `span_s` | Time span of the samples (s) |
| `dt_median_s` | Median sampling interval (s) |
| `dt_max_s` | Longest sampling gap (s) |
| `n_gap_gt2s` | Gaps above 2 s |
| `n_gap_gt10s` | Gaps above 10 s |
| `n_backward` | Steps of the energy counter backward (wrap or reset); 0 means monotone |
| `n_flat_steps` | Steps with unchanged counter |
| `counter_energy_j` | Energy over the span by the counter (J) |
| `power_integral_j` | Integral of the power samples over the span (J) |
| `power_integral_over_counter` | power integral over counter energy |
| `mean_power_w` | Mean power by the counter (W) |
| `util_mean` | Mean GPU utilization (percent) |
| `share_util0` | Share of samples at utilization 0 |
| `idle_w_counter` | Idle baseline: power at utilization 0 with a resident engine (counter, W) |
| `idle_w_power_samples` | Idle baseline by the power samples (W) |
| `idle_time_s` | Time at utilization 0 used for the baseline (s) |

### `energy_serving_blocks.csv`

Serving energy per node-block. **Rows:** 2,344.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `block` | Block (round) index |
| `node` | Node E1 to E4 |
| `arm` | Arm id (`*` for scalars) |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |
| `n` | Requests |
| `n_cloud` | Cloud-served requests of the node-block |
| `duration_s` | Window length (s) |
| `edge_j` | Edge GPU energy over the window (J) |
| `edge_idle_j` | Idle part (J) |
| `cloud_pair_j` | Energy of the cloud pair over the window (J) |
| `cloud_pair_idle_j` | Idle part of the cloud pair (J) |
| `Y_Q` | Quality of the node-block or the unit |
| `cloud_share` | Share of the window's cloud-served requests |
| `cloud_attr_j` | Cloud pair energy attributed to the node-block (J) |
| `cloud_attr_idle_j` | Idle part attributed (J) |
| `edge_active_j` | edge_j minus edge_idle_j |
| `cloud_attr_active_j` | cloud_attr_j minus cloud_attr_idle_j |

### `energy_serving_by_arm.csv`

Serving joules per request by arm, split into idle and active parts. **Rows:** 21.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `unit` | Arm id or policy: `I1-E` incumbent; arm ids of the members; `CIARA` (held-out deployment policy C, arm of the comparison run); `CIARA (pi-mix)` the certified weights on the member levels (modeled mix, not a measurement) |
| `blocks` | Blocks (rounds) of the unit |
| `node_blocks` | Node-blocks |
| `n_requests` | Requests of the unit |
| `edge_power_w_mean` | Mean edge GPU power over the windows (W) |
| `cloud_share_of_cloud_work` | Mean attribution share |
| `cloud_requests_share` | Share of requests served by the cloud |
| `edge_j_per_req` | Edge joules per request |
| `edge_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `edge_idle_j_per_req` | Edge joules per request, idle part |
| `edge_idle_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `edge_active_j_per_req` | Edge joules per request, active part |
| `edge_active_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `cloud_j_per_req` | Cloud joules per request |
| `cloud_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `cloud_idle_j_per_req` | Cloud joules per request, idle part |
| `cloud_idle_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `cloud_active_j_per_req` | Cloud joules per request, active part |
| `cloud_active_j_per_req_se` | Bootstrap standard error over blocks (400 resamples; empty below five blocks) |
| `total_j_per_req` | Edge plus cloud joules per request (serving table) or device plus edge plus cloud (end-to-end table) |
| `total_j_per_req_se` | Bootstrap standard error of the serving total |

### `energy_window_audit.csv`

Coverage of the measurement windows by the energy counters. **Rows:** 8.

| column | meaning |
|---|---|
| `run` | Run: `campaign` (day-7 look, 403 rounds), `e4` (held-out deployment), `e2` (hold-one-configuration), `e1e5` (controllers and CIARA arm) |
| `kind` | Window kind: edge GPU or cloud GPU |
| `windows` | Node-block windows |
| `window_s_median` | Median window length (s) |
| `samples_over_expected_min` | Smallest ratio of samples to the expected number at the sampling interval |
| `first_gap_s_max` | Largest gap from the window start to the first sample (s) |
| `last_gap_s_max` | Largest gap from the last sample to the window end (s) |
| `extrapolated_windows` | Windows whose counter was extrapolated at an edge |
| `windows_below_90pct_samples` | Windows with less than 90 percent of the expected samples |
| `power_integral_over_counter_median` | Median ratio of the power integral to the counter energy per window |
| `rel_diff_outcomes_max` | Largest relative difference between the window energy of this table and `gpu_energy_j` of the outcomes table (that one extrapolates over gaps) |
| `rel_diff_outcomes_median` | Median of the same |
| `windows_rel_diff_outcomes_gt_1pct` | Windows with a difference above 1 percent |

### `fifoonly_result.csv`

CIARA restricted to FIFO admission: result of the offline program, flattened to key paths. **Rows:** 76.

| column | meaning |
|---|---|
| `path` | Key path in steps/fifoonly/result.json |
| `value` | Value (restricted program, policies with levels, quality attributed to the admission choice) |

### `latency_quantiles.csv`

Latency at the device per run, unit and family on a grid of 291 quantiles, dense in the tail, with the share above the family target. **Rows:** 12,804.

| column | meaning |
|---|---|
| `run` | campaign, e4 (per policy), e4_class (policy\|class), e1e5 (per arm, CIARA members mapped to CIARA) |
| `unit` | Arm, policy or policy\|class |
| `family` | Request family |
| `p` | Quantile probability (grid dense in the tail) |
| `latency_s` | Quantile of the latency at the device (inf for a request at the client limit) |
| `n` | Requests |
| `share_above_target` | Share above the family target L1 = 9.504 s or L2 = 44.331 s |
| `share_exceed` | Share with the exceed flag (e4_class: share of deadline misses) |
| `target_s` | Family target |

### `level_a.csv`

Cross-family evidence at INT4: tool-call accuracy, weights, key-value cache, 95th-percentile latency and concurrency or co-location gap. **Rows:** 6.

| column | meaning |
|---|---|
| `family` | Model family (Qwen3, Ministral 3, Granite 4.2) |
| `size` | Parameter count class |
| `ckpt` | Checkpoint tag (INT4, in-house W4A16 build) |
| `quality` | Tool-call accuracy of the family 1 suite (p3 summary) |
| `weights_gib` | Weights in GiB on the E-L class (p25) |
| `kv_tokens` | Key-value cache capacity in tokens (p25) |
| `p95_c1` | p95 latency at 4,096 prompt tokens, one request (s) |
| `p95_c16` | p95 latency at 4,096 prompt tokens, 16 concurrent requests (s) |
| `gap` | Concurrency gap: p95 at 16 requests over p95 at one |
| `admissible` | Admissible on the class (p25) |
| `p95_iso` | Co-location gap inputs from levela_gap: p95 in isolation (empty until the step has run) |
| `p95_col` | p95 under co-location (levela_gap) |
| `colocation_gap` | Ratio co-location over isolation (levela_gap) |
| `colocation_gap_se` | Standard error of the ratio (levela_gap) |

### `levela_gap.csv`

Co-location gap of the cross-family table. **Rows:** 12.

| column | meaning |
|---|---|
| `ckpt` | Checkpoint (family, size, INT4) |
| `cell` | Prompt length and concurrency (L4096_c16) |
| `p95_isolation` | p95 latency in isolation (s) |
| `p95_colocation` | p95 latency under the co-location pattern of the campaign (s) |
| `gap` | Ratio colocation over isolation |
| `gap_se` | Standard error of the ratio |
| `mock` | Flag column, 0 in every row |

### `overhead.csv`

Overheads of the mechanism measured in the campaign logs (switch, gateway, controller, certification). **Rows:** 18.

| column | meaning |
|---|---|
| `id` | Row id of ciara_post.overhead |
| `label` | Row label |
| `sub_item` | 1 when the row is a component of the row above |
| `unit` | s or ms |
| `n` | Events |
| `mean` | Mean |
| `median` | Median |
| `p95` | 95th percentile |
| `p99` | 99th percentile |
| `max` | Maximum |
| `source` | Log field the row is computed from |

### `request_stats.csv`

Per-request statistics of the measurement windows of the campaign and of the held-out deployment, per population, policy, dimension and level. **Rows:** 132.

| column | meaning |
|---|---|
| `population` | campaign (day-7 data, 403 complete rounds plus the partial round) or e4 (held-out deployment) |
| `policy` | incumbent (arm I1-E in the campaign, context I in held-out deployment), certified (held-out deployment context C; campaign: arms weighted by the certified policy masses of the day-7 look), or an arm id (campaign) |
| `dim` | Grouping: overall, class (device class A, B, C), class_family (class and task family, for example A1), link (radio type and coverage level), node (edge node), family (task family) |
| `level` | Group value |
| `n` | Requests in the group (unweighted count) |
| `deadline_miss` | Share of requests that missed their deadline |
| `exceed` | Share above the latency threshold of the family (the tail outcome) |
| `p50` | Median latency at the device (s), answered requests |
| `p95` | 95th percentile latency (s), answered requests (campaign certified rows: weighted quantile) |
| `quality` | Mean quality |
| `cloud_share` | Share served by the cloud |
| `failed_share` | Share of failed requests |
| `e_norm_constant` | Mean e_norm with the constant wire overhead (the headline units) |
| `e_norm_logged` | Mean e_norm as logged (held-out deployment: measured overhead; campaign: constant, equal to e_norm_constant) |

### `screening_arms.csv`

One row per pass and arm of the first three screening passes: burn-in-discarding means and the pass-rule verdict. **Rows:** 45.

| column | meaning |
|---|---|
| `pass_id` | Screening pass or probe, in order: `pass1-pilot` (the pilot pass with five nodes), `pass2`, `pass3`, `pass4-probe`, `pass5-probe`. |
| `arm` | Arm identifier: profile\|batch cap\|router (k0, k1, kA, kH, c32 for the token-budget escalation) with an optional admission suffix (fifo, f1prio); I1-E, I2, I3 are the incumbent regimes. |
| `kind` | incumbent (I1-E, I2, I3 regime) or configuration (profile\|cap\|router arm). |
| `n_blocks` | Number of usable node-blocks of the arm. |
| `Y_Q` | Mean quality (accuracy score) of the arm. |
| `Y_L_f1` | Mean tail exceedance of family 1 (share of requests above the L1 target). |
| `Y_L_f2` | Mean tail exceedance of family 2 (share of requests above the L2 target). |
| `Y_D` | Mean deadline-miss rate. |
| `Y_E_A` | Mean normalized energy of class A (energy divided by twice the class budget E_A; the budget 0.5 equals E_A). |
| `Y_E_B` | Mean normalized energy of class B (same normalization with E_B). |
| `Y_E_C` | Mean normalized energy of class C (same normalization with E_C). |
| `Y_C` | Mean cloud share (share of requests answered by the cloud). |
| `smallest_margin` | Smallest margin over the non-waived rows (budget minus mean minus radius minus eta_dep); empty for incumbents and arms without enough blocks. |
| `smallest_margin_outcome` | Outcome row that attains the smallest margin. |
| `binding_outcome` | Binding outcome recorded by the pass rule. |
| `pass` | Pass-rule verdict at the day-7 radius. |
| `pass_oracle_radius` | Pass-rule verdict at the oracle radius. |
| `fail_reason` | Recorded reason for a failing arm (rows above budget). |
| `u` | Effective count used in the radius. |
| `L` | Union-bound level ln(\|S\| K / delta). |
| `tau_epochs` | Carryover time used in the radius. |
| `eta_dep` | Dependence allowance eta_dep. |
| `band` | Band parameter of the pass rule. |

### `screening_node_blocks.csv`

One row per node-block (measurement window) of the five screening passes; Y_* are the window means. **Rows:** 510.

| column | meaning |
|---|---|
| `pass_id` | Screening pass or probe, in order: `pass1-pilot` (the pilot pass with five nodes), `pass2`, `pass3`, `pass4-probe`, `pass5-probe`. |
| `node` | Edge node (E1 to E5). |
| `block_id` | Block index within the node. |
| `arm_id` | Arm applied in the block. |
| `kind` | Block kind (static). |
| `propensity` | Assignment propensity of the arm. |
| `design` | Assignment design (iid or balanced). |
| `seed` | Seed of the assignment. |
| `rng_draw` | Random draw used for the assignment. |
| `cfg_profile` | Model profile of the block. |
| `cfg_batch_cap` | Batch cap of the block. |
| `cfg_router` | Router of the block. |
| `profile_switched` | The block started with an engine profile switch. |
| `switch_time_s` | Measured switch time of the profile change (empty when none). |
| `block_status` | Block completion status. |
| `n` | Requests in the measurement window. |
| `Y_Q` | Mean quality (accuracy score) of the arm. |
| `Y_L_f1` | Mean tail exceedance of family 1 (share of requests above the L1 target). |
| `Y_L_f2` | Mean tail exceedance of family 2 (share of requests above the L2 target). |
| `Y_D` | Mean deadline-miss rate. |
| `Y_E_A` | Mean normalized energy of class A (energy divided by twice the class budget E_A; the budget 0.5 equals E_A). |
| `Y_E_B` | Mean normalized energy of class B (same normalization with E_B). |
| `Y_E_C` | Mean normalized energy of class C (same normalization with E_C). |
| `Y_C` | Mean cloud share (share of requests answered by the cloud). |
| `n_f1` | Family 1 requests. |
| `n_f2` | Family 2 requests. |
| `lat_p50_s` | Median latency. |
| `lat_p95_s` | p95 latency. |
| `n_failed` | Failed requests. |
| `n_cloud` | Requests answered at the cloud. |
| `gpu_energy_j` | Edge GPU energy in the window. |
| `gpu_power_w_mean` | Mean edge GPU power. |
| `gpu_util_mean` | Mean edge GPU utilization. |
| `gpu_throttled_frac` | Fraction of throttled epochs. |

### `sim_estimators.csv`

Simulation of the estimators (per-epoch randomization against block designs), one row per setting; not testbed data. **Rows:** 12.

| column | meaning |
|---|---|
| `eps`, `l`, `b`, `t_mix` | Exploration probability, block length (epochs; 1 is per-epoch randomization), burn-in (epochs) and mixing time of the carryover (epochs) of the simulated design |
| `N`, `nblocks`, `T_epochs`, `R`, `k_cell_mean` | Nodes, blocks, simulated epochs, replicates, and the mean observations per cell |
| `ht_*`, `hajek_*`, `dm_*` | Horvitz-Thompson, Hajek and cell-mean (difference of means) estimators |
| `*_bias_<outcome>_<arm>`, `*_rmse_<outcome>_<arm>` | Bias and root mean squared error for the outcome (`quality`, `exceed`) of the incumbent (`inc`) or the non-incumbent (`non`) arms; Figure 7 panels (d) and (e) use `dm_bias_exceed_non` and `dm_rmse_exceed_non` |
| `*_missing` | Share of cells without an observation |
| `cover_*`, `width_*`, `radius_*` | Coverage, width and median radius of the intervals built with the Hoeffding, Freedman and empirical Bernstein radii |
| `tag`, `design`, `K`, `X`, `drift`, `secs` | Run labels (design: per-epoch, switchback-no-burnin, safe-block, and others), simulation settings and run time |
| `profile` | str \| --- \| Engine profile identifier. |

### `validation_calibration.csv`

One row per engine profile of the calibration at full speed (condition C100), as in the throttling table of the supplement. **Rows:** 6.

| column | meaning |
|---|---|
| `profile` | Engine profile identifier. |
| `name` | Display name used in the supplement table. |
| `restarts` | Engine restarts behind the medians (condition C100). |
| `decode_c1_tok_s` | Median decode rate at 1 concurrent request, 512 prompt tokens, full speed (C100). |
| `decode_c4_tok_s` | Median decode rate at 4 concurrent requests. |
| `decode_c8_tok_s` | Median decode rate at 8 concurrent requests. |
| `ttft_512_s` | Median time to first token at 512 prompt tokens (1 request). |
| `ttft_2048_s` | Median time to first token at 2,048 prompt tokens. |
| `kv_tokens` | Median key-value cache capacity. |
| `weights_gib` | Median weight memory. |
| `gpu_mib` | Median GPU memory in use. |
| `max_cv_pct` | Largest coefficient of variation of the decode rate over the six cells (512 and 2,048 prompt tokens at 1, 4, 8 requests). |
| `ratio_l4` | Decode rate at 1 request under the matched L4 compute share (29 or 30 percent MPS) divided by the full-speed rate. |
| `ratio_jetson` | Decode rate at 1 request under the matched Jetson share (10 percent MPS) divided by the full-speed rate. |

### `validation_v3_conditions.csv`

The 24 profile-by-condition rows of the calibration: full speed, and the matched compute shares of an L4-class and a Jetson-class target. **Rows:** 24.

| column | meaning |
|---|---|
| `profile` | Engine profile. |
| `condition` | C0 (no MPS), C100, or the matched share (L4 share 29 or 30 percent, Jetson share 10 percent). |
| `restarts` | Restarts behind the medians. |
| `decode_c1_tok_s` | Median decode rate at 1 request. |
| `decode_c4_tok_s` | Median decode rate at 4 requests. |
| `decode_c8_tok_s` | Median decode rate at 8 requests. |
| `ttft_512_s` | Median time to first token at 512 prompt tokens. |
| `ttft_2048_s` | Median time to first token at 2,048 prompt tokens. |
| `decode_c1_q1` | First quartile of the decode rate at 1 request over restarts. |
| `decode_c1_q3` | Third quartile of the same. |
| `kv_tokens` | Median key-value capacity. |
| `gpu_mib` | Median GPU memory in use. |
| `weights_gib` | Median weight memory. |

### `campaign_contraction_test.csv`

The joint contraction test of Stage 1 (condition A9) as flat key and value rows: the holds of the heaviest and the lightest vector of the certified set, the plateau test and the settling test per outcome and switch direction, and the verdict. **Rows:** 762.

| column | meaning |
|---|---|
| `path` | Key of the result, with `/` between the levels (for example the cycle, the hold, the outcome, the statistic). |
| `value` | Value of the key. |

### `s_profile_flat.csv`

The isolated profile of each arm of the certified set (one node, no co-location) as flat key and value rows, beside its certified deployed level. **Rows:** 380.

| column | meaning |
|---|---|
| `path` | Key of the result, with `/` between the levels (arm, outcome, statistic). |
| `value` | Value of the key. |
