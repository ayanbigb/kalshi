# Parameter coverage

The 250-item Kalshi NFL prop/combo checklist mapped to this code. Regenerate with `python -m propbot params`.

| status | meaning | count |
|---|---|---|
| model | pre-game feature the price model uses | 82 |
| context | computed and shown in `matchup`, available to the combo engine, but left out of the price model (weather, and the matchup set that did not help on the market board; `model.drop_feats`) | 57 |
| combo | computed per combo by the combo engine | 47 |
| market | computed from Kalshi quotes at scan time | 25 |
| proxy | close stand-in from free data, in the price model | 23 |
| none | not available from free data | 16 |

| # | parameter | status | implemented as |
|---|---|---|---|
| 1 | Season avg passing yards | model | `passing_yards_szn` |
| 2 | Last 3 games passing yards | context | `passing_yards_m3` |
| 3 | Last 5 games passing yards | context | `passing_yards_m5` |
| 4 | Last 10 games passing yards | context | `passing_yards_m10` |
| 5 | Season avg pass attempts | model | `attempts_szn` |
| 6 | Season avg completions | model | `completions_szn` |
| 7 | Completion % | model | `comp_pct_e6` |
| 8 | Yards per attempt | model | `ypa_e6` |
| 9 | Adjusted yards per attempt | context | `aya_e6` |
| 10 | Passing TDs per game | model | `passing_tds_e6`, `passing_tds_szn` |
| 11 | Interceptions per game | model | `passing_interceptions_e6`, `int_per_att_e6` |
| 12 | Rushing yards per game | model | `rushing_yards_e6`, `rushing_yards_szn` |
| 13 | Rushing attempts per game | model | `carries_e6`, `carries_szn` |
| 14 | Yards per carry | model | `ypc_e6` |
| 15 | Receiving yards per game | model | `receiving_yards_e6`, `receiving_yards_szn` |
| 16 | Receptions per game | model | `receptions_e6`, `receptions_szn` |
| 17 | Targets per game | model | `targets_e6`, `targets_szn` |
| 18 | Catch % | model | `catch_rate_e6` |
| 19 | Yards per reception | context | `ypr_e6` |
| 20 | Receiving TDs per game | model | `receiving_tds_e6` |
| 21 | First-half production | context | `h1_yds_e6`, `h1_share_e6` |
| 22 | Second-half production | context | `h2_yds_e6` |
| 23 | Red-zone touches | model | `rz_targets_e6`, `rz_carries_e6` |
| 24 | Goal-line opportunities | model | `gl_carries_e6`, `ez_targets_e6` |
| 25 | Fantasy points per game | model | `fantasy_points_ppr_e6`, `xfp_e6` |
| 26 | Offensive snap % | model | `offense_pct_e6`, `offense_pct_e2` |
| 27 | Routes run per game | none | route counts are charting data (PFF/FTN paid); snap share x team dropbacks is the proxy in #28 |
| 28 | Route participation rate | proxy | `offense_pct_e6`, `off_dropbacks_e4` |
| 29 | Target share | model | `target_share_e6`, `target_share_e6_norm` |
| 30 | Air-yard share | model | `air_yards_share_e6`, `ngs_iay_share_e6` |
| 31 | aDOT | model | `adot_e6`, `ngs_iay_e6` |
| 32 | Average separation | model | `ngs_sep_e6` |
| 33 | Separation at target | proxy | `ngs_sep_e6`, `ngs_cushion_e6` |
| 34 | Red-zone target share | proxy | `rz_targets_e6`, `off_rz_plays_e4` |
| 35 | Third-down target share | model | `third_targets_e6` |
| 36 | First-read target % | none | first-read charting is proprietary (PFF/Fantasy Points) |
| 37 | Designed rushing attempts | proxy | `carries_e6`, `scrambles_e6` |
| 38 | Scramble rate | model | `scramble_share_e6`, `scrambles_e6` |
| 39 | Short-yardage rushing share | proxy | `gl_carries_e6`, `carry_share_e6` |
| 40 | Two-minute offense participation | none | needs per-play personnel (participation data ended 2023) |
| 41 | No-huddle snap participation | proxy | `off_no_huddle_e4`, `offense_pct_e6` |
| 42 | Touches per offensive snap | model | `touch_per_snap_e6`, `tgt_per_snap_e6`, `carry_per_snap_e6` |
| 43 | Average time on field | proxy | `offense_snaps_e6` |
| 44 | Snap share last 3 games | context | `offense_pct_m3` |
| 45 | Usage trend last 5 games | context | `target_share_trend5`, `carry_share_trend5` |
| 46 | Opp passing yards allowed/g | model | `opp_alw_tm_pass_yds_e5` |
| 47 | Opp rushing yards allowed/g | model | `opp_alw_tm_rush_yds_e5` |
| 48 | Opp receiving yards allowed (to position) | model | `pa_rec_yds_e5` |
| 49 | Opp passing TDs allowed | model | `opp_u_db_tdr`, `pa_td_e5` |
| 50 | Opp rushing TDs allowed | proxy | `pa_td_e5`, `opp_drz_td_e5` |
| 51 | Opp receptions allowed | model | `pa_rec_e5` |
| 52 | Opp yards per carry allowed | model | `opp_alw_off_ypc_e5` |
| 53 | Opp yards per reception allowed | proxy | `opp_u_db_ypt`, `opp_alw_off_yac_e5` |
| 54 | Opp completion % allowed | model | `opp_alw_off_cp_e5`, `opp_u_db_cmp` |
| 55 | Opp pressure rate | model | `opp_dpress_e5`, `opp_u_press_pg` |
| 56 | Opp blitz rate | model | `opp_dblitz_e5` |
| 57 | Opp sack rate | model | `opp_dsack_rate_e5`, `opp_alw_off_sacks_e5` |
| 58 | Opp interception rate | model | `opp_alw_off_ints_e5` |
| 59 | Opp missed tackle rate | model | `opp_dmiss_e5`, `opp_u_def_mtk` |
| 60 | Opp explosive play rate allowed | model | `opp_alw_off_explosive_e5` |
| 61 | Opp 3rd-down conversion allowed | context | `opp_d3conv_e5` |
| 62 | Opp red-zone defense | context | `opp_drz_td_e5` |
| 63 | Opp defensive DVOA | none | DVOA is FTN's proprietary metric; EPA/play allowed (#64) is the public equivalent |
| 64 | Opp EPA/play allowed | model | `opp_alw_off_epa_play_e5`, `opp_alw_off_db_epa_e5`, `opp_alw_off_ru_epa_e5` |
| 65 | Opp man coverage frequency | none | coverage charting (FTN/PFF) is paid; nflverse participation ended 2023 |
| 66 | Opp zone coverage frequency | none | same as #65 |
| 67 | Opp yards allowed to WR1 | context | `opp_alw_wr1_e5` |
| 68 | Opp yards allowed to WR2 | context | `opp_alw_wr2_e5` |
| 69 | Opp yards allowed to TEs | model | `opp_alw_te1_e5`, `pa_rec_yds_e5` |
| 70 | Opp yards allowed to RBs | model | `opp_alw_rb1_e5`, `pa_rush_yds_e5` |
| 71 | Game total | model | `total_line` |
| 72 | Team implied total | model | `implied_total` |
| 73 | Opp implied total | model | `opp_implied_total` |
| 74 | Point spread | model | `spread`, `abs_spread` |
| 75 | Moneyline win probability | context | `ml_win_prob` |
| 76 | Expected offensive plays | model | `exp_plays`, `off_plays_e4` |
| 77 | Expected possessions | proxy | `exp_plays`, `oppoff_off_plays_e4` |
| 78 | Expected plays per drive | none | drive ids are not in the cached play-by-play columns |
| 79 | Expected drives per team | none | same as #78 |
| 80 | Pace of play | model | `off_plays_e4`, `opp_alw_off_plays_e5` |
| 81 | Seconds per offensive snap | proxy | `off_plays_e4` |
| 82 | Neutral-game pass rate | model | `off_neutral_pass_e4`, `off_proe_e4` |
| 83 | Neutral-game rush rate | model | `off_neutral_pass_e4` |
| 84 | Expected game script | model | `spread`, `p_trail_big` |
| 85 | P(blowout) | context | `p_blowout` |
| 86 | P(close game) | context | `p_close` |
| 87 | Expected time of possession | proxy | `off_plays_e4`, `spread` |
| 88 | First-half implied scoring | proxy | `implied_total` |
| 89 | Second-half implied scoring | proxy | `implied_total`, `spread` |
| 90 | P(overtime) | context | `p_ot_proxy` |
| 91 | Temperature | context | `wx_temp` |
| 92 | Feels-like temperature | context | `wx_temp`, `wx_wind`, `wx_humidity` |
| 93 | Wind speed | context | `wx_wind` |
| 94 | Wind gusts | context | `wx_gust` |
| 95 | Wind direction | none | direction relative to the field needs stadium orientation; not modelled |
| 96 | Precipitation probability | context | `wx_precip` |
| 97 | Expected rainfall | context | `wx_precip` |
| 98 | Snowfall | context | `wx_snow` |
| 99 | Humidity | context | `wx_humidity` |
| 100 | Atmospheric pressure | none | not fetched (no measurable link to props in the literature) |
| 101 | Stadium elevation | context | `altitude` |
| 102 | Indoor vs outdoor | model | `indoors` |
| 103 | Field surface | model | `turf` |
| 104 | Stadium home-field advantage | proxy | `is_home`, `spread` |
| 105 | Weather-adjusted passing efficiency | context | `wx_bad`, `sp_badwx_res` |
| 106 | Injury designation | model | `inj_questionable`, `inj_listed` |
| 107 | Practice participation Wed | proxy | `inj_practice` |
| 108 | Practice participation Thu | proxy | `inj_practice` |
| 109 | Practice participation Fri | model | `inj_practice` |
| 110 | Injury history | context | `inj_hist17` |
| 111 | Games missed this season | context | `games_missed_szn` |
| 112 | Expected snap restriction | proxy | `inj_practice`, `games_since_absence` |
| 113 | Starting QB availability | model | `qb_changed`, `qb_games_career` |
| 114 | Offensive line injury count | context | `ol_miss` |
| 115 | Starting LT available | context | `ol_miss` |
| 116 | Starting RT available | context | `ol_miss` |
| 117 | Starting C available | context | `ol_miss` |
| 118 | Starting WR availability | model | `vac_tgt_wr` |
| 119 | Starting RB availability | model | `vac_tgt_rb`, `vac_carry` |
| 120 | Starting TE availability | model | `vac_tgt_te` |
| 121 | Opp CB1 availability | context | `opp_u_cb1_share`, `opp_u_cb1_ypt` |
| 122 | Opp CB2 availability | context | `opp_u_miss_cb` |
| 123 | Opp safety availability | context | `opp_u_miss_s` |
| 124 | Opp linebacker availability | context | `opp_u_miss_lb` |
| 125 | Replacement quality | model | `opp_u_cb_ypt`, `opp_u_s_ypt`, `target_share_e6_norm` |
| 126 | Expected defensive personnel package | none | personnel groupings need participation data (ended 2023) |
| 127 | Offensive line continuity | context | `ol_miss` |
| 128 | Workload after recent injury | context | `games_since_absence`, `offense_pct_m3` |
| 129 | Career vs opponent | context | `vsopp_n`, `vsopp_res_rr_yds`, `vsopp_res_pass_yds` |
| 130 | Last 3 meetings vs opponent | context | `vsopp_last3_res` |
| 131 | Home performance | context | `sp_home_res` |
| 132 | Away performance | context | `sp_home_res` |
| 133 | Divisional games | context | `sp_div_res` |
| 134 | Prime-time games | context | `sp_prime_res`, `primetime` |
| 135 | Cold weather | context | `sp_cold_res` |
| 136 | Rain / bad weather | context | `sp_badwx_res` |
| 137 | Vs top-10 defenses | context | `sp_deftier_res` |
| 138 | Vs bottom-10 defenses | context | `sp_deftier_res` |
| 139 | After a loss | context | `sp_afterloss_res` |
| 140 | After a win | context | `sp_afterloss_res` |
| 141 | With current starting QB | context | `sp_qb_res`, `sp_qb_n` |
| 142 | Without primary teammate | model | `sp_mate_out_res`, `vac_tgt` |
| 143 | Vs man coverage | none | see #65 |
| 144 | Vs zone coverage | none | see #65 |
| 145 | Short rest | context | `sp_shortrest_res`, `short_week` |
| 146 | Days since last game | model | `days_since_last`, `rest` |
| 147 | Travel distance | context | `travel_km` |
| 148 | Time-zone change | context | `tz_shift`, `body_clock` |
| 149 | Consecutive road games | context | `road_streak` |
| 150 | Similar game scripts | context | `sp_script_res` |
| 151 | EPA | model | `passing_epa_e6`, `rushing_epa_e6`, `receiving_epa_e6` |
| 152 | CPOE | model | `passing_cpoe_e6`, `ngs_cpoe_e6` |
| 153 | Success rate | model | `off_success_e4` |
| 154 | Explosive play % | model | `expl_rec_rate_e6`, `expl_rush_rate_e6` |
| 155 | YAC | model | `receiving_yards_after_catch_e6` |
| 156 | Yards after contact | model | `yac_att_e6`, `pfr_yac_e6` |
| 157 | Expected YAC | model | `ngs_yac_oe_e6` |
| 158 | Broken tackles per touch | model | `pfr_ru_bt_e6`, `pfr_rec_bt_e6` |
| 159 | Missed tackles forced | model | `pfr_ru_bt_e6` |
| 160 | Receiver win rate | none | ESPN win rates are not published as data |
| 161 | Pass-block win rate | none | same as #160 |
| 162 | Run-block win rate | none | same as #160 |
| 163 | Pressure-to-sack conversion | proxy | `sack_rate_e6`, `pressure_rate_e6` |
| 164 | QB pressure rate faced | model | `pressure_rate_e6`, `qb_pressure_rate_e6` |
| 165 | QB time to throw | model | `ngs_ttt_e6`, `qb_ngs_ttt_e6` |
| 166 | QB pocket time | proxy | `ngs_ttt_e6` |
| 167 | QB accuracy by depth | proxy | `ngs_cpoe_e6`, `ngs_p_iay_e6`, `bad_throw_rate_e6` |
| 168 | Receiver target efficiency | model | `ypt_e6`, `xcatch_rate_e6` |
| 169 | Yards per route run | proxy | `ypt_e6`, `offense_pct_e6` |
| 170 | Rush yards over expected | model | `ngs_ryoe_att_e6`, `rush_yds_oe_e6` |
| 171 | Receiving yards over expected | model | `rec_yds_oe_e6` |
| 172 | Passing yards over expected | model | `pass_yds_oe_e6` |
| 173 | Expected touchdowns | model | `xtd_e6`, `xrec_td_e6`, `xrush_td_e6` |
| 174 | Red-zone expected points | proxy | `xtd_e6`, `off_rz_plays_e4` |
| 175 | Player statistical variance | model | `receiving_yards_sd8`, `rushing_yards_sd8`, `passing_yards_sd8` |
| 176 | Historical prop correlation | combo | copula rho per (relation, role:stat) pair, models/copula.json |
| 177 | QB pass yds vs WR rec yds | combo | team|QB:pass_yds|WR:rec_yds |
| 178 | QB pass TDs vs WR TDs | combo | team|QB:pass_tds|WR:td |
| 179 | RB rush yds vs team passing | combo | team|QB:pass_yds|RB:rush_yds |
| 180 | RB receptions vs RB rush yds | combo | self|RB:rec|RB:rush_yds |
| 181 | WR1 vs WR2 rec yds | combo | team|WR:rec_yds|WR:rec_yds |
| 182 | WR receptions vs WR rec yds | combo | self|WR:rec|WR:rec_yds |
| 183 | TE vs WR rec yds | combo | team|TE:rec_yds|WR:rec_yds |
| 184 | QB rush yds vs QB pass yds | combo | self|QB:pass_yds|QB:rush_yds |
| 185 | QB rush att vs pass att | combo | self|QB:pass_att|QB:rush_att |
| 186 | Opposing QB pass yds | combo | opp|QB:pass_yds|QB:pass_yds |
| 187 | Opposing WR rec yds | combo | opp|WR:rec_yds|WR:rec_yds |
| 188 | Opposing RB rush yds | combo | opp|RB:rush_yds|RB:rush_yds |
| 189 | Player TDs vs team implied points | model | `implied_total`, `xtd_e6` |
| 190 | Player yards vs game total | model | `total_line` |
| 191 | Player production vs spread | model | `spread`, `sp_script_res` |
| 192 | Same-team target competition | combo | negative team rho between receivers; target_share_e6_norm in the model |
| 193 | Same-team TD competition | combo | team|*td|*td rho |
| 194 | Shared game-script dependency | combo | all same-game legs share latent correlation; risk report counts them |
| 195 | Shared weather dependency | combo | same-game legs share the game's weather; shown in matchup |
| 196 | Shared injury dependency | combo | legs on the same team share vac_* inputs; same-player legs rho |
| 197 | P(B | A) | combo | risk_report pairs[].p_b_given_a |
| 198 | Joint probability of all legs | combo | p_model / p_all_hit |
| 199 | Correlation-adjusted combo probability | combo | p_model vs p_indep_model |
| 200 | P(all legs fail) | combo | p_all_fail |
| 201 | Current YES price | market | yes_ask / yes_bid |
| 202 | Current NO price | market | no_ask / no_bid |
| 203 | Implied probability | market | p_market (mid) |
| 204 | Model probability | market | p_model, p_final (blended) |
| 205 | Probability difference | market | p_model - p_market (disagreement gate) |
| 206 | EV per contract | market | ev |
| 207 | Expected ROI | market | roi |
| 208 | Bid-ask spread | market | spread |
| 209 | Market liquidity | market | yes_ask_size / no_ask_size |
| 210 | Open interest | market | open_interest |
| 211 | Trading volume | market | volume |
| 212 | Price move 5 min | market | move_5m (needs logged scans) |
| 213 | Price move 1 hour | market | move_1h |
| 214 | Price move 24 hours | market | move_24h |
| 215 | Move since open | market | move_open (since first logged) |
| 216 | Price volatility | market | mid_vol |
| 217 | Order book depth | market | depth_top (top level only) |
| 218 | Slippage estimate | market | slippage_est |
| 219 | Transaction fees | market | edge.fee: ceil(0.07 * n * p * (1-p)) |
| 220 | Contract payout | market | $1 per contract |
| 221 | Market closing time | market | close_time |
| 222 | Time to kickoff | market | minutes_to_kickoff |
| 223 | Move after injury news | market | move_* around injury-report times (from logged scans) |
| 224 | Move after weather updates | market | move_* (from logged scans) |
| 225 | Kalshi vs sportsbook consensus | market | odds.py book_shift (needs ODDS_API_KEY) |
| 226 | Number of legs | combo | n_legs; best combo per size |
| 227 | Individual leg probability | combo | p_leg, p_mkt per leg |
| 228 | Combined YES probability | combo | p_model |
| 229 | Combined NO probability | combo | p_ge1_fail |
| 230 | Combined contract price | combo | est_price = fair x (1 + hold); max_price |
| 231 | Combined payout | combo | 1 / est_price per $1 |
| 232 | Combined EV | combo | ev |
| 233 | Profit after fees | combo | ev, roi (fee included) |
| 234 | P(at least one leg fails) | combo | p_ge1_fail |
| 235 | P(two or more fail) | combo | p_ge2_fail |
| 236 | Pairwise leg correlation | combo | pairs[].rho, max_abs_rho, mean_rho |
| 237 | Players from same team | combo | max_same_team |
| 238 | Players from opposing team | combo | n_opposing_pairs |
| 239 | Props from same game | combo | max_legs_one_game, shared_game_legs |
| 240 | Props across games | combo | n_games |
| 241 | Exposure to one game | combo | max_legs_per_game cap |
| 242 | Exposure to one player | combo | max_legs_per_player cap; max_same_stat caps one stat type |
| 243 | Accuracy of similar combos | combo | combo-backtest by leg count (reports/combo_backtest.json) |
| 244 | Max acceptable correlation | combo | max_pair_rho cap |
| 245 | Optimal number of legs | combo | growth-ranked best per size |
| 246 | Bankroll allocation | combo | fractional Kelly stake, capped |
| 247 | Expected max drawdown | combo | portfolio_drawdown |
| 248 | Model calibration error | combo | ece per stat from backtest.json -> calibration band |
| 249 | Confidence interval on combo probability | combo | p_lo / p_hi + Monte Carlo error |
| 250 | EV under pessimistic assumptions | combo | p_pessimistic (legs 90% toward market) |
