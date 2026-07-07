"""
A/B Test: New Onboarding Screen — Conversion & Retention Analysis

Evaluates whether a redesigned onboarding flow increases signup-to-first-action
conversion without hurting Day-7 retention.

Data: synthetic, seeded for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
    confint_proportions_2indep,
    proportion_effectsize,
)
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# --- Test parameters -------------------------------------------------------

BASELINE_CR = 0.12
MDE_ABSOLUTE = 0.02
ALPHA = 0.05
POWER = 0.80

effect_size = proportion_effectsize(BASELINE_CR + MDE_ABSOLUTE, BASELINE_CR)
required_n_per_group = int(np.ceil(
    NormalIndPower().solve_power(
        effect_size=effect_size,
        alpha=ALPHA,
        power=POWER,
        alternative="larger",
    )
))

print("=" * 65)
print("TEST DESIGN")
print("=" * 65)
print(f"Baseline conversion:        {BASELINE_CR:.1%}")
print(f"Minimum Detectable Effect:  +{MDE_ABSOLUTE:.1%} (absolute)")
print(f"Alpha:                      {ALPHA}")
print(f"Power:                      {POWER}")
print(f"Required sample per group:  {required_n_per_group:,}")
print(f"Total sample required:      {2 * required_n_per_group:,}")
print()


# --- Data ------------------------------------------------------------------

N_PER_GROUP = 5000
TRUE_CONTROL_CR = 0.12
TRUE_TEST_CR = 0.145
TRUE_RETENTION_CONTROL = 0.35
TRUE_RETENTION_TEST = 0.34


def simulate_group(n, cr, ret_converted, ret_not_converted, group_name):
    converted = np.random.binomial(1, cr, size=n)
    retention = np.where(
        converted == 1,
        np.random.binomial(1, ret_converted, size=n),
        np.random.binomial(1, ret_not_converted, size=n),
    )
    signup_week = np.random.choice([1, 2, 3, 4], size=n, p=[0.3, 0.3, 0.2, 0.2])
    return pd.DataFrame({
        "user_id": np.arange(n),
        "group": group_name,
        "signup_week": signup_week,
        "converted": converted,
        "retained_d7": retention,
    })


control = simulate_group(
    N_PER_GROUP, TRUE_CONTROL_CR, TRUE_RETENTION_CONTROL, 0.08, "control"
)
test = simulate_group(
    N_PER_GROUP, TRUE_TEST_CR, TRUE_RETENTION_TEST, 0.08, "test"
)

df = pd.concat([control, test], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Total users in experiment: {len(df):,}")
print(df["group"].value_counts().to_string())
print()


# --- Sanity checks ---------------------------------------------------------

print("=" * 65)
print("SANITY CHECKS")
print("=" * 65)

# Sample Ratio Mismatch — verifies bucketing is not broken
observed = df["group"].value_counts().values
expected = np.array([len(df) / 2, len(df) / 2])
chi2, p_srm = stats.chisquare(observed, expected)
print(f"SRM check (chi-square): p = {p_srm:.3f}", end="  ")
if p_srm < 0.001:
    print("⚠️  allocation imbalance, investigate before proceeding")
else:
    print("✅ allocation balanced")

# Signup week distribution should be comparable across groups
week_dist = df.groupby("group")["signup_week"].value_counts(normalize=True).unstack()
print("\nSignup week distribution by group:")
print(week_dist.round(3))
print()


# --- Primary metric: conversion -------------------------------------------

print("=" * 65)
print("PRIMARY METRIC — CONVERSION RATE")
print("=" * 65)

conv = df.groupby("group").agg(
    users=("user_id", "count"),
    converted=("converted", "sum"),
)
conv["cr"] = conv["converted"] / conv["users"]

# Wilson CI is more accurate near the boundaries than the normal approximation
for group in conv.index:
    low, high = proportion_confint(
        conv.loc[group, "converted"], conv.loc[group, "users"], method="wilson"
    )
    print(f"{group:>8}: CR = {conv.loc[group, 'cr']:.3%}  "
          f"95% CI [{low:.3%}, {high:.3%}]  n = {conv.loc[group, 'users']:,}")

# One-sided Z-test: control CR < test CR
z_stat, p_value = proportions_ztest(
    conv["converted"].values,
    conv["users"].values,
    alternative="smaller",
)

control_cr = conv.loc["control", "cr"]
test_cr = conv.loc["test", "cr"]
abs_lift = test_cr - control_cr
rel_lift = abs_lift / control_cr

ci_low, ci_high = confint_proportions_2indep(
    conv.loc["test", "converted"], conv.loc["test", "users"],
    conv.loc["control", "converted"], conv.loc["control", "users"],
    method="wald",
)

print(f"\nAbsolute lift: {abs_lift:+.2%}  (95% CI [{ci_low:+.2%}, {ci_high:+.2%}])")
print(f"Relative lift: {rel_lift:+.1%}")
print(f"Z-statistic:   {z_stat:.3f}")
print(f"P-value:       {p_value:.4f}")

if p_value < ALPHA:
    print(f"\n✅ Statistically significant at alpha = {ALPHA}")
else:
    print(f"\n❌ Not significant at alpha = {ALPHA}")
print()


# --- Guardrail: Day-7 retention -------------------------------------------

print("=" * 65)
print("GUARDRAIL — DAY-7 RETENTION")
print("=" * 65)

ret = df.groupby("group").agg(
    users=("user_id", "count"),
    retained=("retained_d7", "sum"),
)
ret["retention_d7"] = ret["retained"] / ret["users"]

for group in ret.index:
    low, high = proportion_confint(
        ret.loc[group, "retained"], ret.loc[group, "users"], method="wilson"
    )
    print(f"{group:>8}: D7 = {ret.loc[group, 'retention_d7']:.3%}  "
          f"95% CI [{low:.3%}, {high:.3%}]")

z_ret, p_ret = proportions_ztest(
    ret["retained"].values, ret["users"].values, alternative="two-sided"
)
print(f"\nRetention difference p-value: {p_ret:.4f}")
if p_ret < ALPHA:
    diff = ret.loc["test", "retention_d7"] - ret.loc["control", "retention_d7"]
    print("⚠️  Retention dropped significantly — guardrail violated"
          if diff < 0 else "✅ Retention improved significantly")
else:
    print("✅ No significant retention change — guardrail holds")
print()


# --- Segment analysis -----------------------------------------------------

print("=" * 65)
print("SEGMENT ANALYSIS — CONVERSION BY SIGNUP WEEK")
print("=" * 65)

seg = df.groupby(["signup_week", "group"]).agg(
    users=("user_id", "count"),
    converted=("converted", "sum"),
)
seg["cr"] = seg["converted"] / seg["users"]
seg_pivot = seg["cr"].unstack()
seg_pivot["abs_lift"] = seg_pivot["test"] - seg_pivot["control"]
print(seg_pivot.round(4))
print()


# --- Decision -------------------------------------------------------------

print("=" * 65)
print("DECISION")
print("=" * 65)

primary_significant = p_value < ALPHA
guardrail_ok = (p_ret >= ALPHA) or (
    ret.loc["test", "retention_d7"] >= ret.loc["control", "retention_d7"]
)
effect_meaningful = abs_lift >= MDE_ABSOLUTE * 0.5

if primary_significant and guardrail_ok and effect_meaningful:
    decision = "🚀 SHIP"
    reason = (
        f"Conversion lift of {abs_lift:+.2%} is statistically significant "
        f"(p = {p_value:.4f}), effect size is business-meaningful "
        f"(≥ {MDE_ABSOLUTE * 0.5:.1%}), and Day-7 retention did not deteriorate. "
        f"Recommend rolling out to 100%."
    )
elif primary_significant and not guardrail_ok:
    decision = "⛔ HOLD — investigate retention"
    reason = (
        "Primary metric improved but retention guardrail was violated. "
        "The lift may reflect lower-quality users pushed through the funnel. "
        "Do not ship until root cause is understood."
    )
elif primary_significant and not effect_meaningful:
    decision = "⚠️  HOLD — significant but small"
    reason = (
        "Statistically significant but effect is below business threshold. "
        "Consider whether the engineering cost justifies the marginal gain."
    )
else:
    decision = "❌ DO NOT SHIP"
    reason = (
        f"Effect is not statistically distinguishable from noise "
        f"(p = {p_value:.4f})."
    )

print(f"\n{decision}\n")
print(reason)
print()


# --- Visualisation --------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

groups = conv.index.tolist()
crs = conv["cr"].values
cis = [proportion_confint(conv.loc[g, "converted"], conv.loc[g, "users"],
                          method="wilson") for g in groups]
errors = [[crs[i] - cis[i][0] for i in range(2)],
          [cis[i][1] - crs[i] for i in range(2)]]

axes[0].bar(groups, crs, yerr=errors, capsize=8,
            color=["#94a3b8", "#3b82f6"], alpha=0.85)
axes[0].set_title("Conversion Rate with 95% CI", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Conversion rate")
axes[0].set_ylim(0, max(crs) * 1.3)
for i, v in enumerate(crs):
    axes[0].text(i, v + errors[1][i] + 0.003, f"{v:.2%}",
                 ha="center", fontweight="bold")

seg_plot = seg_pivot[["control", "test"]]
seg_plot.plot(kind="bar", ax=axes[1], color=["#94a3b8", "#3b82f6"], alpha=0.85)
axes[1].set_title("Conversion by Signup Week Cohort",
                  fontsize=12, fontweight="bold")
axes[1].set_ylabel("Conversion rate")
axes[1].set_xlabel("Signup week")
axes[1].legend(title="")
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("ab_test_results.png", dpi=120, bbox_inches="tight")
print("Figure saved: ab_test_results.png")
