"""
Identification of Slow Learners Using Machine Learning:
A Comparative Analysis of Classification Algorithms

Complete code reproducing ALL figures (Fig 2–13), tables (Table III–V),
SHAP analysis, ablation study, McNemar's test, and training times.

Dataset: https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style
Place 'student_performance_dataset.csv' in the same directory before running.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             roc_curve, confusion_matrix)
import shap

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ================================================================
# 1. LOAD DATA & CREATE BINARY LABEL
# ================================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv('student_performance_dataset.csv')
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Columns: {list(df.columns)}")

print(f"\nFinalGrade distribution:\n{df['FinalGrade'].value_counts().sort_index()}")

# Binary: FinalGrade 0 → Slow Learner (1), grades 1-3 → Not Slow (0)
df['SlowLearner'] = (df['FinalGrade'] == 0).astype(int)
print(f"\nSlow: {df['SlowLearner'].sum()} ({df['SlowLearner'].mean()*100:.1f}%)")
print(f"Not Slow: {(df['SlowLearner']==0).sum()} ({(1-df['SlowLearner'].mean())*100:.1f}%)")

feature_cols = [c for c in df.columns if c not in ['FinalGrade', 'SlowLearner']]
X = df[feature_cols]
y = df['SlowLearner']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Test slow learners: {y_test.sum()} / {len(y_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# FIG 2 — Class Distribution
# ================================================================
print("\n" + "=" * 60)
print("FIG 2 — Class Distribution")
print("=" * 60)

fig, ax = plt.subplots(figsize=(6, 4))
counts = df['SlowLearner'].value_counts().sort_index()
bars = ax.bar(['Not Slow Learner\n(10,171)', 'Slow Learner\n(3,832)'],
              counts.values, color=['#27AE60', '#E74C3C'],
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f'{val:,}', ha='center', fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Students')
ax.set_title('Fig. 2: Class Split — 3,832 Slow Learners vs 10,171 Non-Slow')
ax.set_ylim(0, max(counts.values) * 1.15)
sns.despine()
plt.tight_layout()
plt.savefig('fig2_class_distribution.png')
plt.close()
print("Saved: fig2_class_distribution.png")

# ================================================================
# FIG 3 — Correlation Matrix
# ================================================================
print("\n" + "=" * 60)
print("FIG 3 — Correlation Matrix")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 10))
corr = df[feature_cols + ['SlowLearner']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Fig. 3: Correlation Matrix — Academic Columns Dominate the Target Row')
plt.tight_layout()
plt.savefig('fig3_correlation_matrix.png')
plt.close()
print("Saved: fig3_correlation_matrix.png")

# ================================================================
# FIG 4 — Per-Column Histograms by Class
# ================================================================
print("\n" + "=" * 60)
print("FIG 4 — Per-Column Histograms by Class")
print("=" * 60)

hist_cols = ['ExamScore', 'AssignmentCompletion', 'Attendance',
             'StudyHours', 'Motivation', 'StressLevel',
             'Age', 'OnlineCourses', 'Resources']

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for idx, col in enumerate(hist_cols):
    ax = axes[idx // 3][idx % 3]
    for label, color, name in [(0, '#27AE60', 'Not Slow'), (1, '#E74C3C', 'Slow')]:
        ax.hist(df[df['SlowLearner'] == label][col], bins=25,
                alpha=0.6, color=color, label=name, density=True)
    ax.set_title(col, fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle('Fig. 4: Per-Column Histograms Split by Class', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('fig4_histograms_by_class.png')
plt.close()
print("Saved: fig4_histograms_by_class.png")

# ================================================================
# 2. MODEL TRAINING WITH GRID SEARCH TUNING
# ================================================================
print("\n" + "=" * 60)
print("2. TRAINING ALL 6 MODELS (with grid search for RF, SVM, GB)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Logistic Regression (no tuning) ---
lr = LogisticRegression(max_iter=1000, random_state=42)

# --- Decision Tree (no tuning, no depth cap) ---
dt = DecisionTreeClassifier(random_state=42)

# --- Random Forest (grid search) ---
print("Tuning Random Forest...")
rf_grid = {
    'max_depth': [5, 10, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'n_estimators': [50]
}
rf_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_grid, cv=cv, scoring='f1', n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train)
rf = rf_search.best_estimator_
print(f"  Best RF params: {rf_search.best_params_}")

# --- SVM (grid search) ---
print("Tuning SVM...")
svm_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}
svm_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_grid, cv=cv, scoring='f1', n_jobs=-1
)
svm_search.fit(X_train_scaled, y_train)
svm = svm_search.best_estimator_
print(f"  Best SVM params: {svm_search.best_params_}")

# --- KNN (k=5, default) ---
knn = KNeighborsClassifier(n_neighbors=5)

# --- Gradient Boosting (grid search) ---
print("Tuning Gradient Boosting...")
gb_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
gb_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_grid, cv=cv, scoring='f1', n_jobs=-1
)
gb_search.fit(X_train_scaled, y_train)
gb = gb_search.best_estimator_
print(f"  Best GB params: {gb_search.best_params_}")

# Train the non-grid-search models
lr.fit(X_train_scaled, y_train)
dt.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)

models = {
    'DT': dt,
    'RF': rf,
    'GB': gb,
    'SVM': svm,
    'LR': lr,
    'KNN': knn
}
print("\nAll models trained.")

# ================================================================
# 3. EVALUATION — TABLE III (Test-Set Scores)
# ================================================================
print("\n" + "=" * 60)
print("3. TABLE III — Test-Set Scores After Tuning")
print("=" * 60)

results = {}
y_preds = {}
y_probas = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_preds[name] = y_pred
    y_probas[name] = y_proba

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'MCC': mcc, 'AUC': auc
    }

results_df = pd.DataFrame(results).T
print("\nTABLE III: Test-Set Scores")
print(results_df.round(4).to_string())

# ================================================================
# FIG 5 — Metric Comparison Bar Chart
# ================================================================
print("\n" + "=" * 60)
print("FIG 5 — Metric Comparison")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 5))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUC']
x = np.arange(len(models))
width = 0.13
colors_met = ['#2980B9', '#27AE60', '#E67E22', '#E74C3C', '#8E44AD', '#1ABC9C']

for i, metric in enumerate(metrics_to_plot):
    vals = [results[m][metric] for m in models]
    ax.bar(x + i * width, vals, width, label=metric, color=colors_met[i])

ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(models.keys(), fontweight='bold')
ax.set_ylim(0.7, 1.05)
ax.set_ylabel('Score')
ax.set_title('Fig. 5: Metric Comparison — KNN Is the Only Model That Falls Behind')
ax.legend(loc='lower left', fontsize=9)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig('fig5_metric_comparison.png')
plt.close()
print("Saved: fig5_metric_comparison.png")

# ================================================================
# FIG 6 — ROC Curves
# ================================================================
print("\n" + "=" * 60)
print("FIG 6 — ROC Curves")
print("=" * 60)

fig, ax = plt.subplots(figsize=(7, 6))
colors_roc = {'DT': '#E74C3C', 'RF': '#27AE60', 'GB': '#2980B9',
              'SVM': '#8E44AD', 'LR': '#E67E22', 'KNN': '#95A5A6'}

for name in models:
    fpr, tpr, _ = roc_curve(y_test, y_probas[name])
    auc_val = roc_auc_score(y_test, y_probas[name])
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})',
            color=colors_roc[name], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Fig. 6: ROC Curves — Five Models at Ceiling, KNN Trails')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
sns.despine()
plt.tight_layout()
plt.savefig('fig6_roc_curves.png')
plt.close()
print("Saved: fig6_roc_curves.png")

# ================================================================
# FIG 7 — Confusion Matrices (Top 3: DT, RF, GB)
# ================================================================
print("\n" + "=" * 60)
print("FIG 7 — Confusion Matrices (Top 3)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name in zip(axes, ['DT', 'RF', 'GB']):
    cm = confusion_matrix(y_test, y_preds[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Slow', 'Slow'],
                yticklabels=['Not Slow', 'Slow'],
                cbar=False, linewidths=1, linecolor='white')
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
fig.suptitle('Fig. 7: Confusion Grids for Top Three — No Off-Diagonal Entries',
             fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('fig7_confusion_matrices.png')
plt.close()
print("Saved: fig7_confusion_matrices.png")

# ================================================================
# 4. SHAP ANALYSIS — FIG 8 (Beeswarm) & FIG 9 (Mean |SHAP|)
# ================================================================
print("\n" + "=" * 60)
print("4. SHAP ANALYSIS (on tuned Random Forest)")
print("=" * 60)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_scaled)

# For binary classification, shap_values may be a list [class0, class1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 = slow learner
else:
    shap_vals = shap_values

# FIG 8 — SHAP Beeswarm
print("Generating Fig. 8 — SHAP Beeswarm...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals, X_test_scaled, feature_names=feature_cols,
                  show=False, plot_size=None)
plt.title('Fig. 8: SHAP Beeswarm — Color = Raw Value, Horizontal = Push Toward Slow Learner')
plt.tight_layout()
plt.savefig('fig8_shap_beeswarm.png')
plt.close()
print("Saved: fig8_shap_beeswarm.png")

# FIG 9 — Mean |SHAP| Bar Chart
print("Generating Fig. 9 — Mean |SHAP| Bar...")
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_vals, X_test_scaled, feature_names=feature_cols,
                  plot_type='bar', show=False, plot_size=None)
plt.title('Fig. 9: Average |SHAP| Per Column — Academic Indicators Dominate')
plt.tight_layout()
plt.savefig('fig9_shap_importance.png')
plt.close()
print("Saved: fig9_shap_importance.png")

# ================================================================
# FIG 10 — Impurity-Based Feature Importance (from RF)
# ================================================================
print("\n" + "=" * 60)
print("FIG 10 — Impurity-Based Feature Importance")
print("=" * 60)

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx], color='#2980B9')
ax.set_xlabel('Impurity-Based Importance')
ax.set_title('Fig. 10: Impurity-Based Column Ranking from the Tuned Forest')
sns.despine()
plt.tight_layout()
plt.savefig('fig10_impurity_importance.png')
plt.close()
print("Saved: fig10_impurity_importance.png")

# ================================================================
# 5. ABLATION STUDY — TABLE IV & FIG 11
# ================================================================
print("\n" + "=" * 60)
print("5. ABLATION STUDY")
print("=" * 60)

# Define feature groups
academic = ['ExamScore', 'AssignmentCompletion', 'Resources']
behavioral = ['StudyHours', 'Attendance', 'Discussions', 'Extracurricular']
demographic = ['Gender', 'Age']
technology = ['Internet', 'EduTech', 'OnlineCourses']
shap_top5_cols = ['ExamScore', 'AssignmentCompletion', 'Resources',
                  'StudyHours', 'Attendance']

ablation_scenarios = {
    'All 15 columns':          feature_cols,
    'Drop Exam+Homework':      [c for c in feature_cols if c not in ['ExamScore', 'AssignmentCompletion']],
    'Drop Behavioral':         [c for c in feature_cols if c not in behavioral],
    'Drop Demographics':       [c for c in feature_cols if c not in demographic],
    'Drop Technology':         [c for c in feature_cols if c not in technology],
    'Exam+Homework only':      ['ExamScore', 'AssignmentCompletion'],
    'SHAP top 5 only':         shap_top5_cols,
}

ablation_results = {}
for scenario_name, cols in ablation_scenarios.items():
    X_tr = X_train[cols].values
    X_te = X_test[cols].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    rf_abl = RandomForestClassifier(
        n_estimators=50, max_depth=5, min_samples_leaf=1, random_state=42
    )
    rf_abl.fit(X_tr_s, y_train)
    y_pred_abl = rf_abl.predict(X_te_s)

    f1_abl = f1_score(y_test, y_pred_abl)
    mcc_abl = matthews_corrcoef(y_test, y_pred_abl)
    ablation_results[scenario_name] = {'F1': f1_abl, 'MCC': mcc_abl, 'Cols': len(cols)}
    print(f"  {scenario_name:<25} F1={f1_abl:.3f}  MCC={mcc_abl:.3f}  ({len(cols)} cols)")

# TABLE IV
print("\nTABLE IV: Ablation Results")
abl_df = pd.DataFrame(ablation_results).T
print(abl_df.to_string())

# FIG 11 — Ablation F1 Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(ablation_results.keys())
f1_vals = [ablation_results[n]['F1'] for n in names]
bar_colors = ['#2980B9'] * len(names)
bar_colors[1] = '#E74C3C'  # highlight the drop

bars = ax.bar(range(len(names)), f1_vals, color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0.7, 1.05)
ax.set_ylabel('F1 Score')
ax.set_title('Fig. 11: F1 Under Each Ablation — Removing Academic Pair Is the Only Thing That Hurts')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('fig11_ablation_f1.png')
plt.close()
print("Saved: fig11_ablation_f1.png")

# ================================================================
# 6. CROSS-VALIDATION — TABLE V & FIG 12
# ================================================================
print("\n" + "=" * 60)
print("6. FIVE-FOLD CROSS-VALIDATION")
print("=" * 60)

cv_models = {
    'DT':  DecisionTreeClassifier(random_state=42),
    'RF':  RandomForestClassifier(n_estimators=50, max_depth=5,
                                  min_samples_leaf=1, random_state=42),
    'GB':  GradientBoostingClassifier(**gb_search.best_params_, random_state=42),
    'SVM': SVC(**svm_search.best_params_, random_state=42),
    'LR':  LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

cv_scores = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=cv, scoring='accuracy')
    cv_scores[name] = scores
    print(f"  {name:<5} Mean={scores.mean():.4f}  Std={scores.std():.4f}")

# TABLE V
print("\nTABLE V: Five-Fold CV Accuracy")
for name, scores in cv_scores.items():
    print(f"  {name:<5} {scores.mean():.4f} ± {scores.std():.4f}")

# FIG 12 — CV Accuracy Boxplot
fig, ax = plt.subplots(figsize=(8, 5))
cv_df = pd.DataFrame(cv_scores)
bp = ax.boxplot([cv_df[col] for col in cv_df.columns],
                labels=cv_df.columns, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))

box_colors = ['#E74C3C', '#27AE60', '#2980B9', '#8E44AD', '#E67E22', '#95A5A6']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Accuracy')
ax.set_title('Fig. 12: CV Accuracy Distributions — Tight Boxes Everywhere Except KNN')
ax.set_ylim(0.88, 1.005)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.savefig('fig12_cv_accuracy.png')
plt.close()
print("Saved: fig12_cv_accuracy.png")

# ================================================================
# 7. McNEMAR'S PAIRED TEST
# ================================================================
print("\n" + "=" * 60)
print("7. McNEMAR'S PAIRED TEST")
print("=" * 60)

def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b):
    """Compute McNemar's chi-squared and p-value."""
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A right, B wrong | c = A wrong, B right
    b = ((correct_a) & (~correct_b)).sum()
    c = ((~correct_a) & (correct_b)).sum()

    if b + c == 0:
        print(f"  {name_a} vs {name_b}: Identical predictions. No test needed.")
        return

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    print(f"  {name_a} vs {name_b}: b={b}, c={c}, χ²={chi2_stat:.2f}, p={p_value:.4f}"
          f" {'*** Significant' if p_value < 0.05 else '(Not significant)'}")

rf_pred = y_preds['RF']
for name in ['KNN', 'LR', 'DT', 'SVM', 'GB']:
    mcnemar_test(y_test, rf_pred, y_preds[name], 'RF', name)

# ================================================================
# 8. TRAINING TIME — FIG 13
# ================================================================
print("\n" + "=" * 60)
print("8. TRAINING TIME (averaged over 5 runs)")
print("=" * 60)

timing_models = {
    'DT':  lambda: DecisionTreeClassifier(random_state=42),
    'LR':  lambda: LogisticRegression(max_iter=1000, random_state=42),
    'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
    'RF':  lambda: RandomForestClassifier(n_estimators=50, max_depth=5,
                                          min_samples_leaf=1, random_state=42),
    'GB':  lambda: GradientBoostingClassifier(**gb_search.best_params_,
                                              random_state=42),
    'SVM': lambda: SVC(**svm_search.best_params_, probability=True,
                       random_state=42),
}

train_times = {}
n_runs = 5
for name, model_fn in timing_models.items():
    times = []
    for _ in range(n_runs):
        m = model_fn()
        start = time.perf_counter()
        m.fit(X_train_scaled, y_train)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    avg = np.mean(times)
    train_times[name] = avg
    print(f"  {name:<5} {avg:.1f} ms")

# FIG 13 — Training Time Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
names_t = list(train_times.keys())
times_t = list(train_times.values())
bar_colors_t = ['#27AE60', '#E67E22', '#95A5A6', '#2980B9', '#8E44AD', '#E74C3C']

bars = ax.bar(names_t, times_t, color=bar_colors_t, edgecolor='white')
ax.set_ylabel('Training Time (ms)')
ax.set_title('Fig. 13: Training Time — SVM Is Orders of Magnitude Slower')
ax.set_yscale('log')

for bar, val in zip(bars, times_t):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f'{val:.0f} ms', ha='center', fontsize=9, fontweight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('fig13_training_time.png')
plt.close()
print("Saved: fig13_training_time.png")

# ================================================================
# DONE
# ================================================================
print("\n" + "=" * 60)
print("ALL DONE — Summary of generated files:")
print("=" * 60)
print("""
Figures:
  fig2_class_distribution.png    — Class split bar chart
  fig3_correlation_matrix.png    — Correlation heatmap
  fig4_histograms_by_class.png   — Per-column histograms
  fig5_metric_comparison.png     — Model metrics grouped bar
  fig6_roc_curves.png            — ROC curves (all 6 models)
  fig7_confusion_matrices.png    — Confusion grids (top 3)
  fig8_shap_beeswarm.png         — SHAP beeswarm plot
  fig9_shap_importance.png       — Mean |SHAP| bar chart
  fig10_impurity_importance.png  — Impurity-based ranking
  fig11_ablation_f1.png          — Ablation F1 comparison
  fig12_cv_accuracy.png          — 5-fold CV boxplot
  fig13_training_time.png        — Training time comparison

Tables printed to console:
  Table III — Test-set scores (Acc, Prec, Rec, F1, MCC, AUC)
  Table IV  — Ablation results
  Table V   — Five-fold CV accuracy
  McNemar's paired test results
""")
