# Identification of Slow Learners Using Machine Learning
### A Comparative Analysis of Classification Algorithms

**Author:** Bahaar Sharma (2210990209)  
**Supervisor:** Dr. Gurpreet Singh  
**Institution:** Department of Computer Science and Engineering, Chitkara University, Punjab  

---

## Abstract

This project investigates whether routine student data can reliably flag slow learners using machine learning. Starting from **14,003 student records** containing grades, attendance, homework rates, study habits, and demographic details, we built a binary label (slow learner or not) and evaluated **six classifiers** spanning four algorithmic families. Four models (Decision Tree, Random Forest, Gradient Boosting, SVM) achieved **perfect test-set accuracy**. SHAP analysis revealed that only **two features — Exam Scores and Assignment Completion** — drive the predictions, confirmed by an ablation study showing these two columns alone replicate the full 15-column pipeline's performance.

---

## Problem Statement

Students often struggle silently. The data needed to help them exists in gradebooks and LMS logs, but nobody examines it until the semester is over. Instructors managing hundreds of students cannot manually track individual performance trends. This project builds a **scalable, explainable early-warning system** that identifies at-risk students using data schools already collect.

---

## Dataset

- **Source:** [Kaggle — Student Performance and Learning Style Dataset](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style)
- **Records:** 14,003 students, 16 columns, zero missing values
- **Label:** `FinalGrade == 0` → Slow Learner (3,832 students, 27.4%), grades 1–3 → Not Slow (10,171, 72.6%)

### Feature Groups

| Group | Columns |
|-------|---------|
| Academic | ExamScore, AssignmentCompletion, Resources |
| Behavioral | StudyHours, Attendance, Discussions, Extracurricular |
| Psychological | Motivation, StressLevel |
| Demographic | Gender, Age |
| Technology | Internet, EduTech, OnlineCourses |
| Preference | LearningStyle |

---

## Methodology

```
Raw Data (14,003) → Preprocessing & Scaling → Binary Label Creation
    → 6 Classifiers (Grid Search Tuned) → SHAP Explanations → Intervention Alerts
```

### Models Used

| Model | Type | Tuning |
|-------|------|--------|
| Logistic Regression | Linear | max_iter=1000 |
| Decision Tree | Tree | No depth cap |
| Random Forest | Bagged Ensemble | GridSearchCV (depth, leaf size) |
| SVM | Margin-based | GridSearchCV (kernel, C) |
| KNN (k=5) | Distance-based | Default |
| Gradient Boosting | Boosted Ensemble | GridSearchCV (lr, depth, subsample) |

- **Split:** 80/20 stratified train-test
- **Scaling:** StandardScaler fitted on training data only
- **Tuning:** 5-fold stratified CV optimizing F1 score

---

## Results

### Model Performance (Table III)

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC |
|-------|----------|-----------|--------|-----|-----|-----|
| DT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RF | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| GB | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LR | 0.999 | 1.000 | 0.996 | 0.998 | 0.997 | 1.000 |
| KNN | 0.914 | 0.888 | 0.785 | 0.833 | 0.778 | 0.964 |

### SHAP Feature Importance

ExamScore and AssignmentCompletion dominate all predictions. Demographics (Gender, Age) contribute near-zero — the model is algorithmically fair.

### Ablation Study (Table IV)

| Scenario | F1 | Columns Used |
|----------|-----|-------------|
| All 15 columns | 1.000 | 15 |
| **Drop Exam + Homework** | **0.823** | **13** |
| Drop Behavioral | 1.000 | 12 |
| Drop Demographics | 1.000 | 13 |
| Drop Technology | 1.000 | 12 |
| **Exam + Homework only** | **1.000** | **2** |
| SHAP top 5 only | 1.000 | 5 |

**Key finding:** Just 2 columns replicate the full pipeline. Removing them is the only thing that hurts.

### Statistical Validation

- **McNemar's RF vs KNN:** χ²=239.0, p<0.001 (significant)
- **McNemar's RF vs LR:** χ²=1.33, p=0.248 (not significant)
- **RF vs DT/SVM/GB:** Identical predictions on every test student

### Training Time

| Model | Time |
|-------|------|
| DT | ~4 ms |
| LR | ~12 ms |
| RF | ~350 ms |
| GB | ~500 ms |
| SVM | ~3,500 ms |

---

## Project Structure

```
├── student_performance_dataset.csv     # Dataset (download from Kaggle)
├── slow_learner_analysis.py            # Complete ML pipeline & all figures
├── Bahaar_Sharma_2210990209_SlowLearner_PPT.pptx  # Presentation
├── 2210990209_BahaarSharma_ResearchPaper_COOP2.pdf # Research paper
└── README.md                           # This file
```

## Generated Figures

| Figure | Description |
|--------|-------------|
| fig2 | Class distribution bar chart |
| fig3 | Correlation matrix heatmap |
| fig4 | Per-column histograms by class |
| fig5 | Model metrics grouped bar chart |
| fig6 | ROC curves (all 6 models) |
| fig7 | Confusion matrices (top 3 models) |
| fig8 | SHAP beeswarm plot |
| fig9 | Mean SHAP importance bar chart |
| fig10 | Impurity-based feature ranking |
| fig11 | Ablation study F1 comparison |
| fig12 | 5-fold CV accuracy boxplot |
| fig13 | Training time comparison |

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap
```

### Steps

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style) and place `student_performance_dataset.csv` in the project folder.
2. Run the analysis:
   ```bash
   python slow_learner_analysis.py
   ```
3. All 12 figures are saved as PNG files. Tables and statistical tests print to console.

---

## Practical Deployment (Minimal Recipe)

A school needs only:
1. Export **2 columns** from the gradebook (exam scores + assignment completion)
2. Train a **Decision Tree** (takes 4 milliseconds)
3. Run **SHAP** for per-student explanations
4. Flag students and hand the list to a counselor

No cloud infrastructure, no GPUs, no data science team required.

---

## Limitations

- Single dataset source — generalizability to other institutions is untested
- No missing values in dataset — unrealistic for real-world gradebooks
- Binary framing erases the spectrum between "slightly behind" and "completely lost"
- Single time snapshot — no temporal trajectory analysis

## Future Scope

- **Temporal prediction** — flag students at 25%, 50%, 75% of course completion
- **Multi-institutional validation** across diverse universities and curricula
- **Qualitative signals** — teacher notes, peer observations, LMS behavioral patterns

---

## Key References

1. S. K. Malik, "Slow learners: Their psychology and educational programmes," *Int. J. Educ. Psychol. Res.*, 2012.
2. C. Romero & S. Ventura, "Educational data mining and learning analytics: An updated survey," *WIREs*, 2020.
3. P. Kaur et al., "Classification and prediction based data mining to predict slow learners," *Procedia CS*, 2015.
4. S. M. Lundberg & S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.
5. M. Adnan et al., "Predicting at-risk students at different percentages of course length," *IEEE Access*, 2021.
6. A. Shamim, "Student performance and learning style dataset," *Kaggle*, 2024.

---

**Bahaar Sharma** | 2210990209 | bahaar209.be22@chitkara.edu.in  
Department of Computer Science and Engineering, Chitkara University, Punjab
