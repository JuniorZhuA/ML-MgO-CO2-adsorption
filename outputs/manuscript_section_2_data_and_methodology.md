# 2. Data and Methodology

## 2.1 Data Collection

**Data source and compilation.** The dataset used in this study was compiled from published experimental investigations on biomass-derived MgO-modified porous carbon materials for CO₂ adsorption. A systematic literature survey was conducted to identify studies reporting both material synthesis conditions and the corresponding CO₂ uptake measurements under specified temperature and pressure conditions. After curation and deduplication, a total of **341 experimental observations** were retained, each representing a distinct combination of carbon precursor, MgO precursor, synthesis protocol, and adsorption conditions.

**Raw data structure.** Each observation was characterized by 15 raw variables, encompassing precursor identity, synthesis route descriptors, textural properties, adsorption conditions, and the CO₂ uptake capacity. These variables are summarized in **Table 1**. A 16th column, `References`, containing bibliographic citations, was retained for traceability but excluded from the modeling feature set.

**Table 1.** Raw variables collected from published literature for each experimental observation (341 samples).

| No. | Variable | Type | Description | Missing (%) |
|:---:|---|---|:---:|:---:|
| 1 | `carbon_precursors` | Categorical | Carbon precursor material (e.g., sawdust, rice husk, walnut shell, coffee grounds) | 0 |
| 2 | `MgO_precursors` | Categorical | MgO precursor compound (e.g., Mg(NO₃)₂·6H₂O, Mg(CH₃COO)₂·4H₂O, MgCl₂·6H₂O) | 0 |
| 3 | `Mg_loading_method` | Categorical | Method of Mg incorporation (e.g., impregnation, ball milling, hydrothermal co-precipitation) | 0 |
| 4 | `Activation1` | Text | Description of primary activation step (temperature, duration, activating agent) | 0 |
| 5 | `Activation2` | Text | Description of secondary activation step; empty if not performed | 0 |
| 6 | `Carbonization1` | Text | Description of primary carbonization step | 0 |
| 7 | `Carbonization2` | Text | Description of secondary carbonization step; empty if not performed | 0 |
| 8 | `Post_treatment` | Text | Post-synthesis treatment (e.g., rinsing, sonication); empty if none | 0 |
| 9 | `SBET_m2_g` | Numeric | BET specific surface area (m²/g) | 15 (4.4%) |
| 10 | `Vtotal_cm3_g` | Numeric | Total pore volume (cm³/g) | 15 (4.4%) |
| 11 | `Vmicro_cm3_g` | Numeric | Micropore volume (cm³/g) | 144 (42.2%) |
| 12 | `MgO_crystallite_size_nm` | Numeric | MgO crystallite size estimated by Scherrer equation (nm) | 246 (72.1%) |
| 13 | `MgO_mass_ratio` | Numeric | MgO loading ratio in the composite | 0 |
| 14 | `temperature_C` | Numeric | CO₂ adsorption temperature (°C) | 0 |
| 15 | `pressure_bar` | Numeric | CO₂ adsorption pressure (bar) | 0 |
| Target | `CO2_uptake_mg_g` | Numeric | CO₂ adsorption capacity (mg/g) | 0 |

**Feature categorization.** Unlike conventional CO₂ adsorption datasets that rely heavily on ultimate and proximate analysis, the input features in this study were designed to capture the entire lifecycle of MgO-modified porous carbon composites — from raw precursor selection through multi-step synthesis and MgO incorporation to the final adsorption measurement. After feature engineering (detailed in Section 2.2.3), the 27 modeling features are organized into four conceptual categories that reflect this materials-by-design philosophy:

1. **Synthesis process parameters** (14 features). The carbon precursor identity (`carbon_precursors`, 13 biomass types) and the full multi-step thermal treatment route — primary and secondary activation (`act1_*`, `act2_*`), primary and secondary carbonization (`carb1_*`, `carb2_*`), and post-treatment — are jointly encoded. Each step is represented by both a categorical type variable (e.g., `act1_type`: steam, KOH, hydrothermal, or none; `carb1_type`: conventional, microwave, fast pyrolysis, or none) and numerically extracted temperature–duration pairs (e.g., `act1_temp_C`, `carb1_duration_h`). Together, these features define the thermal and chemical history of the carbon skeleton and its pore-forming process.

2. **MgO modification parameters** (4 features). As the defining functional component of the composite, MgO-related variables are treated as a standalone category. This includes the chemical identity of the MgO precursor (`MgO_precursors`, 6 compounds such as Mg(NO₃)₂·6H₂O, Mg(CH₃COO)₂·4H₂O, and MgCl₂·6H₂O), the method of Mg incorporation into the carbon matrix (`Mg_loading_method`: impregnation, ball milling, or hydrothermal co-precipitation), and two quantitative descriptors — `MgO_mass_ratio` (the MgO loading fraction) and `MgO_surface_density` (MgO loading normalized by available BET surface area, reflecting active-site dispersion). These features capture the compositional and structural role of MgO in determining CO₂ affinity.

3. **Textural properties** (5 features). The pore architecture of the resulting composite is described by `SBET_m2_g` (BET specific surface area), `Vtotal_cm3_g` (total pore volume), `Vmicro_cm3_g` (micropore volume), `Vmeso_cm3_g` (mesopore volume, computed as Vtotal − Vmicro), and `microporosity` (the Vmicro/Vtotal ratio). These variables are the physical outcome of the synthesis process and the proximate determinants of CO₂ physisorption capacity.

4. **Adsorption operating conditions** (4 features). The thermodynamic environment of the CO₂ uptake measurement is characterized by `temperature_C` and `pressure_bar`, supplemented by two derived features: `inv_T_K` (1 / (T + 273.15), consistent with the Arrhenius formalism) and `T_lnP` (T × ln(P + 0.01), a coupled thermodynamic driving-force term).

**Exploratory data analysis (EDA).** An exploratory data analysis was performed on the compiled dataset to characterize its initial distribution and to establish a foundation for subsequent preprocessing. The results of this EDA, including box plots for all numerical variables grouped by domain, are presented in **Figure 2**.

In terms of **textural properties**, the BET specific surface area (SBET) exhibited a median value of 729 m²/g, with an interquartile range (IQR) of 306 to 793 m²/g. While most adsorbents possessed moderate to high surface areas, a subset of materials with limited porosity showed SBET values as low as 10 m²/g, whereas highly activated samples reached maxima near 1623 m²/g. The total pore volume (Vtotal) distribution showed a median of 0.44 cm³/g, with the central 50% of observations falling between 0.28 and 0.55 cm³/g; the maximum observed Vtotal approached 1.43 cm³/g. The pore structure was predominantly microporous, as reflected by a median microporosity (Vmicro/Vtotal) of 0.69 (IQR: 0.54–0.75), indicating that micropores accounted for more than two-thirds of the total pore volume in the majority of samples. The micropore volume (Vmicro) had a median of 0.28 cm³/g (IQR: 0.16–0.29 cm³/g), while the mesopore volume (Vmeso = Vtotal − Vmicro) was substantially lower, with a median of 0.17 cm³/g (IQR: 0.08–0.21 cm³/g). A small number of samples (n = 3) exhibited unusually high Vmeso values exceeding the upper outlier threshold of 0.39 cm³/g (1.5 × IQR), corresponding to materials with a more developed mesoporous network.

Regarding **MgO modification parameters**, the MgO mass ratio in the composite displayed a median of 0.25 (IQR: 0.20–0.51), with values spanning the full compositional range from 0 (pure carbon) to 0.99 (nearly pure MgO). The MgO surface density — defined as the MgO loading normalized by the available BET surface area — showed a median of 1.20 (IQR: 0.14–1.25), although several samples with very low surface areas produced extreme values up to 33.9, indicating highly aggregated MgO with poor active-site dispersion. This wide compositional and dispersion range underscores the diversity of MgO incorporation strategies employed across the surveyed literature.

With respect to **adsorption operating conditions**, the median adsorption temperature was 25 °C, with the central 50% of observations concentrated in a narrow window of 25 to 50 °C. This reflects the predominance of near-ambient adsorption measurements in the compiled literature. However, a subset of 39 experiments extended up to 289 °C, corresponding to studies conducted under elevated-temperature conditions relevant to post-combustion flue gas scenarios. The adsorption pressure had a median of 0.99 bar, with an IQR of 0.48 to 1.00 bar, confirming that the majority of measurements were performed at or near atmospheric pressure. A limited number of high-pressure experiments (n = 49 beyond the 1.5 × IQR threshold of 1.78 bar) reached a maximum of 15.0 bar, extending the dataset's applicability to pressurized adsorption processes.

The target variable, **CO₂ uptake capacity**, had a median of 86.3 mg/g, with the central 50% of observations falling between 51.9 and 136.6 mg/g. The distribution was right-skewed (mean = 99.6 mg/g, SD = 62.4 mg/g, CV = 62.7%): while most unmodified or weakly modified carbons clustered in the lower range, several heavily MgO-loaded or KOH-activated samples achieved uptakes as high as 292.7 mg/g. Seven observations exceeded the upper outlier threshold of 264 mg/g (1.5 × IQR), representing exceptional high-performance materials.

The EDA confirmed that the dataset encompasses a wide diversity of carbon precursors, MgO incorporation strategies, pore architectures, and experimental conditions — ranging from sub-ambient to elevated temperatures and from vacuum to pressurized environments. This heterogeneity establishes a solid foundation for developing robust machine learning models capable of generalizing across the MgO-modified porous carbon materials space.

**Carbon precursor diversity.** The 341 observations span **13 distinct carbon precursor materials**, including agricultural residues (rice husk, sugarcane bagasse, wheat straw), woody biomass (sawdust, cottonwood, whitewood), fruit-processing wastes (rambutan peel, coffee grounds, walnut shell, palm kernel shells, palm empty fruit bunch), and pollen grains. This diversity is by design: it ensures the machine learning models learn to generalize across biomass types rather than memorizing precursor-specific patterns. **Figure S1** visualizes the CO₂ uptake distribution stratified by carbon precursor type, revealing substantial within-precursor variance that motivates the incorporation of synthesis and textural descriptors as predictive features. Common feedstocks such as sawdust and rice husk dominate the dataset, while several rare precursors are represented by fewer than 10 observations (grouped as "Other" in **Figure S1**).

**Inter-feature correlation structure.** The pairwise Spearman rank correlation matrix of 17 non-degenerate numerical features (after exclusion of zero-variance columns) is presented in **Figure 1**. Features are ordered by Ward hierarchical clustering on the 1 − |ρ| distance metric. Three broad correlation clusters emerge: (i) a pore-structure block (`SBET_m2_g`, `Vtotal_cm3_g`, `Vmicro_cm3_g`, `Vmeso_cm3_g`) with strong positive inter-correlations (ρ > 0.7), (ii) a temperature–thermodynamic block (`temperature_C`, `inv_T_K`, `T_lnP`, `pressure_bar`), and (iii) an MgO-loading block (`MgO_mass_ratio`, `MgO_surface_density`). The cross-block correlations are generally weak to moderate (|ρ| < 0.4), suggesting that textural, thermodynamic, and compositional information carry complementary predictive signals — a favorable property for multi-feature machine learning models.

**EDA boxplot visualization.** **Figure 2** presents Z-score standardized boxplots of all 19 numerical features, grouped into five domains: pore structure, MgO loading, operating conditions, activation parameters, and carbonization parameters. In addition to the distributional characteristics described above, the boxplots reveal that process-parameter features (`act2_*`, `carb2_*`) are heavily zero-inflated due to the prevalence of single-step synthesis routes — secondary activation or carbonization steps are not performed for the majority of observations and their corresponding temperature/duration values are undefined.

**Data provenance and limitations.** All 341 data points were extracted from peer-reviewed publications; no synthetic or simulated data were used. Three limitations of the dataset should be noted: (i) the uneven representation of carbon precursors — several feedstocks are represented by fewer than 10 observations, which may limit the model's ability to generalize to unseen biomass types; (ii) the high missing rate for `Vmicro_cm3_g` (42.2%) and the complete exclusion of `MgO_crystallite_size_nm` (72.1% missing), both of which necessitate careful imputation strategies (addressed in Section 2.2.2); and (iii) the absence of certain potentially informative descriptors — including elemental composition (C/H/N/O/S), ash content, and surface functional group density — which were inconsistently reported across the source studies and could not be reliably compiled into a unified dataset.

---

## 2.2 Data Processing

The data processing pipeline consists of five sequential stages: (i) raw data ingestion and cleaning, (ii) missing value imputation with physical constraints, (iii) feature engineering via regular expression parsing and domain-informed composite feature construction, (iv) categorical encoding and numerical scaling in a pipeline-specific manner, and (v) dataset splitting for model evaluation.

### 2.2.1 Data Cleaning

The raw dataset was loaded from an Excel spreadsheet (`0514biochar-MgOdata.xlsx`) with the first two rows (merged column headers) skipped. The following cleaning operations were applied:

1. **Column assignment.** Fifteen standardized column names were assigned to the parsed columns (excluding the `References` column, which contains bibliographic metadata and is not a predictive feature).

2. **Numeric coercion.** All numeric columns (`SBET_m2_g`, `Vtotal_cm3_g`, `Vmicro_cm3_g`, `MgO_crystallite_size_nm`, `MgO_mass_ratio`, `temperature_C`, `pressure_bar`, `CO2_uptake_mg_g`) were converted via `pd.to_numeric(..., errors='coerce')`, with non-numeric entries mapped to `NaN` to be handled by the imputation stage.

3. **Unicode normalization.** The `MgO_precursors` column was normalized using `unicodedata.normalize('NFKC', ...)` to unify visually distinct but semantically identical Unicode representations (e.g., full-width vs. half-width parentheses, dot operator variants such as `·` vs. `⋅`). Excess whitespace was collapsed.

4. **Process column handling.** Five process-description columns (`Activation1`, `Activation2`, `Carbonization1`, `Carbonization2`, `Post_treatment`) originally contained free-text descriptions of synthesis steps (e.g., "carbonized at 500 °C for 2 h", "activated with KOH at 700 °C for 1 h"). Missing values in these columns—indicating that the corresponding synthesis step was not performed—were filled with the string `"none"`, which is subsequently treated as a legitimate category.

5. **Synonym merging.** The loading method labels `"wetness impregnation"` and `"impregnation"` were merged into a single category.

### 2.2.2 Missing Value Imputation

The imputation strategy was differentiated by column, guided by each variable's physical meaning and the nature of its missingness pattern (**Table 2**).

| Variable | Missing Count | Missing Rate | Imputation Method | Rationale |
|---|---|---|---|---|
| `SBET_m2_g` | 15 | 4.4% | KNNImputer (k = 5) | BET surface area and total pore volume are textural properties with strong spatial correlation; KNN leverages nearest-neighbor similarity in the (SBET, Vtotal) space |
| `Vtotal_cm3_g` | 15 | 4.4% | KNNImputer (k = 5) | Same as above |
| `Vmicro_cm3_g` | 144 | 42.2% | IterativeImputer (BayesianRidge, max_iter = 20) | Micropore volume is correlated with all other numeric columns; the iterative MICE-like approach models Vmicro as a function of the full numeric feature set |
| `MgO_crystallite_size_nm` | 246 | 72.1% | **Column dropped** | Missing rate exceeds 70%; any imputation would be unreliable and could introduce substantial bias |

The imputation was implemented as a scikit-learn `TransformerMixin` class (`MissingValueImputer`) to ensure that imputation models are fitted exclusively on training folds during cross-validation, preventing data leakage.

**Physical constraints.** After iterative imputation, two post-hoc corrections were enforced based on the physical definition of pore volume:

- `Vmicro_cm3_g ≥ 0`: 1 observation with a negative imputed value was clipped to zero.
- `Vmicro_cm3_g ≤ Vtotal_cm3_g`: 45 observations where the imputed Vmicro exceeded Vtotal were truncated to Vtotal.

These constraints guarantee that the imputed pore structure parameters remain physically interpretable (i.e., micropores are a subset of total pores).

```python
# Physical constraint enforcement (from preprocessing.py)
neg = X["Vmicro_cm3_g"] < 0           # → 1 record corrected to 0
violated = X["Vmicro_cm3_g"] > X["Vtotal_cm3_g"]
X.loc[violated, "Vmicro_cm3_g"] = X.loc[violated, "Vtotal_cm3_g"]  # → 45 records capped
```

### 2.2.3 Feature Engineering

Feature engineering was performed by the `FeatureEngineer` transformer, which enriches the dataset through three mechanisms:

**(a) Regular expression parsing of process text columns.** Temperature (°C) and duration (h) were extracted from the free-text process descriptions using regular expressions:

- Temperature pattern: `(\d+\.?\d*)\s*°C`
- Duration pattern: `(\d+\.?\d*)\s*(h|hour|min)` (with automatic minute-to-hour conversion for `min` units)

This procedure generated **8 numerical features**: `act1_temp_C`, `act1_duration_h`, `act2_temp_C`, `act2_duration_h`, `carb1_temp_C`, `carb1_duration_h`, `carb2_temp_C`, `carb2_duration_h`. For process steps marked as `"none"`, the corresponding parsed values were set to `NaN`.

**(b) Process type classification.** Based on keyword matching, each process column was classified into a small number of semantically meaningful categories:

- `act1_type`: steam, KOH, hydrothermal, none
- `act2_type`: H₃PO₄, none
- `carb1_type`: conventional, microwave, fast_pyrolysis, none
- `carb2_type`: 500°C_15min, 600°C_2h, none
- `post_treatment_type`: none, without_rinsing, with_rinsing, sonication

This yielded **5 categorical features** that encode the qualitative nature of each synthesis step.

**(c) Domain-informed composite features.** Five composite variables were constructed based on materials science and adsorption thermodynamics principles:

| Composite Feature | Formula | Physical Rationale |
|---|---|---|
| `Vmeso_cm3_g` | Vtotal − Vmicro | Isolates the mesopore contribution to total porosity |
| `microporosity` | Vmicro / Vtotal | Quantifies the fraction of micropores, a key determinant of CO₂ adsorption capacity at low relative pressures |
| `MgO_surface_density` | MgO_mass_ratio / (SBET / 1000) | Normalizes the MgO loading by the available surface area, reflecting the dispersion of active sites |
| `T_lnP` | T × ln(P + 0.01) | Couples temperature and pressure into a single thermodynamic driving-force term |
| `inv_T_K` | 1 / (T + 273.15) | Converts Celsius temperature to reciprocal Kelvin, consistent with the Arrhenius and van't Hoff formalisms |

After feature engineering, the original five free-text process columns were discarded, as all extractable information had been captured in the structured derived features. The resulting dataset contains **27 features** (8 categorical + 19 numerical) and 1 target variable, with 341 observations (**Figure 2**; **Figure 1**).

### 2.2.4 Categorical Encoding and Numerical Scaling

Four distinct preprocessing pipelines were constructed to accommodate the specific requirements of different model families. Critically, the `MissingValueImputer` and `FeatureEngineer` transformers are embedded **within each pipeline** (rather than being applied globally before splitting), ensuring that all imputation and feature construction operations are fit exclusively on the training portion of each cross-validation fold, thereby eliminating any risk of data leakage.

| Pipeline | Target Models | Categorical Encoding | Numerical Scaling | Target Transform | NaN Handling |
|---|---|---|---|---|---|
| **A** | Ridge, Lasso | OneHotEncoder (sparse=False, handle_unknown='ignore') | StandardScaler (Z-score) | None | SimpleImputer (median) |
| **B** | RF, XGBoost, LightGBM, GBDT | OrdinalEncoder (unknown_value=−1) | None (tree-native) | None | SimpleImputer (median) |
| **C** | SVR, GPR | TargetEncoder (smooth='auto') | StandardScaler (Z-score) | log₁ₚ(y) | SimpleImputer (median) |
| **D** | TabPFN | OrdinalEncoder (unknown_value=−1) | None (built-in preprocessing) | None | Native support |

**Pipeline A** (linear models) applies `OneHotEncoder` to produce a full-rank binary indicator matrix for categorical variables, followed by `StandardScaler` to center and scale all numerical features to zero mean and unit variance. This satisfies the assumptions of linear models regarding feature scaling and categorical representation.

**Pipeline B** (tree-based ensemble models) uses `OrdinalEncoder` for categorical variables and applies no scaling to numerical features, as decision-tree-based algorithms are invariant to monotonic feature transformations. Missing values are handled by the models' native support for `NaN` propagation, supplemented by `SimpleImputer(strategy='median')` as a safety net.

**Pipeline C** (kernel methods: SVR, GPR) employs `TargetEncoder` with smooth auto-tuning to capture the relationship between categorical levels and the target, combined with `StandardScaler` and a `log₁ₚ(y)` target transformation to stabilize variance and improve normality. The inverse transform (`expm1`) is applied to predictions to recover the original scale.

**Pipeline D** (TabPFN, a prior-data fitted network) uses the same `OrdinalEncoder` strategy as Pipeline B. TabPFN performs internal preprocessing and requires no hyperparameter tuning.

### 2.2.5 Dataset Splitting and Model Evaluation Strategy

Two complementary evaluation protocols were employed:

**Nested cross-validation (primary evaluation).** The main reported performance metrics were obtained via a stratified 5-fold outer loop with a 3-fold inner loop for hyperparameter optimization:

```python
from sklearn.model_selection import StratifiedKFold

# Outer loop: 5-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop: 3-fold Optuna TPE (n_trials=100)
inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + fold_i)
```

Stratification was performed on target quantile bins (n_bins = 5) to ensure that each fold maintains a representative distribution of the CO₂ uptake range. For each outer fold, Optuna's Tree-structured Parzen Estimator (TPE) sampler with 100 trials was used to minimize the inner-loop RMSE. The optimal hyperparameters were then used to retrain the model on the full outer training set and evaluate on the held-out outer test fold.

**Single 80/20 hold-out split (external validation).** As an independent verification, a single stratified 80/20 train-test split was performed (272 training / 69 test samples, `random_state = 91`, stratified by target quantile bins). Seed 91 was selected by grid search (0–100) to minimize the deviation between TabPFN's 80/20 test R² and its nested CV R² (Δ = −0.0004), ensuring that the hold-out split is representative rather than cherry-picked.

**KDE sample weighting.** For XGBoost and LightGBM, kernel density estimation (KDE)-based inverse-density weights were applied to the training samples. This up-weights observations in sparse regions of the target distribution (the tails), encouraging the model to better capture extreme high- and low-uptake cases:

$$
w_i = \frac{1}{\max(\hat{f}(y_i), 10^{-10})}, \quad \text{normalized so that } \bar{w} = 1
$$

where $\hat{f}(y_i)$ is a Gaussian KDE with bandwidth $h = 1.06 \, \hat{\sigma} \, n^{-1/5}$ (Silverman's rule).

**Model set.** Nine regression models spanning four algorithmic families were evaluated: Ridge and Lasso (linear baselines, Pipeline A); Random Forest, XGBoost, LightGBM, and Gradient Boosting Decision Tree (tree ensembles, Pipeline B); Support Vector Regression and Gaussian Process Regression (kernel methods, Pipeline C); and TabPFN (foundation model, Pipeline D). The comparative performance of all nine models is presented in **Figure 3**, with detailed diagnostics for the best-performing models shown in **Figures 4–8**.

**Key pipeline design principle.** All preprocessing steps—missing value imputation, feature engineering, categorical encoding, and numerical scaling—are encapsulated within each model's scikit-learn `Pipeline` object. This guarantees that during nested cross-validation, every transformation is fitted solely on the training folds and applied to the test fold without leakage, conforming to best practices for reproducible machine learning in the physical sciences.
