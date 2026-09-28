# Current PyTorch-Practice Simulation and LongGCN Workflow

## Overall goal

The `PyTorch-Practice` repository is currently serving as the simulation and benchmarking framework for evaluating the LongGCN approach for irregular longitudinal data.

The broader scientific question is whether a graph-based model that directly respects the observed longitudinal measurement structure can perform competitively with approaches that first impute missing measurements and then analyze a more conventional complete-data representation.

The main comparison is therefore not simply “GCN versus imputation.” Instead, the study asks how different treatments of incomplete longitudinal data affect downstream predictive performance when the prediction architecture is held as constant as possible.

The reusable mathematical and neural-network components are implemented separately in the `longgcn` Python package. `PyTorch-Practice` generates simulated data, introduces different forms of missingness, performs MICE3D imputation where appropriate, converts the resulting datasets into LongGCN representations, trains prediction models, and compares their performance.

---

## 1. Simulated longitudinal data

The starting point is a simulated irregular longitudinal dataset.

Each patient has a patient-specific sequence of observation times and five possible longitudinal measurements,

```math
x_1,\ldots,x_5.
```

Patients do not necessarily have the same number of observations or the same observation times. Thus, the data already reflect the type of irregular longitudinal structure that motivates the LongGCN methodology.

The underlying simulation contains latent patient-level disease information that affects the longitudinal measurements. Two patient-level prediction outcomes are then generated from this latent disease process.

The first is a binary classification outcome representing whether the patient experiences the simulated disease process.

The second is a continuous regression outcome related to the amount and intensity of the latent disease process.

Thus, exactly the same longitudinal predictor data can be evaluated under both a classification and a regression problem.

For the current run, the common patient split contains 700 training patients, 150 validation patients, and 150 test patients. Importantly, the same patient split is reused across every missing-data condition so that differences in predictive performance are not caused by different train/test samples.

---

## 2. Four original observation conditions

Starting from the simulated complete data, four original longitudinal datasets are constructed.

| Condition | Interpretation |
|---|---|
| `complete` | All five measurements are observed at each simulated visit. |
| `mcar` | Measurements are deleted according to an MCAR mechanism and left missing. |
| `group_specific` | Measurements are observed according to a prespecified group structure. Group 1 contains \(x_1,x_2,x_3\), while Group 2 contains \(x_4,x_5\). |
| `group_specific_mcar` | The structural group-specific pattern is retained, and additional MCAR missingness is imposed on measurements that would otherwise have been observed. |

The distinction between the last two forms of missingness is important.

In the group-specific setting, an absent measurement is not necessarily an accidentally missing value. Its absence can instead be part of the data-collection design. For example, if a visit belongs to Group 1, then \(x_4\) and \(x_5\) may simply not have been scheduled for collection at that visit.

The `group_specific_mcar` condition therefore contains two fundamentally different types of absent measurements: measurements absent because of the structural collection design, and measurements that should have been observed under that design but were subsequently deleted by MCAR.

This distinction lets the simulation evaluate whether it is useful to reconstruct genuinely missing observations without necessarily reconstructing measurements that were never scheduled to be collected.

---

## 3. MICE3D comparison conditions

MICE3D is then applied to selected incomplete datasets to create four additional model-input conditions.

This produces eight total conditions:

| Final condition | Starting data | What MICE3D is allowed to fill |
|---|---|---|
| `complete` | Complete | Nothing |
| `mcar` | MCAR | Nothing |
| `mcar_to_complete` | MCAR | MCAR values |
| `group_specific` | Group-specific | Nothing |
| `group_specific_to_complete` | Group-specific | Structural gaps |
| `group_specific_mcar` | Group-specific + MCAR | Nothing |
| `group_specific_mcar_to_complete` | Group-specific + MCAR | Structural gaps and MCAR values |
| `group_specific_mcar_to_group_specific` | Group-specific + MCAR | MCAR values only; structural gaps are restored afterward |

The final condition is particularly useful.

For

```math
\texttt{group\_specific\_mcar\_to\_group\_specific},
```

MICE3D internally performs its normal complete imputation, but after imputation the structurally absent cells from the corresponding `group_specific` dataset are returned to missing.

Consequently, the final dataset has exactly the original group-specific structure, but accidental MCAR missingness has been repaired.

This gives a particularly clean comparison:

```math
\text{group-specific + MCAR left missing}
```

versus

```math
\text{same structural design, but MCAR values imputed}.
```

There is also a more aggressive comparison in which MICE3D reconstructs everything, including the structurally absent measurements.

MICE3D is applied independently to the training, validation, and test sets as complete sets, following the package's native set-level operation. No outcome information is supplied to the imputation algorithm.

One methodological point worth remembering is that MICE3D's longitudinal Gaussian-process component operates using ordered visit index rather than the actual irregular time gaps. The original simulated time variable is nevertheless retained, and LongGCN subsequently uses the real observation times when constructing temporal edges.

---

## 4. Conversion to the LongGCN representation

Script `05_prepare_longgcn_datasets.py` converts all eight conditions into the representation required by the `longgcn` package.

The conceptual starting point is an observation-level representation in which only genuinely observed scalar values appear. Missing cells in the original wide dataset do not become artificial zero-valued observations; rather, they are absent from the observation-level data.

LongGCN then constructs a patient-specific time-by-measurement representation,

```math
X_i,
```

from those observations. Under the generalized formulation, unobserved measurements enter the resulting matrix representation as structural zeros produced by the observation-to-time and observation-to-measurement mappings.

For each patient, the package also constructs the temporal communication matrix

```math
T_i,
```

whose forward-time entries are based on

```math
\exp\left(
-\frac{\Delta t}{d}
\right).
```

Self-loops have weight 1, earlier observations can send information to later observations, and increasingly distant observations receive smaller weights.

For the current imputation comparison, the temporal decay parameter \(d\) is held fixed across every condition. This is deliberate: the primary experimental variable is the missing-data strategy, so changing temporal weighting at the same time would make the comparison harder to interpret.

The graph objects \(X_i\), \(T_i\), group selectors, and group-specific temporal matrices are precomputed during data preparation.

---

## 5. How measurement groups differ across conditions

The final structure of the dataset determines how measurements are grouped.

Datasets that are complete or complete-like use one measurement group containing all five variables:

```math
\{x_1,x_2,x_3,x_4,x_5\}.
```

These conditions include `complete`, `mcar`, `mcar_to_complete`, `group_specific_to_complete`, and `group_specific_mcar_to_complete`.

The fact that `mcar` still uses one group does not mean all values are observed. It means all five measurement types conceptually belong to the same measurement system; individual missing observations are simply absent.

Datasets that retain the designed measurement structure use two groups:

```math
G_1=\{x_1,x_2,x_3\},
\qquad
G_2=\{x_4,x_5\}.
```

These are `group_specific`, `group_specific_mcar`, and `group_specific_mcar_to_group_specific`.

For each designed group \(g\), the methodology constructs a group-restricted temporal matrix

```math
A_{ig}=P_{ig}T_iP_{ig},
```

so that temporal messages within the initial group-specific stage are transmitted only through compatible group structure.

---

## 6. The 48 prepared PyTorch datasets

Every one of the eight data conditions is prepared for both prediction tasks and all three data splits.

Therefore,

```math
8
\times
2
\times
3
=
48
```

serialized `LongGCNTorchDataset` objects are created.

The three datasets belonging to a single condition/task combination are simply the training, validation, and test components of one modeling experiment.

Thus, the actual number of prediction models being compared is

```math
8\times2=16.
```

There are eight classification fits and eight regression fits.

---

## 7. Current LongGCN prediction architecture

The prediction architecture is defined in `longgcn_models.py`.

The low-level graph construction and neural-network components come from the installed `longgcn` package, while `longgcn_models.py` simply assembles those components into the model used for this simulation.

The current architecture is

```math
\text{Initial latent transformation}
```

followed by

```math
\text{Group layer 1}
\rightarrow
\text{ReLU}
```

```math
\text{Group layer 2}
\rightarrow
\text{ReLU}
```

```math
\text{Temporal layer 1}
\rightarrow
\text{ReLU}
```

```math
\text{Temporal layer 2}
\rightarrow
\text{ReLU}
```

followed by patient-level pooling and a final linear prediction layer.

There are five possible input measurements. The initial latent representation has dimension 8, and all subsequent graph layers currently use hidden dimension 8.

The current pooling operation is max pooling.

After pooling, each patient is represented by a single fixed-length vector, and an ordinary linear layer converts that patient representation into one scalar prediction.

No task-specific output transformation is built into the neural network itself. For classification the scalar is interpreted as a logit, while for regression it is interpreted directly as a continuous prediction.

---

## 8. Relationship between group layers and temporal layers

An important feature of the construction is that the same architecture can be used for both complete and group-structured data.

For a complete-like dataset there is only one measurement group containing all measurements. Every real observation time therefore belongs completely to that single group.

Its group selector is effectively

```math
P_{i1}=I,
```

so

```math
A_{i1}
=
P_{i1}T_iP_{i1}
=
T_i.
```

Thus, for the single-group data, the nominal “group-specific” layers reduce to ordinary temporal graph layers.

The complete-data network can therefore be viewed as having four temporal message-passing layers.

For the structurally grouped data, the first two layers genuinely restrict communication according to the designed measurement groups, while the final two temporal layers allow unrestricted communication across the patient trajectory.

This means the depth of the network remains comparable across conditions; what changes is whether the first stage contains meaningful group restrictions.

---

## 9. Group-specific parameters and model size

The current model uses

```text
parameter_sharing = "group"
```

for the group-specific layers.

This means separate measurement groups receive separate learnable transformations \(W_g\) and biases \(b_g\). The software explicitly supports this group-specific parameterization.

As a consequence, the two-group models contain more learnable parameters than the one-group models.

Under the current 8-dimensional architecture, the one-group model contains 345 trainable parameters, whereas the two-group model contains 489.

This difference is being recorded as part of the experiment output rather than hidden.

It is therefore possible to report both predictive performance and model complexity together.

If a later experiment needs the exact same number of parameters between one-group and two-group models, the `longgcn` architecture also permits shared group parameters. That would be a separate sensitivity analysis rather than changing the current experiment.

---

## 10. Handling padded mini-batches

Patients have different numbers of observation times, so LongGCN uses padded mini-batches.

The model receives tensors such as

```math
X:
B\times K_{\max}\times M
```

and corresponding graph structures.

The batch also contains a `time_mask` identifying which positions correspond to genuine patient observations rather than padding.

This is important because the initial latent transformation contains a bias. A padded all-zero row could otherwise become nonzero after that transformation.

The model explicitly reapplies the time mask after the initial transformation and after graph layers, forcing padded latent positions back to zero. 

Pooling also respects this mask, so padded time positions do not contribute to the final patient representation.

---

## 11. Model training

Models are trained with Adam.

The current optimizer settings are

```math
\text{initial learning rate}=0.001
```

and

```math
\text{weight decay}=0.
```

The optimizer's `weight_decay` parameter is unrelated to the temporal decay parameter \(d\). `weight_decay` would regularize neural-network parameters, whereas \(d\) controls the graph-edge weight

```math
\exp(-\Delta t/d).
```

The training module uses binary cross-entropy with logits for classification and mean squared error for regression.

During each mini-batch the standard PyTorch sequence is used: forward propagation, calculation of the task-specific loss, backpropagation, and an Adam parameter update.

Training and validation losses are calculated as patient-level means. Mini-batch mean losses are weighted by the number of patients in the batch before constructing the overall epoch loss, so a smaller final batch does not receive disproportionate weight.

---

## 12. Validation-driven learning-rate adaptation

Training loss controls the parameter gradients, but it does not control the learning-rate scheduler.

Learning-rate adaptation is based entirely on validation loss.

The current scheduler is `ReduceLROnPlateau`.

The intended settings for the current models are:

```math
\text{initial LR}=0.001,
```

```math
\text{scheduler patience}=10,
```

```math
\text{scheduler factor}=0.5,
```

and

```math
\text{minimum LR}=10^{-6}.
```

Thus, if validation loss does not meaningfully improve for approximately ten epochs, the learning rate is halved.

For example,

```math
0.001
\rightarrow
0.0005
\rightarrow
0.00025
\rightarrow
0.000125
\rightarrow\cdots
```

as successive validation plateaus occur.

The scheduler evaluates validation loss after every epoch.

A validation-loss improvement of approximately

```math
10^{-4}
```

is being used as the meaningful-improvement threshold so that extremely small random fluctuations do not repeatedly reset the training logic.

---

## 13. Best-model selection and early stopping

The model used for final testing is not necessarily the model from the final training epoch.

Validation performance is checked every epoch.

Whenever a new lowest validation loss is observed, a deep copy of the current model parameters is retained as the current best model.

Training itself continues from the current parameter state rather than jumping back to that checkpoint.

At the end of training, the best-validation checkpoint is restored.

The maximum number of epochs is currently 500, but this should now be interpreted as a ceiling rather than the expected duration of every fit.

The planned early-stopping patience is 100 epochs without a meaningful validation improvement.

Conceptually, the procedure is therefore:

```math
\text{validation improves}
\Rightarrow
\text{retain the model and continue}
```

```math
\text{validation plateaus for 10 epochs}
\Rightarrow
\text{halve the learning rate}
```

```math
\text{validation fails to meaningfully improve for 100 epochs}
\Rightarrow
\text{stop training}
```

followed by restoration of the best validation model.

This avoids spending hundreds of epochs training at a nearly zero learning rate after the model has effectively converged.

---

## 14. Classification evaluation

For classification, the neural network produces a raw logit.

During evaluation only, the logit is transformed using the sigmoid function to obtain a probability. A threshold of 0.5 is currently used to obtain the predicted binary class.

The current classification summaries include loss, accuracy, balanced accuracy, sensitivity, specificity, and the full confusion-matrix counts.

Balanced accuracy is especially useful because it gives equal consideration to sensitivity and specificity rather than being dominated by the more common class.

Patient-level logits, probabilities, predicted classes, and true outcomes are also saved so that additional evaluations such as ROC curves can later be produced without retraining the models.

---

## 15. Regression evaluation

For regression, the model's raw scalar output is used directly.

Performance is summarized using

```math
\text{MSE},
\qquad
\text{RMSE},
\qquad
\text{MAE},
\qquad
R^2.
```



Patient-level true and predicted outcomes are also saved.

This permits later predicted-versus-observed plots, residual diagnostics, or other analyses without rerunning the neural networks.

---

## 16. Main scientific comparisons

The experiment contains several complementary comparisons.

The first benchmark is

```math
\texttt{complete}.
```

This provides a reference for predictive performance when no measurements are missing.

For conventional missingness, the key comparison is

```math
\texttt{mcar}
\quad\text{versus}\quad
\texttt{mcar\_to\_complete}.
```

This asks whether directly modeling the observed MCAR data using LongGCN performs differently from first reconstructing the missing values using MICE3D.

For structural missingness, the corresponding comparison is

```math
\texttt{group\_specific}
\quad\text{versus}\quad
\texttt{group\_specific\_to\_complete}.
```

This asks whether retaining and explicitly modeling the designed observation structure is competitive with imputing those structurally absent measurements and analyzing the resulting complete representation.

The most detailed comparison uses the group-specific + MCAR data:

```math
\texttt{group\_specific\_mcar},
```

```math
\texttt{group\_specific\_mcar\_to\_group\_specific},
```

and

```math
\texttt{group\_specific\_mcar\_to\_complete}.
```

These represent three distinct strategies:

leave both structural and accidental missingness untouched;

repair accidental MCAR while preserving the structural observation design;

or impute every absent measurement and analyze a complete representation.

That comparison should help distinguish the value of recovering genuinely missing information from the effect of inventing measurements at times when they were never intended to be observed.

---

## 17. Experimental controls

Several parts of the experiment are deliberately held constant.

The same patient train/validation/test split is used across all conditions.

The same temporal decay parameter \(d\) is used across conditions.

The same hidden dimensions, number of message-passing layers, activation function, pooling strategy, optimizer, starting learning rate, validation procedure, and evaluation metrics are used wherever the data structure permits.

Random-number generators are reset before each model fit so that models with identical architectures begin from comparable random states.

The test set is not used for optimization, learning-rate adjustment, or model selection. It is evaluated only after the best validation model has been selected.

The intention is therefore to make the missing-data representation the primary experimental difference.

---

## 18. Current outputs

For each of the 16 model fits, the modeling script is designed to retain the best model state, the complete training and validation loss histories, the learning-rate history, patient-level test predictions, model settings, parameter counts, and final test metrics.

The parameter counts are especially useful because the two-group models currently have greater capacity than the one-group models due to separate group-specific transformations.

A combined results table can therefore eventually contain fields such as condition, task, number of groups, number of trainable parameters, best epoch, best validation loss, test loss, classification metrics, or regression metrics.

This should make it possible to compare both predictive performance and the complexity of the model used to obtain that performance.

---

## 19. High-level interpretation of the study

At a high level, the project is evaluating two competing philosophies for incomplete irregular longitudinal data.

The first philosophy is:

```math
\text{repair the data first}
\rightarrow
\text{then fit a prediction model}.
```

MICE3D represents this strategy.

The second philosophy is:

```math
\text{retain the observed data structure}
\rightarrow
\text{let the neural network operate on that structure directly}.
```

LongGCN represents this strategy.

The LongGCN formulation does not require every patient, visit, or measurement group to have identical observation patterns. Instead, it creates a patient-specific graph representation from whichever measurements were actually observed and uses real temporal separation when controlling message propagation.

The simulation is therefore designed to test whether explicitly modeling the observation structure can reduce or eliminate the predictive benefit that would otherwise be obtained by imputing missing longitudinal measurements.

It is also possible that imputation will improve prediction. The purpose of the experiment is not to assume that one approach should win, but to identify which strategy is most effective under different forms of missingness.

---

## 20. Current status

The simulation-generation, missingness-generation, patient-splitting, MICE3D-imputation, and LongGCN data-preparation pipelines are now implemented.

All eight data conditions have been converted into classification and regression train/validation/test datasets, giving 48 prepared LongGCN dataset objects.

The next stage is the 16-model training experiment.

The immediate output of that stage will be a direct comparison of predictive performance across complete data, observed-only incomplete data, structurally incomplete data, and several MICE3D-imputed alternatives.

From there, the natural next steps are to summarize the performance differences, inspect variability across repeated simulations or random seeds if desired, investigate sensitivity to model hyperparameters such as the temporal-decay parameter \(d\), and determine whether the main patterns are stable enough to motivate a formal methodological simulation study.