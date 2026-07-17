**Weekly Design/Implementation/Experiment Plan**

## ---

## Apr 29, 2026

## **Project Context and Current Scope for Neuro SDE**

1. The long-term project goal is to build a physiological simulation component that can later support intervention modeling, such as adding a music-control term and eventually being used inside a larger decision-making or RL framework.  
2. The original proposal formulates the physiological simulator as a continuous-time dynamics model with a baseline term f\_base, a future control term f\_ctrl​, and a stochastic term for biological noise.  
3. The scope of the current phase is narrowed to **f\_base​ only**. We want to learn the unconditional baseline physiological dynamics before adding any intervention effect. The immediate goal is to produce a clean, reusable baseline dynamics module that can be expanded later.  
4. The idea is to model baseline physiology as a **continuous-time latent dynamics system** so that:  
   1. irregularly sampled physiological data can be handled naturally,  
   2. the model can be queried at arbitrary future timestamps,  
   3. and the same architecture can later be extended with control inputs and stochastic diffusion.  
5. The current implementation plan is to focus on the ECG waveform itself as the primary signal, rather than using hand-crafted physiological features as the main modeling target. Features such as RR/IBI/HR may be used as supplementary constraints or evaluation signals.  
   1. **RR**: time between consecutive R-peaks in an ECG beat sequence  
   2. **IBI**: inter-beat interval, basically the same idea as RR in this context  
   3. **HR**: heart rate, usually derived from RR/IBI, in beats per minute  
6. The current plan is to use:  
* an encoder that reads raw ECG waveform segments,  
* a latent continuous-time dynamics module that models how the hidden physiological state evolves,  
* and a decoder / prediction head that reconstructs or predicts future ECG waveform segments.  
  In this design, the latent space is intended to capture the underlying physiological state, while the waveform serves as the main observable target that supervises whether the latent dynamics are meaningful.  
  The current phase is intended to be compatible with future expansion. After a usable baseline dynamics module is established, future work can add:  
* a control term f\_ctrl,  
* stochastic diffusion / latent SDE behavior,  
* and eventually simulator-based planning or RL.

## **Implementation**

### **1\. Waveform input / preprocessing module**

**Functionality**

* Read and clean raw ECG segments before model input.  
* Normalize amplitude and optionally resample to a common sampling rate.  
* Optionally extract peaks for auxiliary supervision and evaluation mentioned above(low priority).

### **Input**

* ### `raw ECG waveform`

* ### `sampling rate`

* `lead_selection`

### **Output**

* `waveform_clean`

* ### `timestamps`

* ### `Metadata`

* ### `sampling_rate`

* ### `lead_selection`

* optional `r_peaks`  
* optional `aux_signals` such as cleaned lead, rate, and quality score

**Specific tools**

* Use NeuroKit2 for Python-side ECG preprocessing. `ecg_clean()` provides multiple cleaning methods, `ecg_peaks()` performs R-peak detection, and `ecg_process()` bundles cleaned signal, heart rate, signal quality, and delineation outputs into one processing pipeline.

**Notes**

* Peak detection here is mainly for possible tasks like filtering bad segments or building auxiliary losses.

---

### **2\. Segment and query builder module**

**Functionality**

* Convert long ECG records into training examples with a context and prediction pair.  
* Support arbitrary context and prediction windows.  
* Support irregular or custom future query timestamps, which is important for the continuous-time formulation.  
* Optionally randomize anchor points during training.

**Input**

* `waveform_clean`  
* `timestamps`  
* parameters such as:  
  * `context_window`  
  * `prediction_window`  
  * `start_point(or percentage)`  
  * `end_point`  
  * `anchor_point`  
  * `randomize`  
  * `query_times`

**Output**

* `context_waveform`  
* `context_timestamp`  
* `target_waveform`  
* `target_timestamp`

**Notes**

* Implement this as a PyTorch `Dataset` / `DataLoader` layer plus NumPy slicing utilities.  
* It should remain independent from the encoder and dynamics code.

---

This paper is used as a reference for some of the design: [Continuous-time Autoencoders for Regular and Irregular Time Series Imputation](https://arxiv.org/pdf/2312.16581)

## **3\. Continuous-time context encoder module**

### **Functionality**

* Convert the past ECG waveform context into a latent initial state z\_t0​​.

  ### **Input**

* `context_waveform`  
   Past ECG waveform context, shaped as `[batch, time, leads]`.  
* `context_timestamp`  
* `Sampling_rate`  
* `(optional)lead_metadata`

  ### **Output**

* `z_t0`  
   Latent initial state at the time t\_t0​, shaped as `[batch, latent_dim]`.

  ### **Notes**

Implement this module using a Neural CDE / NCDE-style encoder.

The encoder first constructs a continuous ECG path X(t) from the discrete waveform samples, then evolves a hidden state while reading this path.

Possible tools:

* `torchcde` for Neural CDE implementation

The encoder should be causal: it only reads the context interval before the prediction target.

---

## **4\. Baseline drift module fbase​**

### **Functionality**

* Model the deterministic baseline latent dynamics.  
* This is the core module of the current phase. It should learn how the latent physiological state evolves over continuous time.

  ### **Input**

* `z_t`  
   Current latent state, shaped as `[batch, latent_dim]`.  
* `t`  
   Current continuous time value.

  ### **Output**

* `dz_dt`  
   Latent derivative, shaped as `[batch, latent_dim]`.  
  ---

  ## **5\. Continuous-time solver module**

  ### **Functionality**

* Evolve the latent state forward from the anchor time to the requested future timestamps.  
* This module gives the model its continuous-time prediction ability. Instead of predicting only a fixed future grid, the solver evaluates the latent state at arbitrary query times.

  ### **Input**

* `z_t0`  
* `target_timestamp`  
* `f_base`  
   Baseline drift function used for latent state evolution.

  ### **Output**

* `latent_trajectory`  
   Latent states evaluated at the requested future timestamps, shaped as `[batch, target_time, latent_dim]`.

  ### **Notes**

Use `torchdiffeq` for the first implementation.

The solver should support arbitrary future timestamps.

This module should remain independent from the encoder and decoder so that the dynamics block can be extended later.

---

## **6\. Continuous waveform decoder module**

### **Functionality**

* Decode the future latent trajectory into ECG waveform predictions.  
* The decoder maps each future latent state into waveform values at the requested timestamps. The main output target is the ECG waveform itself.

  ### **Input**

* `latent_trajectory`  
* `target_timestamp`  
* `(Optional)lead_metadata`

  ### **Output**

* `predicted_waveform`  
   Predicted future ECG waveform, shaped as `[batch, time, leads]`.

  ### **Notes**

For multi-lead ECG, the decoder should preserve lead-specific output structure.

---

## **7\. Physiology evaluation module(Low priority)**

### **Functionality**

* Extract physiological measurements from predicted and target waveforms for supplementary evaluation.  
* This module checks whether the generated ECG waveform is physiologically plausible.

  ### **Input**

* `predicted_waveform`  
* `Target_waveform`  
* `sampling_rate`

  ### **Output**

* `R_peaks_predicted`  
* `R_peaks_target`  
* `rr_hr_metrics`  
   Supplementary RR / IBI / HR comparison metrics.  
* `physiology_summary`  
   Summary of physiological plausibility checks for debugging.

  ### **Notes**

Use `NeuroKit2` for ECG peak detection and auxiliary physiological analysis.

Useful functions:

* `nk.ecg_peaks()`  
* `nk.ecg_process()`

The waveform prediction loss remains the main training target. These extracted features are used only as supplementary evaluation or optional regularization and have lower priority.

---

**8\. Training and evaluation**

### **Functionality**

* Train the full baseline prototype end-to-end.  
* The training pipeline includes:  
1. continuous-time context encoder,  
2. baseline drift module fbase​,  
3. continuous-time solver,  
4. waveform decoder.

The model is trained to predict future ECG waveform from past ECG waveform context.

### **Input**

* `Context_waveform`  
* `Context_timestamp`  
* `target_waveform`  
* `target_timestamp`  
* `model_config`

  ### **Output**

* `training_loss`  
* `checkpoints`  
   Saved model checkpoints.  
* `(Optional) validation_metrics`  
   Waveform-level and auxiliary physiology metrics.  
* `(Optional) prediction_plots`  
   Visual comparisons of context, target, and predicted waveform.

