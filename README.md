# Cyclic Mixture Permutation Invariant Training

[**Code**](./code/unsupervised-audio-separation.ipynb)
| [**Presentation**](CycleMix_Presentation.pptx)
| [**Report**](CycleMix_Report.pdf)
| [**Original Papers**](https://arxiv.org/pdf/2202.03875)

# Work overview

As part of the class of E. BACRY on [Audio signal processing – Time-frequency analysis](https://www.master-mva.com/cours/audio-signal-processing-time-frequency-analysis/), I studied the paper **[MixCycle: Unsupervised Speech Separation via
Cyclic Mixture Permutation Invariant Training](https://arxiv.org/pdf/2202.03875)** from Bogaziçi University. 
The paper introduce two unsupervised source separation
methods, which involve self-supervised training from singlechannel two-source speech mixtures.

The main contibrution of that repository is an implementation of all *Supervised* and *Unsupervised* technics seen in the paper. And an application to Speech denoizing and all the hypotesis relaxation that goes with that application.

## Delivrable 

- Report 
  - Part 1 describe and explain the paper
  - Part 2 present the experiments done 
- A presentation
- While not asked, the code is avalable in a single notbook. Be aware that the code is made to work on kaggle and using Weights and Biases API. That allows easy debugging and experimenting.


# Implementations (Trainings)

## Permutation Invariant Training (PIT)

In [PIT](https://arxiv.org/pdf/1607.00325), the separation loss is expressed as:

$$L_{\text{PIT}}(s, \hat{s}) = \min_{P} \sum_{m=1}^{2} -\text{SNR}(s_m, [P\hat{s}]_m),$$

Here, $P$ is a $2 \times 2$ permutation matrix. In this context, there is no predefined order of the source signals (sources are independent). The loss is computed using the permutation that yields the best match between ground-truth reference sources $s$ and estimated sources $\hat{s}$. There are only two $2 \times 2$ permutation matrix which makes the calculation of the loss function negligible compared to the calculation time of the neural network. The loss function minimization problem is therefore trivial to solve and does not pose a problem during training.

![avering](https://github.com/AdhemarDeSenneville/MixCycle_MVA/blob/main/fig/S_PIT.png?raw=true)

## Permutation Invariant Training with Dynamic Mixing (PIT-DM)

[PIT-DM]() is a data augmentation
technique that improves PIT by mixing sources differently each time during training. It is
possible thanks to the ground truth source separation available
![avering](https://github.com/AdhemarDeSenneville/MixCycle_MVA/blob/main/fig/S_PIT-DM.png?raw=true)

## Mixture Invariant Training (MixIT)
Permutation Invariant Training relies on ground truth source signals $s$. The paper [MixIT](https://arxiv.org/pdf/2006.12701) addresses this by considering two mixtures, $x_i$ and $x_j$, each comprising up to $2$ sources. Drawn randomly from an unsupervised dataset, these mixtures are combined to form a Mixture of Mixtures (MoM): $\bar{x} = x_i + x_j$. The separation model $f_\theta$ processes $\bar{x}$, outputting $2*2$ sources to ensure adequacy for any $\bar{x}$.

The unsupervised MixIT loss for estimated sources $\hat{s}$ and mixtures $x_i, x_j$ is defined as:
$$L_{\text{MixIT}}(x_i, x_j, \hat{s}) = \min_A \sum_{i=1}^{2} -\text{SNR}(x_i, [A\hat{s}]_i)$$
Here, $A$ is a $2 \times 4$ binary matrices with column sums to 1, allocating each source $\hat{s}_m$ to either $x_i$ or $x_j$. MixIT minimizes the total loss between the mixtures $x$ and the remixed separated sources $\hat{x} = A\hat{s}$, analogously to the process in PIT.

The main issue with MixIT is at inference.Because when you want to extract the two sources of a signal, you end up with a neural network that extract 4 sources. This increases the inference time without being useful.

![avering](https://github.com/AdhemarDeSenneville/MixCycle_MVA/blob/main/fig/S_MixIT.png?raw=true)

## Mixture Permutation Invariant Training (MixPIT) 
### (from the paper)

MixPIT addresses the over-separation issue inherent in MixIT by aligning the number of model outputs with actual source quantities. In MixPIT, the model, $f_\theta(s_{i 1}+s_{i 2}+ s_{j 1}+s_{j 2}) = f_\theta(x_i + x_j)$, is designed to process a mixture of four distinct sources but generates only two outputs. This model outputs three potential pair combinations: $\{\hat{s}_{i1}+\hat{s}_{i2}, \hat{s}_{j1}+\hat{s}_{j2}\}$, $\{\hat{s}_{i1}+\hat{s}_{j1}, \hat{s}_{i2}+\hat{s}_{j2}\}$, and $\{\hat{s}_{i2}+\hat{s}_{j1}, \hat{s}_{i1}+\hat{s}_{j2}\}$.
Due the statistical independence of sources $s_i$, $s_j$, $s_k$, and $s_l$, the likelihood of the three output pairs is equal. This equality arises because the model cannot differentiate between possible source combinations in the input mixtures.

The training of $f_\theta$ uses the PIT loss function defined as
$$L_{MixPIT}(x_i, x_j, \hat{S}) = L_{PIT}(x_i, x_j, \hat{S}),$$
where $\hat{S}$ represents one of the three output pairs.
In this framework, the PIT loss function $L_{PIT}$ selects the permutation minimizing the loss. Consequently, the first $(x_{i} , x_{j})$ output pair indicates a perfect source match, whereas the second and third are partial matches.
The design of the loss function ensures that at least two sources always match, enabling the model to effectively learn to separate two sources, even with noise from mismatched sources.

For MixPIT testing, the trained network is provided with a single mixture $f_\theta(x_{i})$, to extract the individual source estimates $\hat{s}_{i1}$ and $\hat{s}_{i2}$. By maintaining a balance between the number of model outputs and the actual number of sources, MixPIT effectively mitigates the over-separation issue. Even if the problem of over-separation is solved, the model is not trained on the pure separation of sources, which means that performances may not be very good during inference. MixCycle counterbalance this problem by introducing a 2nd training phase of the neural network.

![avering](https://github.com/AdhemarDeSenneville/MixCycle_MVA/blob/main/fig/S_MixPIT.png?raw=true)

## Cycle of Mixtures

MixCycle is inspired by the same principle as PIT-DM. It uses predicted ground truth of the network as a training example for the next step. These techniques are often used in semi-supervised scenarios where we use predicted results of the neural network and add them to the dataset.

Inspired by continuous learning, the network (with freeze parameters) generate new pseudo ground truth data:

$$\{\tilde{s}_{i1}, \tilde{s}_{i2}\} = f_{\theta}(x_{i}), \quad \{\tilde{s}_{j1}, \tilde{s}_{j2}\} = f_{\theta}(x_{j})$$

Now we can do a classic supervised training step like in PIT-DM. For example if we pick the pair $j1-i2$:

$$\hat{s} = f_{\theta}(\tilde{s}_{j1} + \tilde{s}_{i2}), \quad s = \{\tilde{s}_{j1},\tilde{s}_{i2}\}$$

$$L_{\text{MixCycle}}(s, \hat{s}) = \min_{P} \sum_{m=1}^{2} -\text{SNR}(s_m, [P\hat{s}]_m),$$

![avering](https://github.com/AdhemarDeSenneville/MixCycle_MVA/blob/main/fig/S_MixCycle.png?raw=true)

# Experiments

## Voice Denoising

Experiments focus on speech denoising, which is a special case of source separation. It implies certain hypotheses compare to the general case. Firstly, we consider that there are only two sources: the voice $v$ and the noise $n$ such that:

$$x = \sum_{i=1}^{N} s_i = v + n$$

The noise is not considered to be white or to have any color because it has a structure, it can be the background noise of a train station or of a restaurant. The voice as we know has also a structure. This means that the classical hypothesise of source independence no longueur hold. If the first source is a voice, then the second can only be a noise. However, we can change the equations to our special case, and show that all those training methods still works.

## Hypothesis relaxation

One important critic of MixPIT (and all unsupervised training procedures) is it strong hypothesis that assumes that sources $s_i, s_j , s_k, s_l$ are statistically independent from each others. 
Two significant real-world considerations can challenge its validity. Firstly, the acoustic environment in which the audio is recorded inherently imprints a shared acoustic signature upon the sources. This comprises similar resonance and impulse responses, like two voices recorded in the same room. Such environmental factors create acoustic characteristics that deviate from the assumption of statistical independence. Secondly, the heterogeneity in recording equipment quality and frequency response further complicates this assumption. Variations in microphone quality and their distinct frequency response profiles can lead to a form of frequency-dependent correlation between sources. This correlation manifests as a statistical dependency, driven by the equipment used in the recording process. In practical scenarios, the assumption of statistical independence may be overly simplistic and not fully representative of the complex inter-dependencies present in real-world audio recordings.

To investigate this identified limitations, I wanted to try an experiment wherein the audio quality in the dataset was altered to mimic the frequency response of a varying-quality microphone. This was achieved by applying a band-pass filter with variable bandwidth to simulate the effects of inferior recording equipment. The aim was to observe the impact of such alterations on the training process and the final quality of the model.

## Config

| Parameter       | Value      |
|-----------------|------------|
| GPU Used        | 1x P100    |
| Model           | ConvTasNet |
| Optimizer       | Adam       |
| Learning Rate   | 0.03       |
| Early Stopping  | 3 epochs   |

# Authors : 
- de SENNEVILLE Adhémar (MVA) (19/20)
  
# Credit

[MixCycle: Unsupervised Speech Separation via
Cyclic Mixture Permutation Invariant Training](https://arxiv.org/pdf/2202.03875)
