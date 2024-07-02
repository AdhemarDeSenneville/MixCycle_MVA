Hi Stefan





# Authors : 
- SENNEVILLE Adhemar (MVA)

# Work overview
As part of the class of G. RICHARD and R. BADEAU, I studied the paper **[Style Transfer of Audio Effects with
Differentiable Signal Processing](https://arxiv.org/abs/2207.08759)** from Adobe Research and Queen Mary University.

The main contibrution of that repository is the adaptation of the WaveShaper Pluging from FL-Studio in a Pytorch differenciable version with high expressivity.

## Wave Shaper
The WaveShaper is a plugin that apply a function $f: [0,1] \rightarrow [0,1]$ to all sample of an audio signal. This function is usuly shaped by the utilisator using besier curves or all sortes of interpolations.
![avering](https://github.com/AdhemarDeSenneville/DDSP_WaveShaper/blob/main/fig/waveshaper.jpg?raw=true)


I also included all the images used for the generation of the WaveShaper dataset.

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/WaveShaper_dataset.png)

To my knoledge, the WaveShaper, wile beeing simple, has never been model as a DDSP, this is du to it expresivity that is in theory infinit (one parameter for each value between 0 and 1). 
This are 4 examples of possible utilisation of the WaveShapes in a production Pipeline

- **Gain**: $f(x) = ax$
- **Bit Cruncher**: $f(x) = \left\lfloor \frac{|ax|}{a} \right\rfloor$
- **Saturation**: $f(x) = \max(x, a)$
- **Overdrive**: $f(x) = \tanh(x)$

This plugin could be very useful in the style tranfere a certain gendra relying extensivly on saturation and distortion of sounds.

I tried implementing the waveshaper as a MLP, whever seeing the poor results I opted for 
Using parametric interpolation on 2D points, it was possible to create a Differentiable WaveShaper with high expressivity 

- $MLP$ - $f$ is modeled as a Multi Layer Perceptron
- $LIX_n$ - 
- $DIX_n$ - 
- $DIYX_n$ - 

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_3.png)

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_1.png)

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_2.png)

# Simple utilisation 

The code is made using **Pytorch**. Here is the example of a differentiable processing pipeline using (in serie) a Parametric Equilizer From the Transfere style paper and The WaveShaper.

```python
import torch
from tslearn.datasets import UCR_UEA_datasets
from DTWLoss_CUDA import DTWLoss

# load data
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("SonyAIBORobotSurface2")
from DTWLoss_CUDA import DTWLoss

# convert to torch
X_train = torch.from_numpy(X_train).float().requires_grad_(True)
loss = DTWLoss(gamma=0.1)
optimizer = # your optimizer

##############
# your code ##
##############

value = loss(X_train[0].unsqueeze(0), X_train[1].unsqueeze(0))
optimizer.zero_grad()
value.backward()
optimizer.step()
```

# Experiments

## Training on using Waveshaper
![avering](https://github.com/b-ptiste/dtw-soft/assets/75781257/b1373a3a-f1b7-4ea3-8701-912d511f7c72)


# Credit

[Style Transfer of Audio Effects with Differentiable Signal Processing by Christian J. Steinmetz and Nicholas J. Bryan and Joshua D. Reiss, 2022](https://arxiv.org/abs/2207.08759)
