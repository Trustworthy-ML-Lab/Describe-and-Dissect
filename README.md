# Describe-and-Dissect (DnD)

* This is the official repository for [Describe-and-Dissect](https://arxiv.org/pdf/2403.13771).
  * DnD is a novel method to describe the roles of hidden neurons in vision networks with higher quality than existing neuron-level interpretability tools. 
  * DnD is **training-free**, providing **generative natural language description**, and can easily leverage more capable general purpose models in the future.
* Below we illustrates the pipeline of DnD (*top*) and the results provided by DnD and other methods with human evaluation scores (*bottom*). If you are interested in learning more about DnD, please see [our project website](https://lilywenglab.github.io/Describe-and-Dissect/index.html).
  
<p align="center">
<img src="data/github_overview_fig.png" alt="drawing" width="700"/>
</p>

<p align="center">
<img src="data/DnD_Overview_Fig.png" alt="overview" width="700"/>
</p>
  



## Sources:
* CLIP-Dissect: https://github.com/Trustworthy-ML-Lab/CLIP-dissect
* CLIP: https://github.com/openai/CLIP
* BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base
* GPT-3.5 Turbo: https://platform.openai.com/docs/models/gpt-3-5
* Stable Diffusion: https://huggingface.co/runwayml/stable-diffusion-v1-5


## Cite this work
N. Bai<sup>1</sup>, R. Iyer<sup>1</sup>, T. Oikarinen, and T.-W. Weng, [*Describe-and-Dissect*](https://arxiv.org/pdf/2403.13771), arxiv preprint.

```
@article{bai2024describe,
  title={Describe-and-Dissect: Interpreting Neurons in Vision Networks with Language Models},
  author={Bai, Nicholas and Iyer, Rahul A and Oikarinen, Tuomas and Weng, Tsui-Wei},
  journal={arXiv preprint arXiv:2403.13771},
  year={2024}
}
```
