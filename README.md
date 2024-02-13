<h2 align="center">
  <b>CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision</b>

  <b><i>ICRA 2024</i></b>


<div align="center">
    <a href="TODO-PAPER-ARXIV-LINK" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="TODO-PROJECT-PAGE" target="_blank">
    <img src="https://img.shields.io/badge/Page-CalliRewrite-blue" alt="Project Page"/></a>
</div>
</h2>

This is the repository of [**CalliRewrite: Recovering Handwriting Behaviors from Calligraphy Images without Supervision**](TODO-PAPER-ARXIV-LINK).

CalliRewrite is an unsupervised approach enabling low-cost robotic arms to replicate diverse calligraphic 
glyphs by manipulating different writing tools. We use fine-tuned unsupervised LSTM to perform coarse stroke segmentation, and refine them through a reinforcement learning method to produce 
fine-grained control.

For more information, please visit our [**project page**](TODO-PROJECT-PAGE).

![CalliRewrite Teaser](demo/teaser.png)


## 📬 News

- **2024.2.18** Version 1.0 upload

## How to Use Our Code and Model:
We release our network and checkpoints. You can setup the pipeline under the following guidance.

### 0. Install dependencies
Due to package version dependencies, we need to set up two separate environments for coarse sequence extraction and tool-aware finetuning. To do this, you can follow these steps:

1. Navigate to the directory for coarse sequence extraction:
   ```bash
   cd seq_extract
   ```

2. Create a new conda environment using the specified requirements file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the newly created environment (optional):
   ```bash
   conda activate calli_ext
   ```
   
4. Navigate to the directory for sequence fine-tuning:
   ```bash
   cd ../rl_finetune
   ```

5. Create another new conda environment using the requirements file for fine-tuning:
   ```bash
   conda env create -f environment.yml
   ```

6. Activate the second newly created environment (optional):
   ```bash
   conda activate callli_rl
   ```
By following these steps, you will have two separate conda environments configured for coarse sequence extraction and sequence fine-tuning, ensuring that the correct dependencies are installed for each task.


### 1. Caliberate your own writing utensil

We provide three simple tools for modeling in the reinforcement learning environment: **Calligraphy brush**, **fude pen**, and **flat tip marker**. The first two are soft body with 2 degrees of freedom (DoF). The dynamics or kinematic model of tool motion (angle changing with movement direction) is provided by a simple simulation program. The latter is a rigid tool with 3 DoF and the dynamics is controlled by the wrist (servo motor). If you are going to customize the writing tool (strongly recommended if you want to conduct robotic demonstration), you need to customize the geometric properties (the ranges of $r$ , $l$ , and $\theta$ ) and dynamics (how $\theta$ change with motion), referring to implementations in `Finetune/tools/tool_config.yaml` and `Finetune/tools/tool_class.py`.

For calibrating the geometric properties $r$ and $l$ of the tool, we provide a control script using the Dobot Magician robotic arm to write with a brush as an example. It allows the robotic arm to draw lines on paper at different z-axis heights, and by measuring the thickness of the strokes, one can fit the relationship with z.

### 2. Download pretrained models

### 3. Coarse Sequence Extraction

### 4. Tool-Aware Finetuning

### 5. Visualization and robotic demonstration


## Train on your own data


## Citation
