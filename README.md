## Overview
This is a PyTorch implementation for the paper [Continuous Online Learning-based CSI Feedback in Massive MIMO Systems](https://arxiv.org), which has been submitted to the IEEE for possible publication. The catastrophic forgetting problem in DL-based CSI feedback is first discussed in this paper. The test script and trained models are listed here and the key results can be reproduced as a validation of our work.

## Requirements

The following requirements need to be installed.
- Python == 3.9
- [PyTorch == 1.10.0](https://pytorch.org/get-started/previous-versions/#v1100)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from the [clustered delay line (CDL)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173) model and settings can be found in our paper. On the other hand, we provide a preprocessed dataset, which we adopt in the paper for testing the performance on avoiding catastrophic forgetting. You can download it from [Google Drive](https://drive.google.com/drive/folders/1yLzVBFR5rv3C_ym0PpDAnLyPOW5hiiCt?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XDewsqmvFBAHNCoYtJVwwg) with the password: swgm.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CFNet  # The cloned current repository
│   ├── dataset
|   ├── checkpoint # The checkpoints folder
|   |   ├── CDA_MT_4.pth
|   |   ├── ...
│   ├── models
│   ├── utils
│   ├── main.py
|   ├── run.sh  # The bash script
├── 3GPP  # CDL dataset generated following section A
│   ├── DATA_HtestA.mat
│   ├── ...
...
```

## Key Results Reproduction

The key results reported in Table II of the paper are presented as follows.

<table>
    <tr>
        <td rowspan="2"> CR </td> 
        <td rowspan="2"> Method </td> 
        <td colspan="3"> NMSE(dB) </td> 
        <td rowspan="2"> Checkpoint path <td>
   </tr>
    <tr>
  		  <td> C </td>
        <td> D </td> 
        <td> A </td>  
    </tr>
    <tr>
        <td rowspan="3"> 4 </td> 
        <td> MT-CL </td>
        <td> -15.19 </td> 
        <td> -9.687 </td>     
        <td> -15.86 </td>    
         <td> CDA_MT_4.pth </td> 
    </tr>
      <tr>
        <td> EU-CL </td>
        <td> -13.84 </td> 
        <td> -9.306 </td>     
        <td> -15.06 </td>   
        <td> CDA_EU_4.pth </td>  
    </tr>
          <tr>
        <td> OL </td>
        <td> -4.502 </td> 
        <td> -3.802 </td>     
        <td> -17.17 </td>    
        <td> CDA_OL_4.pth </td>  
    </tr>
        <tr>
        <td rowspan="3"> 16 </td> 
        <td> MT-CL </td>
        <td> -13.39 </td> 
        <td> -8.460 </td>     
        <td> -13.43 </td>    
        <td> CDA_MT_16.pth </td>  
    </tr>
      <tr>
        <td> EU-CL </td>
        <td> -11.24 </td> 
        <td> -6.500 </td>     
        <td> -10.44 </td>    
        <td> CDA_EU_16.pth </td>  
    </tr>
          <tr>
        <td> OL </td>
        <td> -4.956 </td> 
        <td> -2.823 </td>     
        <td> -16.34 </td>   
        <td> CDA_OL_16.pth </td>   
    </tr>
</table>

In order to reproduce the aforementioned key results, you need to download the given dataset and checkpoints. Moreover, you should arrange your project tree as instructed. An example of `run.sh` can be found as follows.

``` bash
python ./main.py \
  --cpu \
  --evaluate \
  --name 'CRNet' \
  --data-dir '/home/3GPP/' \
  --batch-size 100 \
  --workers 0 \
  --cr 4 \ # chosen from [4, 16]
  --scenarios 'CDA' \ # scenario changing sequence
  --pretrained './checkpoint/CDA_MT_4.pth' \
  2>&1 | tee log.out

```

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Please refer to it for more information.