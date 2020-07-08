# Data Pre-processing

Run `getdata.sh` under that directory to obtain and pre-process the data. This script will download and process the official data from UD. For CTB5, CTB6, CTB7, and CTB9, you need to obtain the official data yourself, and then put the raw data folder under the `data_preprocessing` directory. The folder name for the CTB datasets should be:

* CTB5: LDC05T01
* CTB6: LDC07T36
* CTB7: LDC10T07
* CTB9: LDC2016T13

This script will also download the [Stanford CoreNLP Toolkit v3.9.2](https://stanfordnlp.github.io/CoreNLP/history.html) (SCT) and [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) (BNP) from their official website, which are used to obtain the auto-analyzed syntactic knowledge. If you only want to use the knowledge from SCT, you can comment out the script to download BNP in `getdata.sh`. If you want to use the auto-analyzed knowledge from BNP, you need to download both SCT and BNP, because BNP relies on the segmentation results from SCT. 

To run SCT, you need `java 8`; to run BNP, you need `tensorflow==1.1.3`.

You can refer to their websites for more information.

All processed data will appear in `data` directory organized by the datasets, where each of them contains the files with the same file names in the `sample_data` folder.
