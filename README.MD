**Group 10: Phu Gia Hoang, Bekassyl Syzdykov** <br />

**Project title: "A Duet of XLM-RoBERTa and DPCNN for Sentiment Analysis"** <br />

1. **Setup the environment:** <br />
    source setup/conda-create.sh  <br />
    source setup/conda-active.sh  <br />

2. **Install all the dependencies:**  <br />
    pip install -r requirements.txt  <br />

**NOTE: You have change the path to the dataset, according to your 


**Dataset** <br />
Inside a **/data/** directory: <br />
    1. You can find raw, augmented and pre-preprocessed SST-5 dataset. <br />
    2. Moreover, we provide dataset statistics inside **/statistics/** folder. <br />
    3. In addition, classification reports, confusion matrices and training history are provided inside **/history/** folder. <br />


**Data Pre-processing** <br />
Inside a **src/data_prep** directory: <br />
    1. Inside an **/augmentation/** directory, you will find an implementation for the EDA augmentation technique. <br /> 
    2. Inside an **/pre-processing/** directory, you will find an implementation for the data pre-processing and statistics. <br />


**Training XLMR and DPCNN** <br />
Inside a **src/training** directory: <br />
    1. You will find a code for XLMR fine-tuning and XLMR-DPCNN combination

Model weights can be found here: https://mbzuaiac-my.sharepoint.com/:f:/g/personal/phu_hoang_mbzuai_ac_ae/EnUp-qlTpPRBoQ46ISjMqYsBT0mgsC4eOZT0Yw09f5_fJA?e=0pa0fn

 


