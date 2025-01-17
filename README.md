# TOF-MRA-Synth
## Purpose
To synthesize high-quality 3D-TOF-MRA images from 3D-T1-weighted, 3D-T2-weighted, and 3D-FLAIR images using a deep learning model.

## Methods
We developed a conditional generative adversarial network (cGAN) model based on the Unet++ architecture, which demonstrates strong capabilities in integrating global information and fusing cross-modality features. Additionally, we designed a loss function (MIP loss) sensitive to maximum intensity values to enhance the control of fine details in the synthesis of TOF-MRA images.
![all](https://github.com/user-attachments/assets/f20d6bf7-6003-4a6f-b057-ba6077c528ad)

## Data preprocessing
> All the data was registered first by SimpleITK
We use **hdf5** as the data storage method
- data: 3D data array (shape: z, x, y)
- max: max intensity
- max_len: num of slice (Axial)

## Usage
### Environment
`pip install -r requirements.txt`

### Data storage
/hdf5_root/case_names/modality.dhf5  
/dcm_root/case_name/...  
> 'modality' contains '3D-T1W', '3D-T2W', '3D-FLAIR', '3D-TOF-MRA'

### Train
`python train.py`  
options.data_root: hdf5_root

### Inference
`python test_all.py`  
options.data_root: hdf5_root  
options.dcm_root: dicom_root

## Acknowledge
Thanks to Shanghai Sixth People's Hospital.



