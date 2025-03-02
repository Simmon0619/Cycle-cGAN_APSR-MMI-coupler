# MMI Coupler Design with Cycle-Consistent cGAN

## **Description**
This repository provides an open-source implementation of a **cycle-consistent conditional generative adversarial network (cycle cGAN)** for the inverse design of **1×2 multimode interference (MMI) couplers**. MMI couplers are widely used as optical power splitters in **photonic integrated circuits (PICs)** due to their compact footprint, broadband operation, and fabrication tolerance. However, achieving **arbitrary power-splitting ratios with low excess losses (ELs)** remains a significant challenge, often requiring extensive manual tuning and high computational resources.

To address these limitations, this project integrates a **cycle cGAN** as an inverse model and a **fully connected neural network (FCNN)** as a forward model to ensure cycle-consistency verification. The framework efficiently generates MMI designs at a **wavelength of 1550 nm**, covering power-splitting ratios from **1:99 to 99:1** while maintaining **excess losses below 0.7 dB**. This approach significantly reduces computational costs and ensures physically realizable designs, making it a promising tool for advancing **silicon photonics and photonic integration**.

## **Project Structure**
```
MMI_CGAN_Design/
│── data/               # Folder for training/testing datasets
│   ├── 1x2_taper_MMI_v1.h5
│   ├── 1x2_taper_MMI_v2.h5
│   ├── 1x2_taper_MMI_v3.h5
│
│── models/             # Folder for saved trained models
│   ├── forward_model_v3_cycle.pth
│   ├── inverse_model_v3_cycle.pth
│   ├── discriminator_v3_cycle.pth
│
│── Cycle_cgan_MMI_training.py   # Training script
│── Cycle_cgan_MMI_testing.py    # Testing script
│── LICENSE            # License information
│── README.md          # Project description
│── requirements.txt   # Dependencies
```

## **Installation**
To set up the environment and install dependencies, run:
```bash
pip install -r requirements.txt
```

## **Usage**
### **Training the Model**
To train the cycle cGAN model, run:
```bash
python Cycle_cgan_MMI_training.py
```

### **Testing the Model**
To generate designs using the trained model, run:
```bash
python Cycle_cgan_MMI_testing.py
```

## **Requirements**
This project requires the following dependencies:
```
torch
numpy
pandas
h5py
scikit-learn
FrEIA
openpyxl
```

## **Results & Outputs**
- The trained models are stored in the `models/` folder.
- Generated structure parameters are saved as text files.
- The test results, including optimized MMI designs, are stored as an Excel file.

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgment**
This work leverages deep learning techniques for inverse photonic design and contributes to the advancement of silicon photonics and photonic integrated circuits (PICs).
