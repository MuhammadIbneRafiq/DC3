# Group 3
## JBG060 Capstone Data Challenge  2025-2026 Q1
This repository contains the code for the course JBG060 Capstone Data Challenge .
Please read this document carefully as it has been filled out with important information.

## Code structure

## Environment setup instructions
### Environment setup instructions for non-GPU-equipped devices
### Environment setup instructions for GPU-equipped devices

## Argparse
Argparse functionality is included 

[//]: # ( template!!! was used for dc1!!!!!!! in the main.py file. This means the file can be run from the command line while passing arguments to the main function. Right now, there are arguments included for the number of epochs &#40;nb_epochs&#41;, batch size &#40;batch_size&#41;, whether to create balanced batches &#40;balanced_batches&#41;, whether to perform the eda or not &#40;eda&#41;, and whether to show the interpretability using Grad-CAM &#40;int&#41;.)

[//]: # ()
[//]: # (To make use of this functionality, first open the command prompt and change to the directory containing the main.py file.)

[//]: # (For example, if you're main file is in C:\Data-Challenge-1-template-main\dc1\, )

[//]: # (type `cd C:\Data-Challenge-1-template-main\dc1\` into the command prompt and press enter.)

[//]: # ()
[//]: # (Then, main.py can be run by, for example, typing `python main.py --nb_epochs 10 --batch_size 25 --eda true --int true`.)

[//]: # (This would run the script with 10 epochs, a batch size of 25, balanced batches, perform the eda pipeline, and show Grad-CAM, which is also the current default.)

[//]: # (If you would want to run the script with 20 epochs, a batch size of 5, batches that are not balanced, are not interested in the eda pipeline, and not interested in the Grad-CAM figures, you would type `main.py --nb_epochs 20 --batch_size 5 --no-balanced_batches --eda false --int false`.)

[//]: # (If you would want to run on a different model then the default transfer learning model either use --model custom or --model template )

## Authors
* Juliette Hattingh-Haasbroek (1779192)
* Doah Lee (2034395)
* Roan van Merwijk (1856022)
* Elvir Nikq (1931075)
* Muhammad Rafiq (1924214)
* Melissa Selamet (1921495)