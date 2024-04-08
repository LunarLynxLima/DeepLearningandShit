The 2021061A2.py file has implementation for classification :-

Two datasets :
A) Image Dataset :: [torchaudio.datasets.SPEECHCOMMANDS]
CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.

B) Audio Dataset :: [torchvision.datasets.CIFAR10]
Speech Commands Dataset, and used it to add training* and inference sample code to TensorFlow. The dataset has 65,000 one-second long utterances of 30 short words, by thousands of different people.

And implemented achitecture fro the above datasets :
1) ResNet 
2) VGG
3) Inception
4) Custom Architecture



To install the environment follow following steps.

1. Install Minconda/Anaconda in your system
2. Open "Minconda/Anaconda prompt shell" on windows and "terminal" on linux systems.
3. Run "conda create --name <your env name> --file <path to dla2.txt>"
4. Modify only Pipeline/changerollno.py
5. Test experiments by running "python3 main.py T/F" (T for using GPU and F for using CPU)
6. Write code for saving checkpoints in "trainer" function.
