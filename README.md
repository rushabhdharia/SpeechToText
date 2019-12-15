# SpeechToText
Project on Automatic Speech Recogition for course Deep Learning Speech Processing

## Dataset Used
1. LibriSpeech100 (100 hours) - https://www.openslr.org/12

## Steps to execute code
1. Download LibriSpeech100 Dataset from the link mentioned above.
2. Run CreatePickleFiles.ipynb - Creates Pickle Files of the Input Data
3. Create train_all, dev_all and test_all folders in train, dev and test folders respectively which lie inside the LibriSpeech100 folder. We will gather all our input and output files in these directories. 
3. Run GatherPickleAndTextFiles.ipynb - Gathers all the input pickle and output text files and places them in their appropriate folders
4. Run speech2text.py on terminal or Speech2Text.ipynb in a jupyter environment

## References
1. [Hori, Takaaki, Jaejin Cho, and Shinji Watanabe. "End-to-end speech recognition with word-based RNN language models." 2018 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2018.](https://arxiv.org/pdf/1808.02608.pdf)
2. https://web.stanford.edu/class/cs224s/lectures/224s.17.lec8.pdf
3. https://arxiv.org/pdf/1803.10225.pdf

## Technologies Used
1. Miniconda
2. Tensorflow 2
[Setup Tensorflow 2 on Conda](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)

## Collaborators
[Rushabh Dharia](https://github.com/rushabhdharia)  
[Chaitanya Patil](https://github.com/chaitz333)   
