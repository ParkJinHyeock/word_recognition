# word_recognition

## Project for word recognition & human identification using biosensor.
### You can train file using main.py 
#### There are four option.

1. you can choose task using --mode (human/word)
2. you can choose dataset using --path
3. you can choose split mode using --split_mode 
(There are 4 mode : random, human, word, human_word, cross)
cross option make test dataset composed of unseen data depending on your task. 
4. you can choose model using --model (1D/2D) 
(1D model use raw waveform and 2D model use spectrogram to classify)
