# A Defensive Approach for Query-Based Audio Adversarial Attacks
We introduce a state detection defense for query-based attacks to detect the process of generating adversarial examples(AEs) in ASR systems. We analyze the group behavior during the generation of the adversarial samples and detect the attacks based on the association before and after the query. During the query process, the samples inevitably leave digital audio fingerprint information, and we extract the fingerprints of the samples and do similarity matching with the previous queries. Cut off the user's right to continue accessing the model when the likelihood of an attack occurring becomes high. We experimentally validated our scheme for an upwards of 80% detection success rate for existing query-based attacks, and some of them achieve 98%. 
# Pre-requisites
NVIDIA GPU + CUDA cuDNN + Cupy
# Data-sets
Music-sets:https://drive.google.com/file/d/1wPVK9S8TyB0aaXqXFKEebYKuKshmBvDc/view?usp=sharing.    
Mini-LibriSpeech :www.openslr.org/resources/31/train-clean-5.tar.gz
# Obtaining source code
CS(CommanderSong): We contacted the author of CommanderSong and got the source code, and we do not release the source code of CommanderSong according to the author's request. So you can try to get in touch with the author and get the source code.  
DW(Devil's Whisper): You can get the DW's source code from https://github.com/RiskySignal/Devil-Whisper-Attack  
IRTA(["Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition"](https://arxiv.org/abs/1903.10346)):You can get the IRTA's source code from https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/adversarial_asr    
DS(["Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"](https://arxiv.org/abs/1801.01944)):You can get the DS's source code from https://github.com/carlini/audio_adversarial_examples

