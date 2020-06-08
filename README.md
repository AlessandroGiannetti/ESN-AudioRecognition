# ESN-AudioRecognition
Neural Network Project

Audio classification has been a growing field lately since better and cheaper sensors are becoming widely available. The amount of applications for this technology has already proven to be staggering, from events monitoring to speech recognition the amount of practical uses keeps growing [1].
In an effort to continue exploring the use of machine learning techniques for environmental sound classification, our contribution focuses on the implementation of a Recurrent Neural Network (RNN) [2] to classify active machinery in construction sites.

In detail, this paper will describe our first approach at implementing an Echo State Network using a plethora of audio data collected on-site to establish if this technology can be effective in the recognition of different kinds machinery sounds.

Echo State Networks (ESN) are very simple to implement and are readily provided by many libraries on the network.
The aim of the project is to test whether ESNs can be useful tools for the audio data classification problem. 
For this purpose we developed an application for the classification of different types of construction vehicles and tools, through an Echo State Network. The proposed approach consists in splitting the audio data into fragments and sampling them into spectrogram representation to then be classified into the different types of machinery. This approach exhibited great potential in environmental sound classification (ESC) achieving significant accuracy. 



* M. Lukosevicius and H. Jaeger, “Overview of reservoir recipes”, School of  Engineering  and  Science,  Jacobs  University,  Technical  Report  No.11, 2007.
*  Danilo P. Mandic, Jonathon A. Chambers, “Recurrent Neural Networks for Prediction: Learning Algorithms, Architectures and Stability", August 2001
* Mantas Lukosevicius, “A Practical Guide to Applying Echo State Networks”, Jacobs University Bremen
* “Large-Scale Weakly Supervised Audio Classification Using Gated Convolutional Neural Network",  Yong Xu,  Qiuqiang Kong, Wenwu Wang, Mark D. Plumbley  2018 IEEE International Conference on Acoustics
* Brian McFee, Colin Raffel, Dawen Liang, Daniel P.W. Ellis, Matt McVicar, Eric Battenberg, Oriol Nieto “librosa: Audio and Music Signal Analysis in Python”, Proc. of the 14th Python in Science Conf. (SCIPY 2015)
* Alessandro Maccagno, Andrea Mastropietro, Umberto Mazziotta, Michele Scarpiniti, Yong-Cheol Lee, and Aurelio Uncini “A CNN Approach for Audio Classification in Construction Sites”, Sapienza University of Rome,
https://github.com/AndMastro/WreckingNet
* EasyESN, https://github.com/kalekiu/easyesn
* E. Alpaydin, Introduction to Machine Learning, Mit Press, 3rd Ed., 2014.
* D Ruta, B Gabrys, Information fusion, Elsevier 2005.
