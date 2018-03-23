FROM jgc128/uml_nlp_class

RUN pip3 install nltk
RUN python3 -m nltk.downloader punkt
RUN git clone --recursive https://github.com/text-machine-lab/uml_nlp_class.git
RUN cd pytorch_0.4
RUN pip install typing
RUN pip install cmake
RUN python setup.py install
RUN pip install tweet-preprocessor
RUN pip install matplotlib
RUN pip install beautifulsoup4
RUN apt-get install python-tk


ADD . /usr/src/app


CMD ["python3", "train.py"]


