FROM debian:9.3

ENV PATH="${PATH}:/usr/local/anaconda/bin"

# setup
RUN set -x \
    && apt-get update \
    && apt-get install -y bzip2 curl gcc git less man unzip vim

# conda
RUN set -x \
    && cd /tmp \
    && curl -fSLO https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh \
    && sha256sum Anaconda3-5.1.0-Linux-x86_64.sh | grep -q 7e6785caad25e33930bc03fac4994a434a21bc8401817b7efa28f53619fa9c29 \
    && bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p /usr/local/anaconda \
    && conda update conda

# extras
RUN set -x \
    && conda install pytorch-cpu spacy torchvision -c pytorch -y \
    && pip install torchtext \
    && python -m spacy.en.download \
    && pip uninstall -y pillow \
    && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# cleanup
RUN set -x \
    && apt-get remove -y gcc \
    && apt-get autoremove \
    && rm -rf /tmp/*

COPY root /

ENTRYPOINT [ "/bin/bash" ]
CMD [ "-l" ]
