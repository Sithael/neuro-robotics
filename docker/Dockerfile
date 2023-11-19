FROM ros:noetic

ENV RLROOT=/RLROOT
ENV ABOVE_DIR=$RLROOT/..
ENV PYTHONPATH=$PYTHONPATH:$DIR:$ABOVE_DIR
WORKDIR $RLROOT

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
	python3.8-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt
