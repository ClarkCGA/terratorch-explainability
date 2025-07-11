FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/app-root/src
RUN chown -R 10000:0 /opt/app-root
RUN chmod -R g+rwX /opt/app-root

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN  apt update && apt install -y tzdata 
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git build-essential 
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /opt/app-root/src

# Don't copy - use volume mount instead
# COPY --chown=10000:0 . /opt/app-root/src/terratorch

# Create venv 
USER 10000
RUN python3 -m venv /opt/app-root/src/venv
RUN chgrp 0 -R /opt/app-root/src/venv/
RUN chmod 775 -R /opt/app-root/src/venv

# Install dependencies that don't change often
RUN . /opt/app-root/src/venv/bin/activate && pip install --no-cache-dir --upgrade pip
# RUN pip install /opt/app-root/src/terratorch

USER root
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install terratorch in editable mode when container starts
CMD ["bash", "-c", "cd /opt/app-root/src && . venv/bin/activate && pip install -e ./terratorch && bash"]