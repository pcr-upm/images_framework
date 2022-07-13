# This is our first build stage, it will not persist in the final image
FROM ubuntu as intermediate
RUN apt-get -y update && apt-get install -y git
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa
RUN chmod 400 /root/.ssh/id_rsa
# Make sure your domain is accepted
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Download the computer vision framework
RUN git clone git@github.com:bobetocalo/images_framework.git images_framework

# Copy the repository from the previous image
FROM ubuntu
ENV LANG=C.UTF-8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y update && apt-get install -y build-essential wget libsm6 libxext6 libxrender-dev libglib2.0-0
RUN mkdir /home/username
WORKDIR /home/username
COPY --from=intermediate /images_framework /home/username/images_framework
LABEL maintainer="roberto.valle@upm.es"
# Setup conda environment
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/username/miniconda.sh
RUN chmod +x /home/username/miniconda.sh
RUN /home/username/miniconda.sh -b -p /home/username/conda
RUN /home/username/conda/bin/conda create --name framework python=3.6
# Activate conda environment
ENV PATH /home/username/conda/envs/framework/bin:/home/username/conda/bin:$PATH
# Make RUN commands use the new environment (source activate framework)
SHELL ["conda", "run", "-n", "framework", "/bin/bash", "-c"]
# Install dependencies
RUN pip install opencv-python==4.2.0.32 opencv-contrib-python==4.2.0.32 rasterio
