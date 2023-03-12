FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer="fname.lname@domain.com"

# install opencv åreqs
RUN apt-get update \
    && apt-get install libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 libxrender1 wget --no-install-recommends -y

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set username inside docker
ARG uname=user1

# add user uname as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$uname" --shell /bin/bash -g root -G sudo -u 1001 "$uname"
# activate user
USER "$uname"
WORKDIR "/home/$uname"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/home/$uname/miniconda3/bin:${PATH}"
ARG PATH="/home/$uname/miniconda3/bin:${PATH}"

# download and install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && mkdir "/home/$uname/.conda" \
    && bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

# copy env yaml file
COPY environment.yaml "/home/$uname"

# create conda env
RUN conda init bash \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda env create -f environment.yaml

# add conda env activation to bashrc
RUN echo "conda activate control" >> ~/.bashrc

# copy all files from current dir except those in .dockerignore
COPY . "/home/$uname"

CMD ["/bin/bash"]