FROM onnxruntime-cuda:test

#set up environment
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 ffmpeg libsm6 libxext6
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
# copy all files needed for the project
COPY . /app
# assign workdir
WORKDIR /app
VOLUME [ "/app/data" ]
# install homevision
RUN pip3 install -e .

ENTRYPOINT ["uvicorn", "solution_manager.main:app", "--host", "0.0.0.0", "--port", "5555"]