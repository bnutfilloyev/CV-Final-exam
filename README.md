CV-Final-exam
============
[F22] Introduction to Computer Vision Final Exam at Innopolis university


## Liveness Detection
    part 1: Face Detection
    part 2: Eye Detection
    part 3: Occulusion Detection

### Face Detection part
I used the MTCNN face detector

### Eye Detection part
I used CNN model for eye detection

### Occulusion Detection part
I used YOLOv5 model for occulusion detection
    
# Convert Keras model to ONNX
```shell
python -m tf2onnx.convert --keras <input model>.h5  --output  <output model>.onnx
```

# Running app in the docker container
```shell
    docker-compose up --build
```

# Running python script
```shell
    $ conda create -n cv python=3.9 -y && conda activate cv
    $ pip install -r requirements.txt
    $ uvicorn main:app --reload
```