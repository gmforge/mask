# mask
Capture facial expressions on a mesh structure that forms a face.

## How to Run
```bash
cargo build --release
./target/release/mask
```

## Purpose
This is a Rust project to map human expressions from a live video on to a mesh structure forming a face. At the moment the applications is broken into about 4 sections that captures video frames, detect faces in the frame and crops one of the faces, take the cropped image and detects face landmarks from the image, and then render the mesh on a video card. Using this is as an model on how to use tensorflow tflite and other rust AI tools, as well as a test example of using channels between threads to form a pipeline.

### V4L2 Videocamera/webcam frame capture (thread 1)
In this thread we use the Rust **rscam** library which interfaces with the Video for Linux interface to capture frames. It then passes the frames over a channel to the next stage of the pipeline.

### Detect faces and crop out Region of Interest (ROI) (thread 2)
In this thread we use the Rust **rustface** AI library along with its `seeta_fd_frontal_v1.0.bin` model to detect any faces within a reduced version of the image. If we find a face with a reasonable score we will crop the face out of the original image and will pass the region of interest over a channel to the next stage of the pipeline.

### Map Face Landmarks from the ROI (thread 3)
This thread uses the Rust Tensor Flow Lite **tflite** AI library along with the mediapipe `face_landmark.tflite` trained model to detect the face landmarks. If the detection score is high enough we turn those 486 landmarks that form a mesh structure into vertices associated with each point. We then pass these vertices over a channel to the next stage of the pipeline.

### Render the Face Landmarks on the Video Card (thread 4&5)
This thread uses the Rust **wgpu** and **winit** libraries to render the face mesh structure using shaders provided by the github.com/sotrh/learn-wgpu/ tutorial and along with tessellation indices found in the mediapipe javascript face_landmark code.

## Issues and barriers.
- At the moment this project only uses rscam; so will only run on Linux. Will need to use other libraries to access video camera / webcams on other platforms to expand testing on a variety of systems.
- The face_landmark.tflite model seems to not be able to detect winks, large open mouths and eyes, or puckered lips. Tested these expressions with the python example provided by mediapipe face landmark project and noticed the same issues. That is limiting when attempting to capturing facial expressions, and will probably need to explore further solutions.
- There are many pauses where the AI does not make highly confident detection scores and thus the way the code is setup it will just not render frames until it gets better scores. Not sure how to improve the number of confident scores at this time.
