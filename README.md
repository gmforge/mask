# mask
Capture facial expressions on a mesh structure that forms a face.

## How to Run
```bash
cargo build --release
./target/release/mask
```

## Purpose
This is a Rust project to map human expressions from a live video on to a mesh structure forming a face. At the moment the applications is broken into about 4 sections that captures video frames, detect faces in the frame and crops one of the faces, take the cropped image and detects face landmarks from the image, and then have the video card render the mesh and ROI onto a window. This project was mostly started as an example on how to use tensorflow tflite and other rust AI tools, as well as how to use channels between threads to form a pipeline.

### V4L2 Videocamera/webcam frame capture (thread 1)
In this thread we use the Rust **rscam** library which interfaces with the Video for Linux interface to capture frames. It then passes the frames over a channel to the next stage of the pipeline.

### Detect faces and crop out Region of Interest (ROI) (thread 2)
In this thread we use the Rust **rustface** AI library along with its `seeta_fd_frontal_v1.0.bin` model to detect any faces within a reduced image. If we find a face with a reasonable score we will crop the face out of the original image and will pass the region of interest (ROI) over a channel to the next stage of the pipeline. Currently a set size for the ROI is cropped and centered on the detected bound box. The reason for this is that the rustface bound box size varies enough to cause a jittery image, however, the center of the bound box is more stable. This limits the range the face may be away from the camera, but gives a more smooth mesh drawing that is non constantly resizing. Originally we attempted to use the mediapipe `face_detection_short_range.tflite` model. We have yet to get this **tflite** model to produce meaningful results, and is why at this time we are using another face detection library.

### Map Face Landmarks from the ROI (thread 3)
This thread uses the Rust Tensor Flow Lite **tflite** AI library along with the mediapipe `face_landmark.tflite` trained model to detect the face landmarks. If the detection score is high enough we turn those 486 landmarks that form a mesh structure into vertices associated with each point. We then pass these vertices and the ROI image over a channel to the next stage of the pipeline.

### Render the Face Landmarks on the Video Card (thread 4)
This thread uses the Rust **wgpu** and **winit** libraries to render the face mesh structure and cropped image of detected face using shaders provided by the github.com/sotrh/learn-wgpu/ tutorial and along with tessellation indices found in the mediapipe javascript face_landmark code. We moved most of the wgpu code and texture handling library to its own module. The ROI image is drawn as a texture the upper left corner of the window and the mesh of the facial landmarks is drawn at in relation and size to the window as it appears in the ROI image.

## Issues and barriers.
- At the moment this project only uses rscam; so will only run on Linux. Will need to use other libraries to access video camera / webcams on other platforms to expand testing on a variety of systems. Currently looking at imagesnap Rust library to add webcam access on MacOS.
- The face_landmark.tflite model seems to not be able to detect winks, large open mouths and eyes, or puckered lips. Tested these expressions with the python example provided by mediapipe face landmark project and noticed the same issues. That is limiting when attempting to capturing facial expressions, and will probably need to explore further solutions.
- There are many pauses where the AI does not make highly confidence detection scores and thus the way the code is setup it will just not render frames until it gets better scores. Not sure how to improve the number of confidence scores at this time. For instance when a face is looking mostly to the side the rustface detection confidence score drops low enough to not return a face detection result.
