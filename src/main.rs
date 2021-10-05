use std::error::Error;
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};
use std::thread;
use itertools::Itertools;

use rscam::{Camera, Config};

use rustface::{Detector, FaceInfo, ImageData};
use image::{GrayImage, DynamicImage};

use tflite::{FlatBufferModel, InterpreterBuilder, ops::builtin::BuiltinOpResolver};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    // Setup camera
    let (camera_tx, camera_rx) = sync_channel::<DynamicImage>(0);
    thread::spawn(move || camera_pipeline(camera_tx, 640, 480));
    // Setup face detect
    let (face_detect_tx, face_detect_rx) = sync_channel::<DynamicImage>(0);
    thread::spawn(move || detect_faces_pipeline(camera_rx, face_detect_tx));
    // Setup mask
    let (mask_tx, mask_rx) = sync_channel::<Vec<Vertex>>(0);
    thread::spawn(move || mask_pipeline(face_detect_rx, mask_tx));
    // Setup display
    // Note that this UI code for utilizing WGPU came straight from:
    //   https://sotrh.github.io/learn-wgpu/beginner/tutorial4-buffer/
    let event_loop = EventLoop::<CustomEvent>::with_user_event();
    let event_loop_proxy = event_loop.create_proxy();
    thread::spawn(move || display_pipeline(mask_rx, event_loop_proxy));

    let window = WindowBuilder::new()
        .with_title("MasQRaidr")
        .build(&event_loop)?;
    // State::new uses async code, so we're going to wait for it to finish
    let mut state = pollster::block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &mut so we have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually request it.
                window.request_redraw();
            }
            Event::UserEvent(CustomEvent::Mesh(mesh)) => {
                state.vertex_buffer = Some(state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&mesh[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    });
}

// We will hard code the camera to 640x480
fn camera_pipeline(downstream: SyncSender<DynamicImage>,  width: u32, height: u32) {
    // Setup camera using Video4Linux (V4L2) wrapper
    let mut camera = Camera::new("/dev/video0").unwrap();
    camera.start(&Config {
        interval: (1, 30), // 30fps
        resolution: (width, height),
        format: b"MJPG",
        ..Default::default()
    }).unwrap();
    loop {
        // Get picture
        if let Ok(pic) = camera.capture() {
            if let Ok(pic) = image::load_from_memory(&pic[..]) {
                // Send is blocking; so in effect waits to send downstream when
                // detect_faces_pipeline thread is ready for proccessing.
                if downstream.send(pic).is_err() { break; }
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

fn detect_faces(detector: &mut dyn Detector, gray: &GrayImage) -> Vec<FaceInfo> {
    let (width, height) = gray.dimensions();
    let mut image = ImageData::new(gray, width, height);
    detector.detect(&mut image)
}

fn detect_faces_pipeline(upstream: Receiver<DynamicImage>, downstream: SyncSender<DynamicImage>) {
    let model_buffer = include_bytes!("../assets/models/seeta_fd_frontal_v1.0.bin");
    let model = if let Ok(model) = rustface::read_model(&model_buffer[..]) { model } else { return; };
    let mut detector = rustface::create_detector_with_model(model);
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);
    loop {
        // Grab from camera pipeline the resized image and detect face
        // Blocking on upstream resized image for processing
        let pic = if let Ok(fullpic) = upstream.recv() {
            // Resize from 640x480 to 2.5 smaller on each side
            let pic = fullpic.resize(256, 192, image::imageops::FilterType::Triangle);
            let faces = detect_faces(&mut *detector, &pic.to_luma8());
            // Only grab first face
            if let Some(face) =  faces.first() {
                // println!("face detect confidence level: {:?}", face.score);
                if face.score() < 0.0 { continue; }
                let bbox = face.bbox();
                // TODO: As x and y are i32 types need to verify that x and y cannot be
                // off the image. i.e. as in negative values or larger than image.
                fullpic.crop_imm(
                        (bbox.x() as f32 * 2.5).round() as u32,
                        (bbox.y() as f32 * 2.5).round() as u32,
                        (bbox.width() as f32 * 2.5).round() as u32,
                        (bbox.height() as f32 * 2.5).round() as u32,
                    )
                    .resize(192, 192, image::imageops::FilterType::Triangle)
            } else {
                // Was not able to find any faces so we will just send cropped pic
                pic.crop_imm(32, 0, 192, 192)
            }
        } else {
            break;
        };
        // Send when downstream is ready
        if downstream.send(pic).is_err() { break; }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn mask_pipeline(upstream: Receiver<DynamicImage>, downstream: SyncSender<Vec<Vertex>>) {
    let buffer = include_bytes!("../assets/models/face_landmark.tflite");
    let flat_buffer_model = if let Ok(fbm) = FlatBufferModel::build_from_buffer(buffer.to_vec()) { fbm } else { return; };
    let resolver = BuiltinOpResolver::default();
    let builder = if let Ok(b) = InterpreterBuilder::new(flat_buffer_model, resolver) { b } else { return; };
    // let mut interpreter = builder.build_with_threads(4)?;
    let mut interpreter = if let Ok(i) = builder.build() { i } else { return; };
    // interpreter.print_state();
    if interpreter.allocate_tensors().is_err() { return; };
    loop {
        // Blocking on upstream cropped image of detected area for processing
        if let Ok(pic) = upstream.recv() {
            let pic = pic.to_bgr8();
            // Number of pixels occationally come up short when user is partly off the screen.
            if pic.width() != 192 || pic.height() != 192 {
                println!("Got odd size pic {:?}, {:?}", pic.width(), pic.height());
                continue;
            }
            let pic_iter = pic.pixels();
            // Get inputs and outputs after allocation; so locations do not change
            let inputs = interpreter.inputs().to_vec();
            let outputs =interpreter.outputs().to_vec();

            // INPUT: As input tensor data is a mutation we need to declare and use before we manage output.
            // Inputs: 0
            // Tensor   0 input_1              kTfLiteFloat32  kTfLiteArenaRw     442368 bytes ( 0.4 MB)  1 192 192 3
            let input = if let Ok(i) = interpreter.tensor_data_mut::<f32>(inputs[0]) { i } else { return; };
            for (vector, bgr) in input.chunks_exact_mut(3).zip_eq(pic_iter) {
                vector[0] = bgr[0] as f32 / 255.0;
                vector[1] = bgr[1] as f32 / 255.0;
                vector[2] = bgr[2] as f32 / 255.0;
            }
            if interpreter.invoke().is_err() { return; };
            // OUTPUT: As output tensor needs access to data we need to declare after we are done mutating input.
            // Outputs: 213 210
            // Tensor 213 conv2d_21            kTfLiteFloat32  kTfLiteArenaRw       5616 bytes ( 0.0 MB)  1 1 1 1404
            let output1: &[f32] = if let Ok(o) = interpreter.tensor_data(outputs[0]) { o } else { return; };
            // Tensor 210 conv2d_31            kTfLiteFloat32  kTfLiteArenaRw          4 bytes ( 0.0 MB)  1 1 1 1
            let output2: &[f32] = if let Ok(o) = interpreter.tensor_data(outputs[1]) { o } else { return; };
            // println!("mask confidence level: {:?}", output2[0]);
            if output2[0] < 0.0 { continue; }
            let mut vertices: Vec<Vertex> = Vec::new();
            for vertex in output1.chunks(3) {
                // Use Z distance as how dark point is
                let mut grey = (vertex[2]+25.0).abs()/50.0;
                if grey > 1.0 {
                    grey = 1.0
                }
                // Adjust for window reverse direction change in y and 0 for both x and y
                // being in the middle of the window vs bottom left corner.
                // Also adjust for mirror in x direction.
                // Normalize from 0 -> 192 to -1.0 -> 1.0
                vertices.push(Vertex {
                    position: [
                        // x
                        (vertex[0]/-95.5 + 1.0),
                        // y
                        (vertex[1]/-95.5 + 1.0),
                        // z
                        0.0,
                    ],
                    color: [
                        // red
                        grey,
                        // green
                        grey,
                        // blue
                        grey,
                    ]
                })
            }
            // Send vectors to display_pipeline when that process is ready
            if downstream.send(vertices).is_err() { break; }
        } else {
            break;
        }
    }
}

#[derive(Debug, Clone)]
enum CustomEvent {
    Mesh(Vec<Vertex>),
}

fn display_pipeline(upstream: Receiver<Vec<Vertex>>, event_loop_proxy: winit::event_loop::EventLoopProxy<CustomEvent>) {
    // Wake up the event_loop once every face landmark calculation
    // and dispatch a custom event from a different thread.
    loop {
        if let Ok(mesh) = upstream.recv() {
            if event_loop_proxy.send_event(CustomEvent::Mesh(mesh)).is_err() { break; }
        } else {
            break;
        }
    }
}

// tesselation points came from mediapipe javascript repo
const INDICES: &[u16] = &[
    // 600 bytes = 10*10 triangles * 3 vertex-index/triangle * 2 bytes/vertex-index
    127, 34,139,   11,  0, 37,  232,231,120,   72, 37, 39,  128,121, 47,  232,121,128,  104, 69, 67,  175,171,148,  118, 50,101,   73, 39, 40,
      9,151,108,   48,115,131,  194,204,211,   74, 40,185,   80, 42,183,   40, 92,186,  230,229,118,  202,212,214,   83, 18, 17,   76, 61,146,
    160, 29, 30,   56,157,173,  106,204,194,  135,214,192,  203,165, 98,   21, 71, 68,   51, 45,  4,  144, 24, 23,   77,146, 91,  205, 50,187,
    201,200, 18,   91,106,182,   90, 91,181,   85, 84, 17,  206,203, 36,  148,171,140,   92, 40, 39,  193,189,244,  159,158, 28,  247,246,161,
    236,  3,196,   54, 68,104,  193,168,  8,  117,228, 31,  189,193, 55,   98, 97, 99,  126, 47,100,  166, 79,218,  155,154, 26,  209, 49,131,
    135,136,150,   47,126,217,  223, 52, 53,   45, 51,134,  211,170,140,   67, 69,108,   43,106, 91,  230,119,120,  226,130,247,   63, 53, 52,
    238, 20,242,   46, 70,156,   78, 62, 96,   46, 53, 63,  143, 34,227,  123,117,111,   44,125, 19,  236,134, 51,  216,206,205,  154,153, 22,
     39, 37,167,  200,201,208,   36,142,100,   57,212,202,   20, 60, 99,   28,158,157,   35,226,113,  160,159, 27,  204,202,210,  113,225, 46,
     43,202,204,   62, 76, 77,  137,123,116,   41, 38, 72,  203,129,142,   64, 98,240,   49,102, 64,   41, 73, 74,  212,216,207,   42, 74,184,
    169,170,211,  170,149,176,  105, 66, 69,  122,  6,168,  123,147,187,   96, 77, 90,   65, 55,107,   89, 90,180,  101,100,120,   63,105,104,

    // 1200 bytes
     93,137,227,   15, 86, 85,  129,102, 49,   14, 87, 86,   55,  8,  9,  100, 47,121,  145, 23, 22,   88, 89,179,    6,122,196,   88, 95, 96,
    138,172,136,  215, 58,172,  115, 48,219,   42, 80, 81,  195,  3, 51,   43,146, 61,  171,175,199,   81, 82, 38,   53, 46,225,  144,163,110,
     52, 65, 66,  229,228,117,   34,127,234,  107,108, 69,  109,108,151,   48, 64,235,   62, 78,191,  129,209,126,  111, 35,143,  117,123, 50,
    222, 65, 52,   19,125,141,  221, 55, 65,    3,195,197,   25,  7, 33,  220,237, 44,   70, 71,139,  122,193,245,  247,130, 33,   71, 21,162,
    170,169,150,  188,174,196,  216,186, 92,    2, 97,167,  141,125,241,  164,167, 37,   72, 38, 12,   38, 82, 13,   63, 68, 71,  226, 35,111,
    101, 50,205,  206, 92,165,  209,198,217,  165,167, 97,  220,115,218,  133,112,243,  239,238,241,  214,135,169,  190,173,133,  171,208, 32,
    125, 44,237,   86, 87,178,   85, 86,179,   84, 85,180,   83, 84,181,  201, 83,182,  137, 93,132,   76, 62,183,   61, 76,184,   57, 61,185,
    212, 57,186,  214,207,187,   34,143,156,   79,239,237,  123,137,177,   44,  1,  4,  201,194, 32,   64,102,129,  213,215,138,   59,166,219,
    242, 99, 97,    2, 94,141,   75, 59,235,   24,110,228,   25,130,226,   23, 24,229,   22, 23,230,   26, 22,231,  112, 26,232,  189,190,243,
    221, 56,190,   28, 56,221,   27, 28,222,   29, 27,223,   30, 29,224,  247, 30,225,  238, 79, 20,  166, 59, 75,   60, 75,240,  147,177,215,

    // 1800 bytes
     20, 79,166,  187,147,213,  112,233,244,  233,128,245,  128,114,188,  114,217,174,  131,115,220,  217,198,236,  198,131,134,  177,132, 58,
    143, 35,124,  110,163,  7,  228,110, 25,  356,389,368,   11,302,267,  452,350,349,  302,303,269,  357,343,277,  452,453,357,  333,332,297,
    175,152,377,  347,348,330,  303,304,270,    9,336,337,  278,279,360,  418,262,431,  304,408,409,  310,415,407,  270,409,410,  450,348,347,
    422,430,434,  313,314, 17,  306,307,375,  387,388,260,  286,414,398,  335,406,418,  364,367,416,  423,358,327,  251,284,298,  281,  5,  4,
    373,374,253,  307,320,321,  425,427,411,  421,313, 18,  321,405,406,  320,404,405,  315, 16, 17,  426,425,266,  377,400,369,  322,391,269,
    417,465,464,  386,257,258,  466,260,388,  456,399,419,  284,332,333,  417,285,  8,  346,340,261,  413,441,285,  327,460,328,  355,371,329,
    392,439,438,  382,341,256,  429,420,360,  364,394,379,  277,343,437,  443,444,283,  275,440,363,  431,262,369,  297,338,337,  273,375,321,
    450,451,349,  446,342,467,  293,334,282,  458,461,462,  276,353,383,  308,324,325,  276,300,293,  372,345,447,  352,345,340,  274,  1, 19,
    456,248,281,  436,427,425,  381,256,252,  269,391,393,  200,199,428,  266,330,329,  287,273,422,  250,462,328,  258,286,384,  265,353,342,
    387,259,257,  424,431,430,  342,353,276,  273,335,424,  292,325,307,  366,447,345,  271,303,302,  423,266,371,  294,455,460,  279,278,294,

    // 2400 bytes
    271,272,304,  432,434,427,  272,407,408,  394,430,431,  395,369,400,  334,333,299,  351,417,168,  352,280,411,  325,319,320,  295,296,336,
    319,403,404,  330,348,349,  293,298,333,  323,454,447,   15, 16,315,  358,429,279,   14, 15,316,  285,336,  9,  329,349,350,  374,380,252,
    318,402,403,    6,197,419,  318,319,325,  367,364,365,  435,367,397,  344,438,439,  272,271,311,  195,  5,281,  273,287,291,  396,428,199,
    311,271,268,  283,444,445,  373,254,339,  282,334,296,  449,347,346,  264,447,454,  336,296,299,  338, 10,151,  278,439,455,  292,407,415,
    358,371,355,  340,345,372,  346,347,280,  442,443,282,   19, 94,370,  441,442,295,  248,419,197,  263,255,359,  440,275,274,  300,383,368,
    351,412,465,  263,467,466,  301,368,389,  395,378,379,  412,351,419,  436,426,322,    2,164,393,  370,462,461,  164,  0,267,  302, 11, 12,
    268, 12, 13,  293,300,301,  446,261,340,  330,266,425,  426,423,391,  429,355,437,  391,327,326,  440,457,438,  341,382,362,  459,457,461,
    434,430,394,  414,463,362,  396,369,262,  354,461,457,  316,403,402,  315,404,403,  314,405,404,  313,406,405,  421,418,406,  366,401,361,
    306,408,407,  291,409,408,  287,410,409,  432,436,410,  434,416,411,  264,368,383,  309,438,457,  352,376,401,  274,275,  4,  421,428,262,
    294,327,358,  433,416,367,  289,455,439,  462,370,326,    2,326,370,  305,460,455,  254,449,448,  255,261,446,  253,450,449,  252,451,450,

    // 3000 bytes
    256,452,451,  341,453,452,  413,464,463,  441,413,414,  258,442,441,  257,443,442,  259,444,443,  260,445,444,  467,342,445,  459,458,250,
    289,392,290,  290,328,460,  376,433,435,  250,290,392,  411,416,433,  341,463,464,  453,464,465,  357,465,412,  343,412,399,  360,363,440,
    437,399,456,  420,456,363,  401,435,288,  372,383,353,  339,255,249,  448,261,255,  133,243,190,  133,155,112,   33,246,247,   33,130, 25,
    398,384,286,  362,398,414,  362,463,341,  263,359,467,  263,249,255,  466,467,260,   75, 60,166,  238,239, 79,  162,127,139,   72, 11, 37,
    121,232,120,   73, 72, 39,  114,128, 47,  233,232,128,  103,104, 67,  152,175,148,  119,118,101,   74, 73, 40,  107,  9,108,   49, 48,131,
     32,194,211,  184, 74,185,  191, 80,183,  185, 40,186,  119,230,118,  210,202,214,   84, 83, 17,   77, 76,146,  161,160, 30,  190, 56,173,
    182,106,194,  138,135,192,  129,203, 98,   54, 21, 68,    5, 51,  4,  145,144, 23,   90, 77, 91,  207,205,187,   83,201, 18,  181, 91,182,
    180, 90,181,   16, 85, 17,  205,206, 36,  176,148,140,  165, 92, 39,  245,193,244,   27,159, 28,   30,247,161,  174,236,196,  103, 54,104,
     55,193,  8,  111,117, 31,  221,189, 55,  240, 98, 99,  142,126,100,  219,166,218,  112,155, 26,  198,209,131,  169,135,150,  114, 47,217,
    224,223, 53,  220, 45,134,   32,211,140,  109, 67,108,  146, 43, 91,  231,230,120,  113,226,247,  105, 63, 52,  241,238,242,  124, 46,156,

    // 3600 bytes
     95, 78, 96,   70, 46, 63,  116,143,227,  116,123,111,    1, 44, 19,    3,236, 51,  207,216,205,   26,154, 22,  165, 39,167,  199,200,208,
    101, 36,100,   43, 57,202,  242, 20, 99,   56, 28,157,  124, 35,113,   29,160, 27,  211,204,210,  124,113, 46,  106, 43,204,   96, 62, 77,
    227,137,116,   73, 41, 72,   36,203,142,  235, 64,240,   48, 49, 64,   42, 41, 74,  214,212,207,  183, 42,184,  210,169,211,  140,170,176,
    104,105, 69,  193,122,168,   50,123,187,   89, 96, 90,   66, 65,107,  179, 89,180,  119,101,120,   68, 63,104,  234, 93,227,   16, 15, 85,
    209,129, 49,   15, 14, 86,  107, 55,  9,  120,100,121,  153,145, 22,  178, 88,179,  197,  6,196,   89, 88, 96,  135,138,136,  138,215,172,
    218,115,219,   41, 42, 81,    5,195, 51,   57, 43, 61,  208,171,199,   41, 81, 38,  224, 53,225,   24,144,110,  105, 52, 66,  118,229,117,
    227, 34,234,   66,107, 69,   10,109,151,  219, 48,235,  183, 62,191,  142,129,126,  116,111,143,  118,117, 50,  223,222, 52,   94, 19,141,
    222,221, 65,  196,  3,197,   45,220, 44,  156, 70,139,  188,122,245,  139, 71,162,  149,170,150,  122,188,196,  206,216, 92,  164,  2,167,
    242,141,241,    0,164, 37,   11, 72, 12,   12, 38, 13,   70, 63, 71,   31,226,111,   36,101,205,  203,206,165,  126,209,217,   98,165, 97,
    237,220,218,  237,239,241,  210,214,169,  140,171, 32,  241,125,237,  179, 86,178,  180, 85,179,  181, 84,180,  182, 83,181,  194,201,182,

    // 4200 bytes
    177,137,132,  184, 76,183,  185, 61,184,  186, 57,185,  216,212,186,  192,214,187,  139, 34,156,  218, 79,237,  147,123,177,   45, 44,  4,
    208,201, 32,   98, 64,129,  192,213,138,  235, 59,219,  141,242, 97,   97,  2,141,  240, 75,235,  229, 24,228,   31, 25,226,  230, 23,229,
    231, 22,230,  232, 26,231,  233,112,232,  244,189,243,  189,221,190,  222, 28,221,  223, 27,222,  224, 29,223,  225, 30,224,  113,247,225,
     99, 60,240,  213,147,215,   60, 20,166,  192,187,213,  243,112,244,  244,233,245,  245,128,188,  188,114,174,  134,131,220,  174,217,236,
    236,198,134,  215,177, 58,  156,143,124,   25,110,  7,   31,228, 25,  264,356,368,    0, 11,267,  451,452,349,  267,302,269,  350,357,277,
    350,452,357,  299,333,297,  396,175,377,  280,347,330,  269,303,270,  151,  9,337,  344,278,360,  424,418,431,  270,304,409,  272,310,407,
    322,270,410,  449,450,347,  432,422,434,   18,313, 17,  291,306,375,  259,387,260,  424,335,418,  434,364,416,  391,423,327,  301,251,298,
    275,281,  4,  254,373,253,  375,307,321,  280,425,411,  200,421, 18,  335,321,406,  321,320,405,  314,315, 17,  423,426,266,  396,377,369,
    270,322,269,  413,417,464,  385,386,258,  248,456,419,  298,284,333,  168,417,  8,  448,346,261,  417,413,285,  326,327,328,  277,355,329,
    309,392,438,  381,382,256,  279,429,360,  365,364,379,  355,277,437,  282,443,283,  281,275,363,  395,431,369,  299,297,337,  335,273,321,

    // 4800 bytes
    348,450,349,  359,446,467,  283,293,282,  250,458,462,  300,276,383,  292,308,325,  283,276,293,  264,372,447,  346,352,340,  354,274, 19,
    363,456,281,  426,436,425,  380,381,252,  267,269,393,  421,200,428,  371,266,329,  432,287,422,  290,250,328,  385,258,384,  446,265,342,
    386,387,257,  422,424,430,  445,342,276,  422,273,424,  306,292,307,  352,366,345,  268,271,302,  358,423,371,  327,294,460,  331,279,294,
    303,271,304,  436,432,427,  304,272,408,  395,394,431,  378,395,400,  296,334,299,    6,351,168,  376,352,411,  307,325,320,  285,295,336,
    320,319,404,  329,330,349,  334,293,333,  366,323,447,  316, 15,315,  331,358,279,  317, 14,316,    8,285,  9,  277,329,350,  253,374,252,
    319,318,403,  351,  6,419,  324,318,325,  397,367,365,  288,435,397,  278,344,439,  310,272,311,  248,195,281,  375,273,291,  175,396,199,
    312,311,268,  276,283,445,  390,373,339,  295,282,296,  448,449,346,  356,264,454,  337,336,299,  337,338,151,  294,278,455,  308,292,415,
    429,358,355,  265,340,372,  352,346,280,  295,442,282,  354, 19,370,  285,441,295,  195,248,197,  457,440,274,  301,300,368,  417,351,465,
    251,301,389,  394,395,379,  399,412,419,  410,436,322,  326,  2,393,  354,370,461,  393,164,267,  268,302, 12,  312,268, 13,  298,293,301,
    265,446,340,  280,330,425,  322,426,391,  420,429,437,  393,391,326,  344,440,438,  458,459,461,  364,434,394,  428,396,262,  274,354,457,

    // 5112 bytes = 4800 bytes + (10*5+2) triangles * 3 vertex-index/triangle *2 bytes/vertex-index
    // NOTE: 5112/4 = 1278 evenly; so no padding needed to align on 4 byte boundary
    317,316,402,  316,315,403,  315,314,404,  314,313,405,  313,421,406,  323,366,361,  292,306,407,  306,291,408,  291,287,409,  287,432,410,
    427,434,411,  372,264,383,  459,309,457,  366,352,401,    1,274,  4,  418,421,262,  331,294,358,  435,433,367,  392,289,439,  328,462,326,
     94,  2,370,  289,305,455,  339,254,448,  359,255,446,  254,253,449,  253,252,450,  252,256,451,  256,341,452,  414,413,463,  286,441,414,
    286,258,441,  258,257,442,  257,259,443,  259,260,444,  260,467,445,  309,459,250,  305,289,290,  305,290,460,  401,376,435,  309,250,392,
    376,411,433,  453,341,464,  357,453,465,  343,357,412,  437,343,399,  344,360,440,  420,437,456,  360,420,363,  361,401,288,  265,372,353,
    390,339,249,  339,448,255,
];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/mesh_shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                //cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLAMPING
                clamp_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        // This will be filled in by windows events
        let vertex_buffer = None;
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {}

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_frame()?.output;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            if let Some(vertex_buffer) = &self.vertex_buffer {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
