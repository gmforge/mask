mod render;
mod texture;
use std::error::Error;
use std::sync::mpsc::{sync_channel, SyncSender, Receiver};
use std::thread;
use itertools::Itertools;

use rscam::{Camera, Config};

use rustface::{Detector, FaceInfo, ImageData};
use image::{GrayImage, DynamicImage, Bgr};

use tflite::{FlatBufferModel, InterpreterBuilder, ops::builtin::BuiltinOpResolver};

use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use self::render::{ColorVertex, State};

#[derive(Debug, Clone)]
enum CustomEvent {
    ImageMesh((DynamicImage, Vec<ColorVertex>)),
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    // Setup camera
    let (camera_tx, camera_rx) = sync_channel::<DynamicImage>(0);
    thread::spawn(move || camera_pipeline(camera_tx, 640, 480));
    // Setup face detect
    let (face_detect_tx, face_detect_rx) = sync_channel::<DynamicImage>(0);
    thread::spawn(move || detect_faces_pipeline(camera_rx, face_detect_tx));
    // Setup display
    // Note that this UI code for utilizing WGPU came straight from:
    //   https://sotrh.github.io/learn-wgpu/beginner/tutorial4-buffer/
    let event_loop = EventLoop::<CustomEvent>::with_user_event();
    let event_loop_proxy = event_loop.create_proxy();
    // Setup mask to send events to windo event loop
    thread::spawn(move || mask_pipeline(face_detect_rx, event_loop_proxy));

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
            Event::UserEvent(CustomEvent::ImageMesh((image, mesh))) => {
                state.color_vertex_buffer = Some(state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Color Vertex Buffer"),
                    contents: bytemuck::cast_slice(&mesh[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                }));
                // TODO: setup texture
                // pub texture_bind_group_layout: wgpu::BindGroupLayout,
                // pub diffuse_texture: Option<texture::Texture>,
                // pub diffuse_bind_group: Option<wgpu::BindGroup>,
                // Image is rgb8 and need rgba8
                let image = image.to_rgba8();
                let diffuse_texture =
                    texture::Texture::from_image(&state.device, &state.queue, &DynamicImage::ImageRgba8(image), Some("face detected")).unwrap();
                state.diffuse_bind_group = Some(state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &state.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                        },
                    ],
                    label: Some("diffuse_bind_group"),
                }));
                state.diffuse_texture = Some(diffuse_texture);

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
    detector.set_window_size(48);
    detector.set_min_face_size(48);
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
            let mut max_score = 0.0;
            let mut face = None;
            for f in &faces {
                if f.score() > max_score {
                    max_score = f.score();
                    face = Some(f);
                }
            }
            if let Some(face) =  face {
                let bbox = face.bbox();
                // Since detection crops out bottom of the mouth and captures much of the forehead
                // will add extra 20% of height.
                //let dh = bbox.height() / 5;
                //y = if y + (dh as i32) < 192 { y + (dh as i32) } else { 192 };
                // Note detection bbox is jittery, as its size increases and decreases;
                // so need to act like steady cam or find some way to keep the size
                // consistent. For the moment we will always send 2/3 of the image height
                // or a 128x128 image centered on the detected face to the next stage.
                // This will also fix the mouth being cropped out of the detected image,
                // but at the expense of limiting how close the face may be from the camera.
                let cx = bbox.x() as u32 + (bbox.width() / 2);
                let cy = bbox.y() as u32 + (bbox.height() / 2);
                let x = if cx < 65 { 0 } else if cx > (256-64) { 256-128 } else { cx-64 };
                let y = if cy < 64 { 0 } else if cy > (192-64) { 192-128 } else { cy-64 };
                fullpic.crop_imm(
                        x * 25 / 10,
                        y * 25 / 10,
                        128 * 25 / 10,
                        128 * 25 / 10,
                    )
                    .resize_exact(192, 192, image::imageops::FilterType::Triangle)
            } else {
                // Was not able to find any faces so we will just send cropped pic
                //pic.crop_imm(32, 0, 192, 192)
                println!("Unable to detect face with sufficient score");
                continue;
            }
        } else {
            break;
        };
        // Send when downstream is ready
        if downstream.send(pic).is_err() { break; }
    }
}

fn mask_pipeline(upstream: Receiver<DynamicImage>, event_loop_proxy: winit::event_loop::EventLoopProxy<CustomEvent>) {
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
        if let Ok(image) = upstream.recv() {
            let pic = image.to_bgr8();
            // Number of pixels occationally come up short when user is partly off the screen,
            // or when float rounding is off by 1 pixel.
            // This should not be a problem any more since we are now using resize_exact
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
            for (vector, &Bgr([b, g, r])) in input.chunks_exact_mut(3).zip_eq(pic_iter) {
                vector[0] = b as f32 / 255.0; // x
                vector[1] = g as f32 / 255.0; // y
                vector[2] = r as f32 / 255.0; // z
            }
            if interpreter.invoke().is_err() { return; };
            // OUTPUT: As output tensor needs access to data we need to declare after we are done mutating input.
            // Outputs: 213 210
            // Tensor 213 conv2d_21            kTfLiteFloat32  kTfLiteArenaRw       5616 bytes ( 0.0 MB)  1 1 1 1404
            let output1: &[f32] = if let Ok(o) = interpreter.tensor_data(outputs[0]) { o } else { return; };
            // Tensor 210 conv2d_31            kTfLiteFloat32  kTfLiteArenaRw          4 bytes ( 0.0 MB)  1 1 1 1
            //let output2: &[f32] = if let Ok(o) = interpreter.tensor_data(outputs[1]) { o } else { return; };
            //println!("mask confidence level: {:?}", output2[0]);
            //if output2[0] < 0.0 { continue; }
            let mut vertices: Vec<ColorVertex> = Vec::new();
            for vertex in output1.chunks(3) {
                // Use Z distance as how dark point is
                let grey = 1.0 - ((vertex[2]+25.0).abs()/50.0).min(1.0);
                // Adjust for window reverse direction change in y and 0 for both x and y
                // being in the middle of the window vs bottom left corner.
                // Also adjust for mirror in x direction.
                // Normalize from 0 -> 191 to -1.0 -> 1.0
                vertices.push(ColorVertex {
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
                    ],
                })
            }
            // Wake up the event_loop once every face landmark calculation
            // and dispatch a custom event on a different thread.
            if event_loop_proxy.send_event(CustomEvent::ImageMesh((image, vertices))).is_err() { break; }
        } else {
            break;
        }
    }
}
