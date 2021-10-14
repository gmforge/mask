// Mesh Vertex shader

struct MeshVertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] color: vec3<f32>;
};

struct MeshVertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
};

[[stage(vertex)]]
fn main_mesh(
    model: MeshVertexInput,
) -> MeshVertexOutput {
    var out: MeshVertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Mesh Fragment shader

[[stage(fragment)]]
fn main_mesh(in: MeshVertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
