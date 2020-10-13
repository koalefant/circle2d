use miniquad::*;

type IndexType = u16;
type Vec2 = glam::Vec2;
fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}
use circle2d::{reali, realf, Real, Real2};

// Please ignore the Geometry-mess, it is a quick copy-paste for drawing
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            uv: [0.0, 0.0],
            color: [0, 0, 0, 0],
        }
    }
}

pub trait VertexSize: std::clone::Clone + std::default::Default {
    fn size_of() -> usize;
}
pub trait VertexColor {
    fn set_color(&mut self, color: [u8; 4]);
    fn set_alpha(&mut self, alpha: u8);
    fn alpha(&self) -> u8;
}
pub trait VertexPos2 {
    fn set_pos(&mut self, pos: [f32; 2]);
}
pub trait VertexUV {
    fn set_uv(&mut self, uv: [f32; 2]);
}
impl VertexSize for Vertex {
    fn size_of() -> usize {
        std::mem::size_of::<Self>()
    }
}
impl VertexPos2 for Vertex {
    fn set_pos(&mut self, pos: [f32; 2]) {
        self.pos = pos;
    }
}
impl VertexUV for Vertex {
    fn set_uv(&mut self, uv: [f32; 2]) {
        self.uv = uv;
    }
}
impl VertexColor for Vertex {
    fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }
    fn set_alpha(&mut self, alpha: u8) {
        self.color[3] = alpha;
    }
    fn alpha(&self) -> u8 {
        self.color[3]
    }
}

pub struct Geometry<Vertex: VertexSize> {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<IndexType>,
}

impl<Vertex: VertexSize> Geometry<Vertex> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn allocate(
        &mut self,
        num_vertices: usize,
        num_indices: usize,
        def: Vertex,
    ) -> (&mut [Vertex], &mut [IndexType], IndexType) {
        let old_vertices = self.vertices.len();
        self.vertices.resize(old_vertices + num_vertices, def);
        let old_indices = self.indices.len();
        self.indices
            .resize(old_indices + num_indices, 0 as IndexType);
        (
            &mut self.vertices[old_vertices..],
            &mut self.indices[old_indices..],
            old_vertices as IndexType,
        )
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.vertices.clear();
    }
}

impl<Vertex: VertexSize + VertexPos2 + VertexColor> Geometry<Vertex> {
    #[inline]
    pub fn add_circle_aa(&mut self, pos: Vec2, radius: f32, num_segments: usize, def: Vertex) {
        let (vs, is, first) = self.allocate(2 * num_segments + 1, num_segments * 9, def);
        for (i, pair) in vs.chunks_mut(2).enumerate() {
            let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
            let cos = angle.cos();
            let sin = angle.sin();
            for (v, p) in pair.iter_mut().zip(&[0.0, 1.0]) {
                v.set_pos([
                    pos.x() + cos * (radius - 0.5 + p),
                    pos.y() + sin * (radius - 0.5 + p),
                ]);
                v.set_alpha(((1.0 - p) * 255.0) as u8);
            }
        }
        vs.last_mut().unwrap().set_pos([pos[0], pos[1]]);
        let central_vertex = num_segments * 2;

        for (section_i, section) in is.chunks_mut(9).enumerate() {
            let section_n = (section_i + 1) % num_segments;
            let indices = [
                central_vertex,
                section_i * 2 + 0,
                section_n * 2 + 0,
                section_i * 2 + 0,
                section_i * 2 + 1,
                section_n * 2 + 0,
                section_i * 2 + 1,
                section_n * 2 + 1,
                section_n * 2 + 0,
            ];
            for (dest, src) in section.iter_mut().zip(&indices) {
                *dest = first + *src as IndexType;
            }
        }
    }

    #[inline]
    pub fn add_circle_outline_aa(
        &mut self,
        pos: Vec2,
        radius: f32,
        thickness: f32,
        num_segments: usize,
        def: Vertex,
    ) {
        if thickness > 1.0 {
            let (vs, is, first) = self.allocate(4 * num_segments, num_segments * 18, def);
            let ht = (thickness - 1.0) * 0.5;
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in
                    pair.iter_mut()
                        .zip(&[(-ht - 1.0, 0), (-ht, 255), (ht, 255), (ht + 1.0, 0)])
                {
                    v.set_pos([pos.x() + cos * (radius + p.0), pos.y() + sin * (radius + p.0)]);
                    v.set_alpha(p.1);
                }
            }

            for (section_i, section) in is.chunks_mut(18).enumerate() {
                let section_n = (section_i + 1) % num_segments;
                let indices = [
                    section_i * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_i * 4 + 3,
                    section_n * 4 + 2,
                    section_i * 4 + 3,
                    section_n * 4 + 3,
                    section_n * 4 + 2,
                ];
                for (dest, src) in section.iter_mut().zip(&indices) {
                    *dest = first + *src as IndexType;
                }
            }
        } else {
            let (vs, is, first) = self.allocate(4 * num_segments, num_segments * 12, def);
            for (i, pair) in vs.chunks_mut(4).enumerate() {
                let angle = i as f32 / num_segments as f32 * 2.0 * std::f32::consts::PI;
                let cos = angle.cos();
                let sin = angle.sin();
                for (v, p) in
                    pair.iter_mut()
                        .zip(&[(-1.0, 0), (0.0, (255.0 * thickness) as u8), (1.0, 0)])
                {
                    v.set_pos([pos[0] + cos * (radius + p.0), pos[1] + sin * (radius + p.0)]);
                    v.set_alpha(p.1);
                }
            }
            for (section_i, section) in is.chunks_mut(12).enumerate() {
                let section_n = (section_i + 1) % num_segments;
                let indices = [
                    section_i * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_n * 4 + 1,
                    section_n * 4 + 0,
                    section_i * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 1,
                    section_i * 4 + 2,
                    section_n * 4 + 2,
                    section_n * 4 + 1,
                ];
                for (dest, src) in section.iter_mut().zip(&indices) {
                    *dest = first + *src as IndexType;
                }
            }
        }
    }

    // Assumes coordinates to be pixels
    // based on AddPoyline from Dear ImGui by Omar Cornut (MIT)
    //     // Assumes coordinates to be pixels
    pub fn add_polyline_aa(&mut self, points: &[Vec2], color: [u8; 4], closed: bool, thickness: f32) {
        if points.len() < 2 {
            return;
        }
        let count = match closed {
            true => points.len(),
            false => points.len() - 1,
        };
        let gradient_size = 1.0;
        let thick_line = thickness > gradient_size;

        let color_transparent = [color[0], color[1], color[2], 0];
        let index_count = if thick_line { count * 18 } else { count * 12 };
        let vertex_count = if thick_line { points.len() * 4 } else { points.len() * 3 };
        let (vs, is, first) = self.allocate(vertex_count, index_count, Vertex::default());
        let mut temp_normals = Vec::new();
        let mut temp_points = Vec::new();
        temp_normals.resize(points.len(), vec2(0., 0.));
        temp_points.resize(points.len() * if thick_line { 4 } else { 2 }, vec2(0., 0.));
        for i1 in 0..count {
            let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
            let mut delta = points[i2] - points[i1];
            let len2 = delta.dot(delta);
            if len2 > 0.0 {
                let len = len2.sqrt();
                delta /= len;
            }
            temp_normals[i1] = vec2(delta.y(), -delta.x());
        }
        if !closed {
            temp_normals[points.len() - 1] = temp_normals[points.len() - 2];
        }
        if !thick_line {
            if !closed {
                temp_points[0] = points[0] + temp_normals[0] * gradient_size;
                temp_points[1] = points[1] - temp_normals[1] * gradient_size;

                temp_points[(points.len() - 1) * 2 + 0] =
                    points[points.len() - 1] + temp_normals[points.len() - 1] * gradient_size;
                temp_points[(points.len() - 1) * 2 + 1] =
                    points[points.len() - 1] - temp_normals[points.len() - 1] * gradient_size;
            }

            let mut idx1 = first;
            for i1 in 0..count {
                let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
                let idx2 = if (i1 + 1) == points.len() { first } else { idx1 + 3 };

                let mut dm = (temp_normals[i1] + temp_normals[i2]) * 0.5;
                // average normals
                let mut dm_len2 = dm.dot(dm);
                if dm_len2 < 0.5 {
                    dm_len2 = 0.5;
                }
                let inv_len2 = gradient_size / dm_len2;
                dm *= inv_len2;

                // compute points
                temp_points[i2 * 2 + 0] = points[i2] + dm;
                temp_points[i2 * 2 + 1] = points[i2] - dm;

                // indices
                is[i1 * 12..(i1 + 1) * 12].copy_from_slice(&[
                    idx2 + 0,
                    idx1 + 0,
                    idx1 + 2,
                    idx1 + 2,
                    idx2 + 2,
                    idx2 + 0,
                    idx2 + 1,
                    idx1 + 1,
                    idx1 + 0,
                    idx1 + 0,
                    idx2 + 0,
                    idx2 + 1,
                ]);

                idx1 = idx2;
            }

            for i in 0..points.len() {
                vs[i * 3 + 0].set_pos(points[i].into());
                vs[i * 3 + 0].set_color(color);
                vs[i * 3 + 1].set_pos(temp_points[i * 2 + 0].into());
                vs[i * 3 + 1].set_color(color_transparent);
                vs[i * 3 + 2].set_pos(temp_points[i * 2 + 1].into());
                vs[i * 3 + 2].set_color(color_transparent);
            }
        } else {
            let half_inner_thickness = (thickness - gradient_size) * 0.5;
            if !closed {
                temp_points[0] = points[0] + temp_normals[0] * (half_inner_thickness + gradient_size);
                temp_points[1] = points[0] + temp_normals[0] * half_inner_thickness;
                temp_points[2] = points[0] - temp_normals[0] * half_inner_thickness;
                temp_points[3] = points[0] - temp_normals[0] * (half_inner_thickness + gradient_size);

                temp_points[(points.len() - 1) * 4 + 0] =
                    points[points.len() - 1] + temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);
                temp_points[(points.len() - 1) * 4 + 1] =
                    points[points.len() - 1] + temp_normals[points.len() - 1] * (half_inner_thickness);
                temp_points[(points.len() - 1) * 4 + 2] =
                    points[points.len() - 1] - temp_normals[points.len() - 1] * (half_inner_thickness);
                temp_points[(points.len() - 1) * 4 + 3] =
                    points[points.len() - 1] - temp_normals[points.len() - 1] * (half_inner_thickness + gradient_size);
            }

            let mut idx1 = first;
            for i1 in 0..count {
                let i2 = if (i1 + 1) == points.len() { 0 } else { i1 + 1 };
                let idx2 = if (i1 + 1) == points.len() { first } else { idx1 + 4 };

                let mut dm = temp_normals[i1] + temp_normals[i2] * 0.5;

                // direction of first edge
                let v0 = vec2(-temp_normals[i1].y(), temp_normals[i1].x());

                // project direction of first edge on second edge normal
                if closed || i2 != count {
                    let dot = v0.dot(temp_normals[i2]);
                    // Negative direction of 2nd edge
                    let v1 = vec2(temp_normals[i2].y(), -temp_normals[i2].x());
                    // Scale
                    dm = (v0 + v1) / dot;
                } else {
                    let mut dm_len2 = dm.dot(dm);
                    if dm_len2 < 0.5 {
                        dm_len2 = 0.5;
                    }
                    let inv_len2 = 1.0 / dm_len2;
                    dm *= inv_len2;
                }

                let dm_out = dm * (half_inner_thickness + gradient_size);
                let dm_in = dm * half_inner_thickness;

                // points
                temp_points[i2 * 4 + 0] = points[i2] + dm_out;
                temp_points[i2 * 4 + 1] = points[i2] + dm_in;
                temp_points[i2 * 4 + 2] = points[i2] - dm_in;
                temp_points[i2 * 4 + 3] = points[i2] - dm_out;

                // indices
                is[18 * i1..18 * (i1 + 1)].copy_from_slice(&[
                    idx2 + 1,
                    idx1 + 1,
                    idx1 + 2,
                    idx1 + 2,
                    idx2 + 2,
                    idx2 + 1,
                    idx2 + 1,
                    idx1 + 1,
                    idx1 + 0,
                    idx1 + 0,
                    idx2 + 0,
                    idx2 + 1,
                    idx2 + 2,
                    idx1 + 2,
                    idx1 + 3,
                    idx1 + 3,
                    idx2 + 3,
                    idx2 + 2,
                ]);
                idx1 = idx2;
            }

            for i in 0..points.len() {
                vs[i * 4 + 0].set_pos(temp_points[i * 4 + 0].into());
                vs[i * 4 + 0].set_color(color_transparent);

                vs[i * 4 + 1].set_pos(temp_points[i * 4 + 1].into());
                vs[i * 4 + 1].set_color(color);

                vs[i * 4 + 2].set_pos(temp_points[i * 4 + 2].into());
                vs[i * 4 + 2].set_color(color);

                vs[i * 4 + 3].set_pos(temp_points[i * 4 + 3].into());
                vs[i * 4 + 3].set_color(color_transparent);
            }
        }
    }

}

impl<Vertex: VertexSize + VertexPos2 + VertexUV> Geometry<Vertex> {
    pub fn add_rect_uv(&mut self, rect: [f32; 4], uv: [f32; 4], def: Vertex) -> &mut [Vertex] {
        let (vs, is, first) = self.allocate(4, 6, def);

        for ((dest, pos), uv) in vs
            .iter_mut()
            .zip(
                [
                    [rect[0], rect[1]],
                    [rect[2], rect[1]],
                    [rect[2], rect[3]],
                    [rect[0], rect[3]],
                ]
                .iter(),
            )
            .zip(
                [
                    [uv[0], uv[1]],
                    [uv[2], uv[1]],
                    [uv[2], uv[3]],
                    [uv[0], uv[3]],
                ]
                .iter(),
            )
        {
            dest.set_pos([pos[0], pos[1]]);
            dest.set_uv([uv[0], uv[1]]);
        }
        for (dest, index) in is.iter_mut().zip(&[0, 1, 2, 0, 2, 3]) {
            *dest = index + first;
        }
        vs
    }

    pub fn add_rect_outline(&mut self, rect: [f32; 4], thickness: f32, def: Vertex) {
        let (vs, is, first) = self.allocate(8, 24, def);
        let ht = thickness * 0.5;
        let positions = [
            [rect[0] - ht, rect[1] - ht],
            [rect[2] + ht, rect[1] - ht],
            [rect[2] + ht, rect[3] + ht],
            [rect[0] - ht, rect[3] + ht],
            [rect[0] + ht, rect[1] + ht],
            [rect[2] - ht, rect[1] + ht],
            [rect[2] - ht, rect[3] - ht],
            [rect[0] + ht, rect[3] - ht],
        ];
        for (dest, src) in vs.iter_mut().zip(positions.iter()) {
            dest.set_pos([src[0], src[1]]);
        }
        let indices = [
            0, 1, 4, 4, 1, 5, 1, 2, 5, 5, 2, 6, 2, 3, 6, 6, 3, 7, 3, 0, 7, 7, 0, 4,
        ];
        for (dest, src) in is.iter_mut().zip(indices.iter()) {
            *dest = src + first;
        }
    }
}

struct Stage {
    world: circle2d::World,
    grabbed_body: Option<circle2d::BodyKey>,
    mouse_pos: Vec2,

    pipeline: Pipeline,
    bindings: Bindings,
    white_image: miniquad::Texture,
    background_image: miniquad::Texture,
    last_frame: f64,
    uniforms: shader::Uniforms,
    index_buffer: Option<miniquad::Buffer>,
    vertex_buffer: Option<miniquad::Buffer>,
    geometry: Geometry<Vertex>,
}

impl Stage {
    pub fn new(ctx: &mut Context) -> Stage {
        // generate background image based on SDF
        let mut pixels = Vec::new();
        let w = ctx.screen_size().0 as usize;
        let h = ctx.screen_size().1 as usize;
        pixels.resize_with(w * h * 4, || 0);
        for i in 0..w * h {
            let x = i % w;
            let y = i / w;
            let d: f32 = sd_terrain(vec2(x as f32 + 0.5, y as f32 + 0.5).into()).into();
            let fill = 1.0 - d.max(0.0).min(1.0);
            let val = (fill * 255.0) as u8;
            pixels[i * 4 + 0] = val;
            pixels[i * 4 + 1] = val;
            pixels[i * 4 + 2] = val;
            pixels[i * 4 + 3] = 255;
        }
        let background_image = miniquad::Texture::from_rgba8(ctx, w as _, h as u16, &pixels);
        let white_image = miniquad::Texture::from_rgba8(ctx, 1, 1, &[0xff, 0xff, 0xff, 0xff]);

        #[rustfmt::skip]
        let vertices: [Vertex; 4] = [
            Vertex { pos : [ -1.0, -1.0 ], uv: [ 0., 0. ], color: [ 255, 255, 255, 255 ] },
            Vertex { pos : [  1.0, -1.0 ], uv: [ 1., 0. ], color: [ 255, 255, 255, 255 ] },
            Vertex { pos : [  1.0,  1.0 ], uv: [ 1., 1. ], color: [ 255, 255, 255, 255 ] },
            Vertex { pos : [ -1.0,  1.0 ], uv: [ 0., 1. ], color: [ 255, 255, 255, 255 ] },
        ];
        let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);

        let bindings = Bindings {
            vertex_buffers: vec![vertex_buffer],
            index_buffer,
            images: vec![],
        };

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::META);
        let pipeline = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Byte4),
            ],
            shader,
            miniquad::PipelineParams {
                color_blend: Some((
                    miniquad::Equation::Add,
                    miniquad::BlendFactor::Value(miniquad::BlendValue::SourceAlpha),
                    miniquad::BlendFactor::OneMinusValue(miniquad::BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );

        let uniforms = shader::Uniforms {
            screen_size: [ctx.screen_size().0, ctx.screen_size().1],
        };

        let time = miniquad::date::now();

        let mut geometry = Geometry::new();
        geometry.vertices.reserve(1024 * 1024);
        geometry.indices.reserve(1024 * 1024);

        let mut world = circle2d::World::new();

        // add terrain body
        let terrain = world.add_body(circle2d::BodyDef {
            typ: circle2d::BodyType::Static,
            shape: circle2d::Shape::Map,
            shape_flags: 2,
            mass: None,
            inertia: None,
            restitution: reali(0),
            friction: reali(1),
            ..Default::default()
        });

        // add some circles
        let mut num_created = 0;
        while num_created < 80 {
            let x = unsafe { miniquad::rand() as f32 / miniquad::RAND_MAX as f32 * w as f32 };
            let y = unsafe { miniquad::rand() as f32 / miniquad::RAND_MAX as f32 * h as f32 };
            let radius =
                8.0 + 16.0 * unsafe { miniquad::rand() as f32 / miniquad::RAND_MAX as f32 };
            let d: Real = (sd_terrain(vec2(x, y).into()) - realf(radius)).min(world.sample_distance(
                vec2(x, y).into(),
                terrain,
                0xffffffff,
                reali(32),
            ).into());
            if d < reali(0) {
                // hitting something, try again
                continue;
            }

            world.add_body(circle2d::BodyDef {
                shape: circle2d::Shape::Circle(radius.into()),
                shape_flags: 1 | 2,
                pos: vec2(x, y).into(),
                mass: Some(reali(1)),
                inertia: Some(reali(1)),
                friction: realf(0.5),
                restitution: realf(0.6),
                ..Default::default()
            });
            num_created += 1;
        }

        Stage {
            world,
            pipeline,
            bindings,
            uniforms,
            last_frame: time,
            index_buffer: None,
            vertex_buffer: None,
            geometry,
            background_image,
            white_image,
            grabbed_body: None,
            mouse_pos: vec2(0., 0.),
        }
    }
}

pub fn sd_box(p: Real2, b: Real2) -> Real {
    let q = p.abs() - b;
    q.max(Real2::zero()).length() + q.x().max(q.y()).min(reali(0))
}

pub fn sd_circle(p: Real2, center: Real2, r: Real) -> Real {
    (p - center).length() - r
}

pub fn sd_line(p: Real2, a: Real2, b: Real2, r: Real) -> Real {
    let pa = p - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).max(reali(0)).min(reali(1));
    (pa - ba * h).length() - r
}

fn sd_terrain(p: Real2) -> Real {
    let mut d = -sd_box(p - vec2(1280. * 0.5, 720. * 0.5 - 100.0).into(), vec2(590., 210.).into());
    d = d.max(-sd_box(p - vec2(400., 470.).into(), vec2(200., 200.).into()));
    d = d.max(-sd_circle(p, vec2(940., 500.).into(), reali(150)));

    d = d.min(sd_circle(p, vec2(1040., 350.).into(), reali(60)));
    d = d.min(sd_circle(p, vec2(170., 450.).into(), reali(100)));
    d = d.min(sd_line(p, vec2(200., 200.).into(), vec2(600.0, 350.0).into(), reali(10)));
    d = d.min(sd_line(p, vec2(800., 350.).into(), vec2(1000.0, 150.0).into(), reali(10)));
    d
}

impl EventHandler for Stage {
    fn update(&mut self, _ctx: &mut Context) {
        let time = miniquad::date::now();
        let time_step = 0.016;
        // do not simulate more than 3 frames
        let mut remaining_dt = (time - self.last_frame).min(time_step * 3.0);
        let distance_func = |p: Real2| sd_terrain(p);
        let normal_func = |p: Real2| -> Real2 {
            let samples = [
                distance_func(p),
                distance_func(p + vec2(0.1, 0.0).into()),
                distance_func(p + vec2(0.0, 0.1).into()),
            ];
            Real2::new(samples[1] - samples[0], samples[2] - samples[0]).normalize()
        };
        let dt = time_step as f32;
        while remaining_dt > time_step {
            let mut started_contacts = Vec::new();
            let mut finished_contacts = Vec::new();

            // apply some gravity on all bodies
            let active_bodies: Vec<_> = self.world.active_bodies.iter().copied().collect();
            for b in active_bodies.into_iter() {
                self.world
                    .body_set_force_by_acceleration(b, vec2(0., 800.0).into());
            }
            if let Some(b) = self.grabbed_body {
                let pos = self.world.body_position(b).into();
                let force = (self.mouse_pos - pos) * 100.0;
                self.world.body_apply_force(b, force.into());
            }

            self.world.simulate_physics(
                distance_func,
                normal_func,
                dt.into(),
                &mut started_contacts,
                &mut finished_contacts,
            );

            // here one can update velocities after simulation if needed

            // update body positions/orientation
            self.world.update_positions(dt.into(), &distance_func);

            remaining_dt -= time_step;
        }
        self.last_frame = time - remaining_dt;
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32) {
        self.mouse_pos = vec2(x, y);
    }

    fn mouse_button_down_event(&mut self, _c: &mut Context, button: MouseButton, x: f32, y: f32) {
        let mut bodies = Vec::new();
        match button {
            MouseButton::Left => {
                self.world.find_exact(
                    &mut bodies,
                    vec2(x, y).into(),
                    circle2d::Shape::Circle(reali(1)),
                    1,
                    &sd_terrain,
                );
                self.grabbed_body = bodies.first().copied();
            }
            MouseButton::Right => {
                let r = reali(100);
                self.world.find_exact(
                    &mut bodies,
                    vec2(x, y).into(),
                    circle2d::Shape::Circle(r),
                    1,
                    &sd_terrain,
                );
                for b in bodies.into_iter() {
                    let delta = self.world.body_position(b) - vec2(x, y).into();
                    self.world.body_set_velocity(
                        b,
                        delta.normalize() * (reali(1) - delta.length() / r) * reali(3000),
                    );
                }
            }
            _ => {}
        }
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        _button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.grabbed_body = None;
    }

    fn draw(&mut self, c: &mut Context) {
        // drawing
        self.geometry.clear();

        let def = Vertex {
            pos: [0., 0.],
            uv: [0., 0.],
            color: [255, 255, 255, 255],
        };
        self.geometry.add_rect_uv(
            [
                0.,
                0.,
                self.background_image.width as _,
                self.background_image.height as _,
            ],
            [0.0, 0.0, 1.0, 1.0],
            def,
        );

        for k in self.world.bodies() {
            let shape = self.world.body_shape(k);
            let pos = self.world.body_position(k).into();
            let vel: Vec2 = self.world.body_velocity(k).into();
            let rotation: f32 = self.world.body_rotation(k).into();
            let shape_flags = self.world.body_shape_flags(k);
            let contact_count: u32 = self
                .world
                .persistent_contacts
                .keys()
                .map(|t| {
                    let mut result = 0;
                    if t.0 == k {
                        result += 1;
                    }
                    if t.1 == k {
                        result += 1;
                    }
                    result
                })
                .sum();

            let color = match (
                self.world.body_type(k) == circle2d::BodyType::Static
                    || self.world.body_is_sleeping(k),
                contact_count,
            ) {
                (true, _) => [160, 160, 255, 255],
                (false, 0) => [255, 255, 0, 255],
                (false, _) => [255, 100, 100, 255],
            };

            match shape {
                circle2d::Shape::Circle(radius) => {
                    let radius: f32 = radius.into();
                    if (shape_flags & self.world.collision_shape_flags) == 0 {
                        self.geometry.add_circle_outline_aa(
                            pos,
                            radius - 1.0,
                            2.0,
                            32,
                            Vertex { color, ..def },
                        );
                    } else {
                        self.geometry
                            .add_circle_aa(pos.into(), radius, 32, Vertex { color, ..def });
                    }
                    let dir = Vec2::new(rotation.cos(), rotation.sin());
                    self.geometry.add_polyline_aa(
                        &[pos, pos + dir * radius],
                        [color[0] / 2, color[1] / 2, color[2] / 2, 255],
                        false,
                        2.0,
                    );
                }
                _ => {}
            }
            // velocity vector
            self.geometry.add_polyline_aa(
                &[pos, pos + vel / 4.0],
                [color[0] / 2, color[1] / 2, color[2] / 2, 255],
                false,
                4.0,
            );
            self.geometry
                .add_polyline_aa(&[pos, pos + vel / 4.0], color, false, 2.0);
        }

        for (_k, p) in &self.world.persistent_contacts {
            let color = [50, 50, 0, 255];
            for point in p.points.iter() {
                self.geometry.add_polyline_aa(
                    &[point.position.into(), (point.position + point.normal * reali(5)).into()],
                    color,
                    false,
                    1.0,
                );
                self.geometry.add_circle_aa(
                    point.position.into(),
                    1.5,
                    8,
                    Vertex {
                        color: [25, 25, 0, 255],
                        ..def
                    },
                );
            }
        }

        // update buffers
        if let Some(b) = self.index_buffer {
            if b.size() < self.geometry.indices.len() * std::mem::size_of::<IndexType>() {
                b.delete();
                self.index_buffer = None;
            }
        }
        if self.index_buffer.is_none() {
            self.index_buffer = Some(miniquad::Buffer::stream(
                c,
                miniquad::BufferType::IndexBuffer,
                self.geometry.indices.capacity() * std::mem::size_of::<IndexType>(),
            ));
        }
        if let Some(b) = self.vertex_buffer {
            if b.size() < self.geometry.vertices.len() * Vertex::size_of() {
                b.delete();
                self.vertex_buffer = None;
            }
        }
        if self.vertex_buffer.is_none() {
            self.vertex_buffer = Some(miniquad::Buffer::stream(
                c,
                miniquad::BufferType::VertexBuffer,
                self.geometry.vertices.capacity() * std::mem::size_of::<Vertex>(),
            ));
        }
        self.index_buffer
            .unwrap()
            .update(c, self.geometry.indices.as_slice());
        self.vertex_buffer
            .unwrap()
            .update(c, self.geometry.vertices.as_slice());
        self.bindings = Bindings {
            vertex_buffers: vec![self.vertex_buffer.unwrap()],
            index_buffer: self.index_buffer.unwrap(),
            images: vec![self.background_image],
        };

        self.uniforms.screen_size = [c.screen_size().0, c.screen_size().1];

        c.begin_default_pass(Default::default());
        c.apply_pipeline(&self.pipeline);
        c.apply_bindings(&self.bindings);
        c.apply_uniforms(&self.uniforms);
        c.draw(0, 6, 1);
        self.bindings.images[0] = self.white_image;

        c.apply_bindings(&self.bindings);
        c.draw(6, self.geometry.indices.len() as _, 1);
        c.end_render_pass();

        c.commit_frame();
    }
}

fn main() {
    miniquad::start(
        conf::Conf {
            window_title: "Circle2D Demo".into(),
            window_width: 1280,
            window_height: 720,
            ..conf::Conf::default()
        },
        |mut ctx| UserData::owning(Stage::new(&mut ctx), ctx),
    );
}

mod shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
    attribute vec2 pos;
    attribute vec2 uv;
    attribute vec4 color;

    uniform vec2 screen_size;

    varying highp vec4 v_color;
    varying highp vec2 v_uv;

    void main() {
        gl_Position = vec4(pos * vec2(2.0, -2.0) / screen_size.xy + vec2(-1.0, 1.0), 0, 1);
        v_color = color / 255.0;
        v_uv = uv;
    }"#;

    pub const FRAGMENT: &str = r#"#version 100
    precision highp float;

    varying vec4 v_color;
    varying vec2 v_uv;
    uniform sampler2D tex;
        
    void main() {
        gl_FragColor = v_color * texture2D(tex, v_uv);
    }"#;

    pub const META: ShaderMeta = ShaderMeta {
        images: &["tex"],
        uniforms: UniformBlockLayout {
            uniforms: &[UniformDesc::new("screen_size", UniformType::Float2)],
        },
    };

    #[repr(C)]
    pub struct Uniforms {
        pub screen_size: [f32; 2],
    }
}
