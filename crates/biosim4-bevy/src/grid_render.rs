//! Grid texture renderer.
//!
//! One `Image` asset whose pixel data we rewrite each frame from the
//! simulation's [`Grid`] / agents / food layer. Displayed via a single
//! `Sprite`, scaled up by [`SimControls::pixel_scale`]. Y is flipped on
//! upload so row 0 is the top of the world — matches the WASM frontend and
//! avoids inverting the camera.
//!
//! Challenge overlays (circles / rectangles / points returned by
//! `ChallengeRegistry::get_overlays`) are drawn as 2D gizmos on top, so they
//! ride above the texture without a separate render pass.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use biosim4_core::grid::{BARRIER, EMPTY};
use biosim4_core::registry::challenge::ChallengeOverlay;
use biosim4_core::types::Coord;

use crate::sim::{Sim, SimControls};

/// Marker for the sprite that displays the grid.
#[derive(Component)]
pub struct GridSprite;

#[derive(Resource)]
pub struct GridTextureHandle {
    pub handle: Handle<Image>,
    pub width: u32,
    pub height: u32,
}

pub struct GridRenderPlugin;

impl Plugin for GridRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_grid_sprite)
            .add_systems(Update, (
                resize_or_update_texture,
                draw_challenge_overlays,
            ));
    }
}

fn spawn_grid_sprite(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    sim: Res<Sim>,
    controls: Res<SimControls>,
) {
    let (w, h) = (sim.state.config.size_x as u32, sim.state.config.size_y as u32);
    let mut img = make_grid_image(w, h);
    encode_into(&sim, image_bytes_mut(&mut img));
    let handle = images.add(img);

    let px = controls.pixel_scale;
    commands.spawn((
        GridSprite,
        Sprite {
            image: handle.clone(),
            color: Color::WHITE,
            custom_size: Some(Vec2::new(w as f32 * px, h as f32 * px)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    commands.insert_resource(GridTextureHandle { handle, width: w, height: h });
}

fn make_grid_image(w: u32, h: u32) -> Image {
    let mut img = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        vec![0u8; (w * h * 4) as usize],
        // sRGB so Bevy's sprite shader passes the bytes through to the
        // framebuffer un-gamma-shifted. With plain `Rgba8Unorm` the linear
        // pipeline interprets the data as already-gamma-corrected and the
        // output ends up nearly black.
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    // Nearest-neighbor sampler keeps cells pixel-crisp as you zoom in.
    img.sampler = ImageSampler::nearest();
    img
}

fn image_bytes_mut(img: &mut Image) -> &mut [u8] {
    img.data.as_mut().expect("image was created with Some(data)").as_mut_slice()
}

fn resize_or_update_texture(
    sim: Res<Sim>,
    mut controls: ResMut<SimControls>,
    mut images: ResMut<Assets<Image>>,
    mut handle: ResMut<GridTextureHandle>,
    mut sprite_q: Query<&mut Sprite, With<GridSprite>>,
) {
    let (w, h) = (sim.state.config.size_x as u32, sim.state.config.size_y as u32);

    // If grid dimensions changed (e.g. after Recreate with a different size),
    // discard and rebuild the texture asset.
    if w != handle.width || h != handle.height {
        let mut img = make_grid_image(w, h);
        encode_into(&sim, image_bytes_mut(&mut img));
        let new_handle = images.add(img);
        handle.handle = new_handle.clone();
        handle.width = w;
        handle.height = h;
        if let Ok(mut sprite) = sprite_q.single_mut() {
            sprite.image = new_handle;
        }
        controls.grid_dirty = false;
    } else if controls.grid_dirty {
        if let Some(img) = images.get_mut(&handle.handle) {
            encode_into(&sim, image_bytes_mut(img));
        }
        controls.grid_dirty = false;
    }

    // Keep the sprite's world size synced with pixel_scale.
    let target = Vec2::new(w as f32 * controls.pixel_scale, h as f32 * controls.pixel_scale);
    if let Ok(mut sprite) = sprite_q.single_mut() {
        if sprite.custom_size != Some(target) {
            sprite.custom_size = Some(target);
        }
    }
}

/// Encode the simulation into an RGBA byte buffer. Same encoding as
/// `biosim4-wasm`'s `render_frame_into` but inlined here to avoid pulling a
/// `wasm-bindgen`-bearing crate into a native binary.
fn encode_into(sim: &Sim, buf: &mut [u8]) {
    let sx = sim.state.config.size_x as usize;
    let sy = sim.state.config.size_y as usize;
    let needed = sx * sy * 4;
    if buf.len() != needed { return; }

    for y in 0..sy {
        // Flip Y so the visual row 0 is the world top.
        let world_y = sy - 1 - y;
        let row_base = y * sx * 4;
        for x in 0..sx {
            let coord = Coord::new(x as i16, world_y as i16);
            let cell = sim.state.grid.at(coord);
            let rgb = match cell {
                EMPTY => {
                    let food = sim.state.food.get(coord);
                    if food > 0.01 {
                        [0, (food * 120.0) as u8, 0]
                    } else {
                        [0, 0, 0]
                    }
                }
                BARRIER => [80, 80, 80],
                id => sim.state.population.get(id).map(|a| a.color).unwrap_or([0, 0, 0]),
            };
            let off = row_base + x * 4;
            buf[off]     = rgb[0];
            buf[off + 1] = rgb[1];
            buf[off + 2] = rgb[2];
            buf[off + 3] = 255;
        }
    }
}

/// Project a world-space point onto a grid cell index. The grid sprite is
/// centered at (0, 0) in world space with size `grid_w*pixel × grid_h*pixel`.
/// Returns `None` if the point is outside the grid bounds.
pub fn sprite_to_cell(
    p: Vec2,
    grid_w: u32, grid_h: u32, pixel: f32,
) -> Option<(u16, u16)> {
    let half_w = grid_w as f32 * pixel * 0.5;
    let half_h = grid_h as f32 * pixel * 0.5;
    let lx = (p.x + half_w) / pixel;
    let ly = (p.y + half_h) / pixel;
    if lx < 0.0 || ly < 0.0 { return None; }
    let xi = lx.floor() as i32;
    let yi = ly.floor() as i32;
    if xi < 0 || yi < 0 || xi >= grid_w as i32 || yi >= grid_h as i32 { return None; }
    Some((xi as u16, yi as u16))
}

/// Draw challenge overlay shapes (the visual hint for where survivors are
/// supposed to end up). Gizmo lines anti-alias for free, so the circles read
/// nicely at any zoom.
fn draw_challenge_overlays(
    sim: Res<Sim>,
    controls: Res<SimControls>,
    mut gizmos: Gizmos,
) {
    let world = sim.state.world();
    let overlays = sim.state.challenges.get_overlays(&world);
    if overlays.is_empty() { return; }
    let sx = sim.state.config.size_x as f32;
    let sy = sim.state.config.size_y as f32;
    let px = controls.pixel_scale;
    let half_w = sx * px * 0.5;
    let half_h = sy * px * 0.5;

    // ChallengeOverlay coords are in BIOSIM cell units (0..sx, 0..sy), not
    // normalized 0..1 — they get baked by each challenge's `overlays()` impl
    // before we ever see them. Convert one biosim unit to one world pixel by
    // multiplying by `px` and re-centering.
    let to_world = |bx: f32, by: f32| -> Vec2 {
        Vec2::new(bx * px - half_w, by * px - half_h)
    };
    let to_color = |c: [u8; 4]| -> Color {
        // Alpha from challenge overlays is small (often 40/255) — bump it
        // up so the outline reads on a dark canvas.
        let a = (c[3] as f32 / 255.0).clamp(0.0, 1.0).max(0.4);
        Color::srgba(
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0,
            a,
        )
    };

    // sx/sy are used to suppress a "could be unused" warning when no overlay
    // path references them — they're computed above for `half_w` / `half_h`.
    let _ = (sx, sy);

    for o in overlays {
        match o {
            ChallengeOverlay::Circle { cx, cy, radius, color } => {
                let center = to_world(cx, cy);
                let r = radius * px;
                let col = to_color(color);
                gizmos.circle_2d(center, r, col);
                gizmos.circle_2d(center, (r - 2.0).max(1.0), col.with_alpha(0.4 * col.alpha()));
            }
            ChallengeOverlay::Rectangle { x, y, w, h, color } => {
                let tl = to_world(x, y);
                let br = to_world(x + w, y + h);
                let center = (tl + br) * 0.5;
                let size = (br - tl).abs();
                gizmos.rect_2d(Isometry2d::from_translation(center), size, to_color(color));
            }
            ChallengeOverlay::Points { points, color, size } => {
                let col = to_color(color);
                let r = (size * px * 0.5).max(2.0);
                for (bx, by) in points {
                    gizmos.circle_2d(to_world(bx, by), r, col);
                }
            }
        }
    }
}
