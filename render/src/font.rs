//! A TrueType font rasterized once into an R8 coverage atlas (§19's UI font).
//! GPU-free on purpose: `Font::rasterize` and all metric queries run under
//! plain `cargo test`, and `UiPass` only consumes the produced atlas + quads.
//!
//! The bundled face is DejaVu Sans Mono (render/fonts/, Bitstream Vera
//! license — see render/fonts/LICENSE); its monospace advances make the
//! menu/console layout maths predictable. Glyphs are rasterized for the whole
//! printable ASCII range at `raster_em` pixels and drawn scaled, so text can
//! be any screen size without re-rasterizing.

use ab_glyph::{Font as _, ScaleFont as _};

/// Printable ASCII range we cache: 95 glyphs.
pub const FIRST: char = ' ';
pub const LAST: char = '~';
const COUNT: usize = (LAST as usize) - (FIRST as usize) + 1;
/// Grid layout. Cell 0 is a solid block for `UiPass::rect`, glyphs follow.
const COLS: usize = 16;
const ROWS: usize = (COUNT + 1).div_ceil(COLS);
/// Blank border around every glyph cell; keeps LINEAR sampling off neighbours.
const PAD: usize = 2;

/// One cached glyph. All metrics are in *raster pixels* at `Font::raster_em`;
/// multiply by `size / raster_em` to get screen pixels.
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    /// Pen advance; equal for every glyph (monospace).
    pub advance: f32,
    /// Left edge of the bitmap relative to the pen (negative = overhang).
    pub min_x: f32,
    /// Top edge of the bitmap relative to the baseline (negative = above it).
    pub min_y: f32,
    pub w: f32,
    pub h: f32,
}

/// A rasterized font: coverage atlas plus per-glyph metrics.
pub struct Font {
    /// R8 coverage, `width` × `height`, row-major.
    pub atlas: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Indexed by `c as usize - FIRST as usize`.
    glyphs: Vec<Glyph>,
    /// Raster size the atlas was built at; `size / raster_em` scales to screen.
    pub raster_em: f32,
    /// Distance from one line's top to the next (ascent − descent + gap).
    pub line: f32,
    /// Baseline offset from the line's top.
    pub ascent: f32,
    /// UV of the solid cell's centre, for sampling rectangles.
    pub solid: [f32; 2],
}

impl Font {
    /// Rasterize `ttf` at `raster_em` pixels. `None` if the data is not a font.
    pub fn rasterize(ttf: &[u8], raster_em: u32) -> Option<Self> {
        let font = ab_glyph::FontVec::try_from_vec(ttf.to_vec()).ok()?;
        let scaled = font.as_scaled(raster_em as f32);
        let ascent = scaled.ascent();
        let descent = scaled.descent();
        let line = ascent - descent + scaled.line_gap();
        let advance = (FIRST..=LAST)
            .map(|c| scaled.h_advance(scaled.glyph_id(c)))
            .fold(0.0f32, f32::max);
        let cell_w = advance.ceil() as usize + PAD * 2;
        let cell_h = line.ceil() as usize + PAD * 2;
        let width = COLS * cell_w;
        let height = ROWS * cell_h;
        let mut atlas = vec![0u8; width * height];
        // Cell 0: solid block.
        for row in 0..cell_h {
            let start = row * width;
            atlas[start..start + cell_w].fill(255);
        }
        let solid = [
            (cell_w as f32 * 0.5) / width as f32,
            (cell_h as f32 * 0.5) / height as f32,
        ];

        let mut glyphs = Vec::with_capacity(COUNT);
        for (i, c) in (FIRST..=LAST).enumerate() {
            let idx = i + 1;
            let ox = (idx % COLS) * cell_w + PAD;
            let oy = (idx / COLS) * cell_h + PAD;
            let id = scaled.glyph_id(c);
            let advance = scaled.h_advance(id);
            // Space has no outline but still advances the pen.
            let glyph = ab_glyph::Glyph {
                id,
                scale: (raster_em as f32).into(),
                position: ab_glyph::point(0.0, 0.0),
            };
            let outlined = (c != ' ').then(|| scaled.outline_glyph(glyph)).flatten();
            let glyph = match outlined {
                None => Glyph {
                    u0: 0.0,
                    v0: 0.0,
                    u1: 0.0,
                    v1: 0.0,
                    advance,
                    min_x: 0.0,
                    min_y: 0.0,
                    w: 0.0,
                    h: 0.0,
                },
                Some(o) => {
                    let b = o.px_bounds();
                    let base_x = b.min.x.floor() as i64;
                    let base_y = b.min.y.floor() as i64;
                    let ox = ox as i64;
                    let oy = oy as i64;
                    o.draw(|x, y, cov| {
                        let px = ox + x as i64;
                        let py = oy + y as i64;
                        // Stay inside this glyph's cell; an oblique face could
                        // in theory overhang the padding.
                        if px >= ox - PAD as i64
                            && px < ox + advance.ceil() as i64 + PAD as i64
                            && py >= oy - PAD as i64
                            && py < oy + line.ceil() as i64 + PAD as i64
                            && px >= 0
                            && py >= 0
                            && (px as usize) < width
                            && (py as usize) < height
                        {
                            atlas[py as usize * width + px as usize] =
                                (cov.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    });
                    let ox = ox as usize;
                    let oy = oy as usize;
                    let gw = (b.max.x.ceil() - base_x as f32) as usize;
                    let gh = (b.max.y.ceil() - base_y as f32) as usize;
                    Glyph {
                        u0: ox as f32 / width as f32,
                        v0: oy as f32 / height as f32,
                        u1: (ox + gw) as f32 / width as f32,
                        v1: (oy + gh) as f32 / height as f32,
                        advance,
                        min_x: b.min.x,
                        min_y: b.min.y,
                        w: gw as f32,
                        h: gh as f32,
                    }
                }
            };
            glyphs.push(glyph);
        }

        Some(Self {
            atlas,
            width,
            height,
            glyphs,
            raster_em: raster_em as f32,
            line,
            ascent,
            solid,
        })
    }

    /// The cached glyph for `c`, if `c` is printable ASCII.
    pub fn glyph(&self, c: char) -> Option<&Glyph> {
        (FIRST..=LAST)
            .contains(&c)
            .then(|| &self.glyphs[c as usize - FIRST as usize])
    }

    /// Advance-scaled width of `text` at `size` screen pixels per em.
    /// Unknown characters are skipped, matching `UiPass::text`'s drawing.
    pub fn width(&self, text: &str, size: f32) -> f32 {
        let s = size / self.raster_em;
        text.chars()
            .filter_map(|c| self.glyph(c))
            .map(|g| g.advance * s)
            .sum()
    }

    /// One line's height at `size` screen pixels per em.
    pub fn height(&self, size: f32) -> f32 {
        self.line * size / self.raster_em
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTF: &[u8] = include_bytes!("../fonts/DejaVuSansMono.ttf");
    const EM: u32 = 18;

    fn font() -> Font {
        Font::rasterize(TTF, EM).expect("the bundled font rasterizes")
    }

    /// Coverage sum of one glyph's cell (its PAD border excluded).
    fn cell_coverage(f: &Font, c: char) -> u32 {
        let i = c as usize - FIRST as usize + 1;
        let cell_w = f.width / COLS;
        let cell_h = f.height / ROWS;
        let cx = (i % COLS) * cell_w + PAD;
        let cy = (i / COLS) * cell_h + PAD;
        let mut sum = 0;
        for y in cy..cy + cell_h - PAD * 2 {
            for x in cx..cx + cell_w - PAD * 2 {
                sum += f.atlas[y * f.width + x] as u32;
            }
        }
        sum
    }

    #[test]
    fn every_printable_glyph_rasterizes() {
        let f = font();
        assert_eq!(f.glyphs.len(), COUNT);
        for c in FIRST..=LAST {
            let g = f.glyph(c).unwrap_or_else(|| panic!("no glyph {c:?}"));
            assert!(g.advance > 0.0, "{c:?} advances the pen");
            if c == ' ' {
                assert_eq!(g.w, 0.0, "space draws nothing");
            } else {
                assert!(cell_coverage(&f, c) > 0, "{c:?} has coverage");
            }
        }
        assert!(f.glyph('é').is_none(), "outside the cached range");
    }

    #[test]
    fn advances_are_monospace() {
        let f = font();
        let advances: Vec<f32> = (FIRST..=LAST)
            .map(|c| f.glyph(c).unwrap().advance)
            .collect();
        for a in &advances {
            assert!(
                (a - advances[0]).abs() < 1e-3,
                "monospace: every advance equals {}, got {a}",
                advances[0]
            );
        }
    }

    #[test]
    fn rasterize_is_deterministic() {
        let a = font();
        let b = font();
        assert_eq!(a.atlas, b.atlas, "same input → same atlas bytes");
        assert_eq!(a.width("SAME", 24.0), b.width("SAME", 24.0));
        // Golden checksum: catches silent changes to raster_em/PAD/grid or the
        // bundled TTF. Update intentionally, never casually.
        let mut h = 0xcbf29ce484222325u64;
        for byte in &a.atlas {
            h ^= *byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for c in FIRST..=LAST {
            let g = a.glyph(c).unwrap();
            for v in [g.advance, g.min_x, g.min_y, g.w, g.h] {
                for byte in v.to_bits().to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        assert_eq!(h, 9116152690862709773, "golden atlas hash — see test docs");
    }

    #[test]
    fn glyph_cells_keep_their_padding_clear() {
        let f = font();
        let cell_w = f.width / COLS;
        let cell_h = f.height / ROWS;
        // Every glyph cell's PAD border must be empty (no bleed for LINEAR).
        for i in 1..=COUNT {
            let x0 = (i % COLS) * cell_w;
            let y0 = (i / COLS) * cell_h;
            for y in y0..y0 + cell_h {
                for x in [x0, x0 + cell_w - 1] {
                    assert_eq!(f.atlas[y * f.width + x], 0, "cell {i} x-border at {x},{y}");
                }
            }
            for x in x0..x0 + cell_w {
                for y in [y0, y0 + cell_h - 1] {
                    assert_eq!(f.atlas[y * f.width + x], 0, "cell {i} y-border at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn width_and_height_scale_with_size() {
        let f = font();
        let a = f.width("HELLO", 18.0);
        assert!(a > 0.0);
        assert!(
            (f.width("HELLO", 36.0) - a * 2.0).abs() < 1e-3,
            "width ∝ size"
        );
        assert!(
            (f.height(36.0) - f.height(18.0) * 2.0).abs() < 1e-3,
            "height ∝ size"
        );
        let adv = f.glyph('A').unwrap().advance;
        assert!(
            (a - 5.0 * adv).abs() < 1e-3,
            "width is five monospace advances"
        );
    }
}
