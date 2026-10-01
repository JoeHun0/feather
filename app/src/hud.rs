//! The in-game HUD (§19): crosshair, hit marker, health bar, kills. The
//! layout is a pure function of the framebuffer size — menu.rs's pattern —
//! so the tests check exactly what gets drawn. Drawn only while playing:
//! never over the menus, and never under `--bench`.

use feather_game::weapon::Score;
use feather_render::{Font, UiPass};

/// The HUD's geometry for one framebuffer size. Rects are `(x, y, w, h)` in
/// physical pixels, top-left origin (UiPass's space).
pub struct HudLayout {
    /// Screen px per font px: the scale unit for the crosshair and the bar.
    pub px: f32,
    /// Text size (em, screen px) the labels draw at.
    pub size: f32,
    /// Clearance from the screen edges for the anchored pieces.
    pub margin: f32,
    /// The crosshair's centre dot.
    pub dot: [f32; 4],
    /// Its four arms, up/down/left/right.
    pub ticks: [[f32; 4]; 4],
    /// The hit marker: four blocks on the diagonals around the centre
    /// (UiPass has no rotated rects, so an X reads as a diamond of blocks).
    pub marker: [[f32; 4]; 4],
    /// The health bar's back, bottom-left; the fill scales inside it.
    pub bar: [f32; 4],
    /// "HEALTH 100" sits just above the bar.
    pub health_text: (f32, f32),
    /// "KILLS N"'s baseline y; its x is right-anchored at draw time.
    pub kills_y: f32,
}

/// Where the HUD's pieces go on a `w`×`h` framebuffer. Pure, so the tests
/// can hold it to the screen.
pub fn hud_layout(font: &Font, w: f32, h: f32) -> HudLayout {
    let px = (h / 260.0).max(2.0).floor();
    let margin = px * 4.0;
    let (cx, cy) = (w * 0.5, h * 0.5);
    // Crosshair: a dot with four arms around a gap.
    let t = px.clamp(1.0, 2.0); // arm thickness
    let arm = px * 2.0;
    let gap = px * 1.5;
    let dot = [cx - t * 0.5, cy - t * 0.5, t, t];
    let ticks = [
        [cx - t * 0.5, cy - gap - arm, t, arm],
        [cx - t * 0.5, cy + gap, t, arm],
        [cx - gap - arm, cy - t * 0.5, arm, t],
        [cx + gap, cy - t * 0.5, arm, t],
    ];
    // Hit marker: blocks centred on the diagonals at the crosshair's ring.
    let m = px.clamp(2.0, 4.0);
    let r = gap + arm * 0.5;
    let o = m * 0.5;
    let marker = [
        [cx - r - o, cy - r - o, m, m],
        [cx + r - o, cy - r - o, m, m],
        [cx - r - o, cy + r - o, m, m],
        [cx + r - o, cy + r - o, m, m],
    ];
    let bar_h = px * 2.0;
    let bar = [margin, h - margin - bar_h, px * 22.0, bar_h];
    // Text size (em, screen px): same scale the menu uses.
    let size = (h / 36.0).max(10.0);
    let text_h = font.height(size);
    let health_text = (margin, bar[1] - text_h - px * 0.75);
    let kills_y = h - margin - text_h;
    HudLayout {
        px,
        size,
        margin,
        dot,
        ticks,
        marker,
        bar,
        health_text,
        kills_y,
    }
}

/// Draw one frame's HUD. `health` is the 0–1 fraction; `frame` is
/// `FrameCount`, for the hit marker's fade.
pub fn draw(ui: &mut UiPass, w: f32, h: f32, health: f32, score: &Score, frame: u64) {
    let l = hud_layout(ui.font(), w, h);
    let px = l.px;
    let size = l.size;
    let cross = [0.9, 0.9, 0.9, 0.75];
    ui.rect(l.dot[0], l.dot[1], l.dot[2], l.dot[3], cross);
    for t in &l.ticks {
        ui.rect(t[0], t[1], t[2], t[3], cross);
    }
    // The hit marker fades over 12 frames (~0.2 s at 60 Hz).
    let age = frame.saturating_sub(score.last_hit_frame);
    if age < 12 {
        let a = 0.9 * (1.0 - age as f32 / 12.0);
        for m in &l.marker {
            ui.rect(m[0], m[1], m[2], m[3], [1.0, 0.95, 0.7, a]);
        }
    }
    // Health: a bar and its number, anchored bottom-left.
    ui.rect(
        l.bar[0],
        l.bar[1],
        l.bar[2],
        l.bar[3],
        [0.1, 0.1, 0.1, 0.55],
    );
    let inset = (px * 0.4).max(1.0);
    if health > 0.0 {
        ui.rect(
            l.bar[0] + inset,
            l.bar[1] + inset,
            (l.bar[2] - 2.0 * inset) * health,
            (l.bar[3] - 2.0 * inset).max(1.0),
            [0.55, 0.8, 0.35, 0.9],
        );
    }
    let health_text = format!("HEALTH {}", (health * 100.0).round() as u32);
    ui.text(
        l.health_text.0,
        l.health_text.1,
        size,
        [0.85, 0.85, 0.85, 0.9],
        &health_text,
    );
    // Kills, bottom-right, right-anchored.
    let kills = format!("KILLS {}", score.kills);
    ui.text(
        w - l.margin - ui.text_width(&kills, size),
        l.kills_y,
        size,
        [0.85, 0.85, 0.85, 0.9],
        &kills,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same font the renderer draws with.
    fn font() -> Font {
        Font::rasterize(include_bytes!("../../render/fonts/DejaVuSansMono.ttf"), 18)
            .expect("the bundled font rasterizes")
    }

    /// One source of truth for every size the layout must hold to: the
    /// crosshair centred, everything on screen, the anchors where they
    /// belong. (Mutants that nudge the centre or swap an anchor must fail
    /// here.)
    #[test]
    fn layout_is_centered_on_screen_and_anchored() {
        let font = font();
        for (w, h) in [(1920.0, 1080.0), (1280.0, 720.0), (800.0, 600.0)] {
            let l = hud_layout(&font, w, h);
            let (cx, cy) = (w * 0.5, h * 0.5);
            assert_eq!(l.dot[0] + l.dot[2] * 0.5, cx, "dot centred, {w}x{h}");
            assert_eq!(l.dot[1] + l.dot[3] * 0.5, cy, "dot centred, {w}x{h}");
            // The arms mirror about the centre: the up arm's far end is as
            // far above centre as the down arm's is below.
            let (up, down) = (&l.ticks[0], &l.ticks[1]);
            assert_eq!(up[1] + up[3], cy - (down[1] - cy), "arms mirror, {w}x{h}");
            let (left, right) = (&l.ticks[2], &l.ticks[3]);
            assert_eq!(
                left[0] + left[2],
                cx - (right[0] - cx),
                "arms mirror, {w}x{h}"
            );
            for r in [l.dot, l.bar].into_iter().chain(l.ticks).chain(l.marker) {
                assert!(r[0] >= 0.0 && r[1] >= 0.0, "on screen {w}x{h}: {r:?}");
                assert!(
                    r[0] + r[2] <= w && r[1] + r[3] <= h,
                    "on screen {w}x{h}: {r:?}"
                );
            }
            // Anchors: health bottom-left, kills bottom-right, number above bar.
            assert!(
                l.bar[0] + l.bar[2] <= w * 0.5,
                "bar in the left half, {w}x{h}"
            );
            assert!(
                l.bar[1] + l.bar[3] <= h - l.margin + 1e-6,
                "bar bottom, {w}x{h}"
            );
            assert!(
                l.health_text.0 >= l.margin - 1e-6,
                "health text left, {w}x{h}"
            );
            assert!(l.health_text.1 < l.bar[1], "number above the bar, {w}x{h}");
            assert!(
                l.kills_y + font.height(l.size) <= h - l.margin + 1e-6,
                "kills bottom, {w}x{h}"
            );
        }
    }

    /// Small windows still get a usable HUD: the font floor keeps text
    /// legible and the bar stays a bar.
    #[test]
    fn tiny_windows_stay_legible() {
        let l = hud_layout(&font(), 640.0, 480.0);
        assert!(l.px >= 2.0, "the px unit has a floor");
        assert!(l.size >= 10.0, "the font has a floor");
        assert!(l.bar[2] > l.bar[3] * 3.0, "the bar is wider than tall");
        assert!(l.margin >= l.px, "the margin clears the text");
    }
}
