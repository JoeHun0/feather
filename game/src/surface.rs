//! Surfaces underfoot (§20): what a material, a collider and the player's
//! feet agree to call the ground, so the audio can pick its footsteps.

/// What the player is standing on, which picks the footstep sounds. A glTF
/// material names it in `extras.surface`, and each scene collider carries it
/// in rapier's `user_data` (`tag_surface`). The five are the Kenney pack's
/// footstep sets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Surface {
    /// Index 0, which is also rapier's default `user_data`: every collider
    /// nobody tagged (the built-in level, untagged scenes) is concrete.
    #[default]
    Concrete = 0,
    Grass,
    Wood,
    Carpet,
    Snow,
}

impl Surface {
    /// Every surface, in index order.
    pub const ALL: [Surface; 5] = [
        Surface::Concrete,
        Surface::Grass,
        Surface::Wood,
        Surface::Carpet,
        Surface::Snow,
    ];

    /// The name scenes use, which is also the pack's file stem.
    pub fn name(self) -> &'static str {
        match self {
            Surface::Concrete => "concrete",
            Surface::Grass => "grass",
            Surface::Wood => "wood",
            Surface::Carpet => "carpet",
            Surface::Snow => "snow",
        }
    }

    /// A surface named in a scene, in any case.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|s| s.name().eq_ignore_ascii_case(name))
    }

    /// Its position in [`Surface::ALL`]: per-surface tables use it, and it's
    /// the value colliders store.
    pub fn index(self) -> usize {
        self as usize
    }

    /// Back from [`Surface::index`]; anything out of range is concrete.
    pub fn from_index(i: usize) -> Self {
        Self::ALL.get(i).copied().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_names_round_trip() {
        for (i, s) in Surface::ALL.into_iter().enumerate() {
            assert_eq!(s.index(), i);
            assert_eq!(Surface::from_index(i), s);
            assert_eq!(Surface::from_name(s.name()), Some(s));
        }
        assert_eq!(Surface::from_name("Grass"), Some(Surface::Grass));
        assert_eq!(Surface::from_name("lava"), None);
        assert_eq!(Surface::from_index(99), Surface::Concrete);
        assert_eq!(Surface::default(), Surface::Concrete);
    }
}
