//! The weather tables' global swap, in their own process. Unit tests share
//! one process and would race on the global, so everything that installs or
//! resets lives here, leaving no trace (a `reset` at the end of each test).

use feather_game::weather::{self, Key, WeatherData};

#[test]
fn installed_tables_replace_the_defaults_everywhere() {
    assert_eq!(weather::tables().len(), 3, "the compiled-in defaults first");
    assert_eq!(weather::choice_name(1), "CLEAR");

    // CLEAR's first two keys under a new name: a one-weather install is seen
    // by every reader.
    let (h0, k0) = weather::WEATHERS[0].keys[0];
    let (h1, k1) = weather::WEATHERS[0].keys[1];
    let noon = Key {
        sun_intensity: 3.5,
        ..k1
    };
    weather::install(vec![WeatherData {
        name: "MIST".into(),
        keys: vec![(h0, k0), (h1, noon)],
    }]);

    assert_eq!(weather::tables().len(), 1);
    assert_eq!(weather::tables()[0].name, "MIST");
    assert_eq!(weather::choice_name(1), "MIST");
    assert_eq!(
        weather::choice_name(2),
        "LEVEL",
        "a stale choice reads LEVEL"
    );
    assert_eq!(weather::choice_named("mist"), Some(1));
    assert_eq!(weather::choice_named("clear"), None);
    assert_eq!(weather::choice_named("level"), Some(0));
    let live: Vec<WeatherData> = weather::tables().iter().map(WeatherData::from).collect();
    assert!(weather::problems(&live).is_empty());

    // Sampling works on the installed table: halfway between the keys,
    // sun_intensity is the geometric mean of the two.
    let mid = weather::tables()[0].sample((h0 + h1) * 0.5);
    let want = (k0.sun_intensity * 3.5f32).sqrt();
    assert!(
        (mid.sun_intensity - want).abs() < 1e-6,
        "{}",
        mid.sun_intensity
    );

    weather::reset();
    assert_eq!(weather::tables().len(), 3);
    assert_eq!(weather::choice_name(1), "CLEAR");
}

#[test]
fn problems_names_every_broken_thing() {
    // One table with every rule broken at once; `problems` must name each.
    let base = weather::WEATHERS[0].keys[0].1;
    let mut dead_colour = base;
    dead_colour.sky_zenith *= 0.0;
    let mut bad_shape = base;
    bad_shape.fog_density = -1.0;
    let mut bad_exposure = base;
    bad_exposure.exposure_min = 4.0;
    bad_exposure.exposure_max = 2.0;
    let broken = WeatherData {
        name: "mist".into(), // lowercase: the menu font can't show it
        keys: vec![
            (12.0, base),
            (6.0, base),  // unsorted
            (25.0, base), // out of range
            (3.0, dead_colour),
            (4.0, bad_shape),
            (5.0, bad_exposure),
        ],
    };
    let empty = WeatherData {
        name: "EMPTY".into(),
        keys: Vec::new(),
    };
    let bad = weather::problems(&[broken, empty]);
    let joined = bad.join("\n");
    for needle in [
        "name must be A-Z",
        "no keys",
        "unsorted",
        "hour out of range",
        "colour",
        "shape",
        "exposure range",
    ] {
        assert!(joined.contains(needle), "{needle} missing from:\n{joined}");
    }
    // The compiled-in defaults pass their own rules.
    let data: Vec<WeatherData> = weather::WEATHERS.iter().map(WeatherData::from).collect();
    assert_eq!(weather::problems(&data), Vec::<String>::new());
}
