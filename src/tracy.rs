use std::{cell::RefCell, sync::OnceLock};
pub use tracy_client::{span_location, Span, SpanLocation};

struct Settings {
    depth: u16,
    every: u64,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| Settings {
        depth: std::env::var("PRPR_TRACY_CALLSTACK_DEPTH")
            .ok().and_then(|value| value.parse::<u16>().ok()).unwrap_or(32).min(62),
        every: std::env::var("PRPR_TRACY_CALLSTACK_EVERY")
            .ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(64).max(1),
    })
}

thread_local! {
    static COUNTERS: RefCell<Vec<(usize, u64)>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub fn span(location: &'static SpanLocation) -> Option<Span> {
    tracy_client::Client::running().map(|client| client.span(location, 0))
}

pub fn span_with_callstack(location: &'static SpanLocation) -> Option<Span> {
    let client = tracy_client::Client::running()?;
    let settings = settings();
    let depth = if settings.depth == 0 || !tracy_client::Client::is_connected() {
        0
    } else {
        COUNTERS.with(|counters| {
            let mut counters = counters.borrow_mut();
            let key = location as *const SpanLocation as usize;
            let index = counters.iter().position(|entry| entry.0 == key)
                .unwrap_or_else(|| { counters.push((key, 0)); counters.len() - 1 });
            let count = &mut counters[index].1;
            let depth = if *count == 0 { settings.depth } else { 0 };
            *count = (*count + 1) % settings.every;
            depth
        })
    };
    Some(client.span(location, depth))
}

/// A brief startup marker registers workers which may otherwise remain idle.
/// It measures registration only, not the worker's lifetime or CPU utilization.
pub fn register_thread(name: &str) {
    if let Some(client) = tracy_client::Client::running() {
        client.set_thread_name(name);
        let span = client.span(span_location!("Worker::register"), 0);
        span.emit_text(name);
    }
}
