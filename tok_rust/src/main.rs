//! GPT-4o's tokenizer (o200k_base) written in plain Rust.
//!
//! Same algorithm as tok_python.py, line for line: a regex cuts the text into pieces, then each piece
//! starts as single bytes and the lowest-ranked pair is merged until no pair is in the vocabulary.
//!
//! The style argument picks the version. `copies` and `slices` are the two in the video:
//! - `copies` keeps every part as its own byte vector, like Python's bytes objects;
//! - `slices` keeps windows into the original piece and copies nothing.
//!
//! The optimisation round adds, one at a time (joined with `+`, e.g. `cached+fx+fastre`):
//! - `fx`: FxHash instead of the standard SipHash for the vocabulary (tiktoken does this);
//! - `cached`: tiktoken's merge loop: each pair's rank is remembered, so a merge looks up only its two
//!   new neighbours instead of every pair again;
//! - `fastre`: the `regex` crate instead of `fancy-regex`. o200k's pattern has one look-ahead, `\s+(?!\S)`,
//!   which `regex` can't run; it is done by hand after the match (see `split_fast`).
//! Build-time options (allocator, target-cpu, PGO) are in bench_opt.py.
use anyhow::{Context, Result};
use base64::Engine;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::time::Instant;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type Ranks<S> = HashMap<Vec<u8>, u32, S>;

fn load<S: BuildHasher + Default>(dir: &str) -> Result<(Ranks<S>, String)> {
    let text = std::fs::read_to_string(format!("{dir}/o200k.ranks"))?;
    let mut ranks = Ranks::<S>::default();
    for line in text.lines() {
        let (tok, rank) = line.split_once(' ').context("bad ranks line")?;
        ranks.insert(base64::engine::general_purpose::STANDARD.decode(tok)?, rank.parse()?);
    }
    Ok((ranks, std::fs::read_to_string(format!("{dir}/o200k.pattern"))?))
}

fn bpe<S: BuildHasher>(piece: &[u8], ranks: &Ranks<S>) -> Vec<u32> {
    let mut parts: Vec<Vec<u8>> = piece.iter().map(|b| vec![*b]).collect();
    while parts.len() > 1 {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..parts.len() - 1 {
            let pair = [parts[i].as_slice(), parts[i + 1].as_slice()].concat();
            if let Some(&r) = ranks.get(&pair) {
                if best.map_or(true, |(b, _)| r < b) {
                    best = Some((r, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let right = parts.remove(i + 1);
        parts[i].extend(right);
    }
    parts.iter().map(|p| ranks[p]).collect()
}

/// The same algorithm written the Rust way: a part is a window into `piece` (a slice), so nothing is copied.
/// `cuts` holds where each part starts; merging two parts just deletes the cut between them.
fn bpe_slices<S: BuildHasher>(piece: &[u8], ranks: &Ranks<S>) -> Vec<u32> {
    let mut cuts: Vec<usize> = (0..=piece.len()).collect();
    while cuts.len() > 2 {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..cuts.len() - 2 {
            if let Some(&r) = ranks.get(&piece[cuts[i]..cuts[i + 2]]) {
                if best.map_or(true, |(b, _)| r < b) {
                    best = Some((r, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        cuts.remove(i + 1);
    }
    cuts.windows(2).map(|w| ranks[&piece[w[0]..w[1]]]).collect()
}

/// tiktoken's merge loop (`_byte_pair_merge`): next to each cut, the rank of the pair that starts there.
/// After a merge only the two pairs that touch it can change, so only those two are looked up again;
/// finding the lowest rank is then a scan over plain numbers, with no hashing.
fn bpe_cached<S: BuildHasher>(piece: &[u8], ranks: &Ranks<S>) -> Vec<u32> {
    let rank_of = |a: usize, b: usize| *ranks.get(&piece[a..b]).unwrap_or(&u32::MAX);
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
    for i in 0..piece.len() - 1 {
        parts.push((i, rank_of(i, i + 2)));
    }
    parts.push((piece.len() - 1, u32::MAX));
    parts.push((piece.len(), u32::MAX));
    loop {
        let (mut min, mut at) = (u32::MAX, 0);
        for (i, &(_, r)) in parts[..parts.len() - 1].iter().enumerate() {
            if r < min {
                (min, at) = (r, i);
            }
        }
        if min == u32::MAX {
            break;
        }
        // parts[at] and parts[at + 1] become one; the pairs starting at at-1 and at now reach one part further
        // (+3 because parts[at + 1] is still there)
        let reach = |parts: &Vec<(usize, u32)>, i: usize| if i + 3 < parts.len() { rank_of(parts[i].0, parts[i + 3].0) } else { u32::MAX };
        if at > 0 {
            parts[at - 1].1 = reach(&parts, at - 1);
        }
        parts[at].1 = reach(&parts, at);
        parts.remove(at + 1);
    }
    parts.windows(2).map(|w| ranks[&piece[w[0].0..w[1].0]]).collect()
}

/// The regex step, with either engine. `Fast` is the `regex` crate with o200k's look-ahead branch
/// `\s+(?!\S)|\s+` written as plain `\s+`; the look-ahead is then applied by hand: a run of spaces that is
/// followed by a non-space gives its last space to the next piece (so " world" stays one piece).
enum Splitter {
    Fancy(fancy_regex::Regex),
    Fast(regex::Regex),
}

impl Splitter {
    fn new(pattern: &str, fast: bool) -> Result<Self> {
        if !fast {
            return Ok(Splitter::Fancy(fancy_regex::Regex::new(pattern)?));
        }
        let plain = pattern.replace(r"|\s+(?!\S)|\s+", r"|\s+");
        anyhow::ensure!(plain != pattern, "pattern changed: the look-ahead branch was not found");
        Ok(Splitter::Fast(regex::Regex::new(&plain)?))
    }

    fn pieces<'t>(&self, text: &'t str, mut each: impl FnMut(&'t [u8])) -> Result<()> {
        match self {
            Splitter::Fancy(re) => {
                for m in re.find_iter(text) {
                    each(m?.as_str().as_bytes());
                }
            }
            Splitter::Fast(re) => {
                let mut at = 0;
                while let Some(m) = re.find_at(text, at) {
                    let mut end = m.end();
                    let s = m.as_str();
                    // only the `\s+` branch can give back a space: a match of only whitespace that does not end in
                    // a line break (those come from `\s*[\r\n]+`), is more than one character, and is followed by
                    // something that is not whitespace
                    if s.chars().all(char::is_whitespace) && !s.ends_with(['\r', '\n']) {
                        if let Some(next) = text[end..].chars().next() {
                            let last = s.chars().next_back().map_or(0, char::len_utf8);
                            if !next.is_whitespace() && s.len() > last {
                                end -= last;
                            }
                        }
                    }
                    each(&text.as_bytes()[m.start()..end]);
                    at = end;
                }
            }
        }
        Ok(())
    }
}

type Merge<S> = fn(&[u8], &Ranks<S>) -> Vec<u32>;

fn encode<S: BuildHasher>(text: &str, ranks: &Ranks<S>, re: &Splitter, bpe: Merge<S>) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    re.pieces(text, |piece| match ranks.get(piece) {
        Some(&r) => out.push(r),
        None => out.extend(bpe(piece, ranks)),
    })?;
    Ok(out)
}

fn run<S: BuildHasher + Default + Sync>(dir: &str, workers: usize, data: &str, style: &str, mode: Option<&str>) -> Result<()> {
    let has = |f: &str| style.split('+').any(|x| x == f);
    let bpe: Merge<S> = if has("copies") {
        bpe
    } else if has("cached") {
        bpe_cached
    } else {
        bpe_slices
    };
    let (ranks, pattern) = load::<S>(dir)?;
    let re = Splitter::new(&pattern, has("fastre"))?;
    let docs: Vec<String> = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/corpus/docs-{data}.json"))?)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;

    match mode {
        Some("merge-only") => {
            // time only the merge loop: the pieces that are not one vocabulary entry, cut out before the clock starts
            let mut pieces: Vec<Vec<u8>> = Vec::new();
            for d in &docs {
                re.pieces(d, |p| {
                    if !ranks.contains_key(p) {
                        pieces.push(p.to_vec());
                    }
                })?;
            }
            let t = Instant::now();
            let n: usize = pieces.iter().map(|p| bpe(p, &ranks).len()).sum();
            let secs = t.elapsed().as_secs_f64();
            println!(r#"{{"impl": "rust-{style}", "data": "{data}", "mode": "merge-only", "pieces": {}, "seconds": {secs}, "tokens": {n}}}"#, pieces.len());
            return Ok(());
        }
        Some("split-only") => {
            // time only the regex step: cut every document into pieces, look nothing up
            let t = Instant::now();
            let mut n = 0usize;
            for d in &docs {
                re.pieces(d, |_| n += 1)?;
            }
            let secs = t.elapsed().as_secs_f64();
            println!(r#"{{"impl": "rust-{style}", "data": "{data}", "mode": "split-only", "pieces": {n}, "seconds": {secs}}}"#);
            return Ok(());
        }
        Some(other) => anyhow::bail!("unknown mode {other}: use merge-only or split-only"),
        None => {}
    }

    let t = Instant::now();
    let tokens: Vec<Vec<u32>> = if workers == 1 {
        docs.iter().map(|d| encode(d, &ranks, &re, bpe)).collect::<Result<_>>()?
    } else {
        // each thread gets its own copy of the regex: threads sharing one regex wait on its internal cache
        // (tiktoken does the same, for the same reason)
        pool.install(|| {
            docs.par_iter()
                .map_init(|| Splitter::new(&pattern, has("fastre")).expect("pattern compiled once already"), |r, d| encode(d, &ranks, r, bpe))
                .collect::<Result<_>>()
        })?
    };
    let secs = t.elapsed().as_secs_f64();

    std::fs::create_dir_all(format!("{dir}/out"))?;
    std::fs::write(format!("{dir}/out/rust-{style}-{data}-{workers}.json"), serde_json::to_string(&tokens)?)?;
    let n: usize = tokens.iter().map(Vec::len).sum();
    println!(r#"{{"impl": "rust-{style}", "data": "{data}", "workers": {workers}, "seconds": {secs}, "tokens": {n}}}"#);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).context("usage: tok_rust <bench dir> <workers> [dataset] [style, e.g. slices or cached+fx+fastre] [merge-only|split-only]")?;
    let workers: usize = args.get(2).map_or(Ok(1), |w| w.parse())?;
    let data = args.get(3).map_or("books", String::as_str);
    let style = args.get(4).map_or("copies", String::as_str);
    let mode = args.get(5).map(String::as_str);
    for f in style.split('+') {
        anyhow::ensure!(["copies", "slices", "cached", "fx", "fastre"].contains(&f), "unknown style part {f}");
    }
    if style.split('+').any(|x| x == "fx") {
        run::<rustc_hash::FxBuildHasher>(dir, workers, data, style, mode)
    } else {
        run::<std::hash::RandomState>(dir, workers, data, style, mode)
    }
}
