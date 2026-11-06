//! GPT-4o's tokenizer (o200k_base) written in plain Rust.
//!
//! Same algorithm as tok_python.py, line for line: a regex cuts the text into pieces, then each piece
//! starts as single bytes and the lowest-ranked pair is merged until no pair is in the vocabulary.
//! Two versions of the merge loop: `copies` keeps every part as its own byte vector, like Python's bytes
//! objects; `slices` keeps windows into the original piece and copies nothing. Only the standard HashMap is used.
use anyhow::{Context, Result};
use base64::Engine;
use fancy_regex::Regex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

fn load(dir: &str) -> Result<(HashMap<Vec<u8>, u32>, Regex)> {
    let text = std::fs::read_to_string(format!("{dir}/o200k.ranks"))?;
    let mut ranks = HashMap::new();
    for line in text.lines() {
        let (tok, rank) = line.split_once(' ').context("bad ranks line")?;
        ranks.insert(base64::engine::general_purpose::STANDARD.decode(tok)?, rank.parse()?);
    }
    let pat = Regex::new(&std::fs::read_to_string(format!("{dir}/o200k.pattern"))?)?;
    Ok((ranks, pat))
}

fn bpe(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
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
fn bpe_slices(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
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

type Merge = fn(&[u8], &HashMap<Vec<u8>, u32>) -> Vec<u32>;

fn encode(text: &str, ranks: &HashMap<Vec<u8>, u32>, pat: &Regex, bpe: Merge) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for m in pat.find_iter(text) {
        let piece = m?.as_str().as_bytes();
        match ranks.get(piece) {
            Some(&r) => out.push(r),
            None => out.extend(bpe(piece, ranks)),
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: tok_rust <bench dir> <workers> [dataset] [copies|slices] [merge-only]")?;
    let workers: usize = std::env::args().nth(2).map_or(Ok(1), |w| w.parse())?;
    let data = std::env::args().nth(3).unwrap_or_else(|| "books".into());
    let style = std::env::args().nth(4).unwrap_or_else(|| "copies".into());
    let bpe: Merge = match style.as_str() {
        "copies" => bpe,
        "slices" => bpe_slices,
        other => anyhow::bail!("unknown style {other}: use copies or slices"),
    };
    let (ranks, pat) = load(&dir)?;
    let docs: Vec<String> = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/corpus/docs-{data}.json"))?)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;

    if std::env::args().nth(5).as_deref() == Some("merge-only") {
        // time only the merge loop: the pieces that are not one vocabulary entry, cut out before the clock starts
        let mut pieces: Vec<Vec<u8>> = Vec::new();
        for d in &docs {
            for m in pat.find_iter(d) {
                let p = m?.as_str().as_bytes();
                if !ranks.contains_key(p) {
                    pieces.push(p.to_vec());
                }
            }
        }
        let t = Instant::now();
        let n: usize = pieces.iter().map(|p| bpe(p, &ranks).len()).sum();
        let secs = t.elapsed().as_secs_f64();
        println!(r#"{{"impl": "rust-{style}", "data": "{data}", "mode": "merge-only", "pieces": {}, "seconds": {secs}, "tokens": {n}}}"#, pieces.len());
        return Ok(());
    }

    let t = Instant::now();
    let tokens: Vec<Vec<u32>> = if workers == 1 {
        docs.iter().map(|d| encode(d, &ranks, &pat, bpe)).collect::<Result<_>>()?
    } else {
        // each thread gets its own copy of the regex: threads sharing one regex wait on its internal cache
        // (tiktoken does the same, for the same reason)
        pool.install(|| docs.par_iter().map_init(|| pat.clone(), |p, d| encode(d, &ranks, p, bpe)).collect::<Result<_>>())?
    };
    let secs = t.elapsed().as_secs_f64();

    std::fs::create_dir_all(format!("{dir}/out"))?;
    std::fs::write(format!("{dir}/out/rust-{style}-{data}-{workers}.json"), serde_json::to_string(&tokens)?)?;
    let n: usize = tokens.iter().map(Vec::len).sum();
    println!(r#"{{"impl": "rust-{style}", "data": "{data}", "workers": {workers}, "seconds": {secs}, "tokens": {n}}}"#);
    Ok(())
}
