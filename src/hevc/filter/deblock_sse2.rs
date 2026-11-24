//! The luma deblocking filter of one segment on SSE2, for samples of at most
//! [`MAX_BIT_DEPTH`] bits: [`super::luma_lines`] with the four lines in the
//! lanes of a vector.
//!
//! At that depth every intermediate value of clauses 8.7.2.5.3 and 8.7.2.5.7
//! fits a 16-bit lane (the widest, `9 * (q0 - p0) - 3 * (q1 - p1) + 8`, stays
//! within twelve times the largest sample), so the lanes hold the samples
//! as they are and SSE2's 16-bit minimum and maximum do the clipping. SSE2 is
//! part of every x86-64 processor, so there is nothing to detect.
//!
//! Besides entering the module, the only unsafe operations are the loads and
//! stores, each of which reads or writes exactly the slice it is given.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmplt_epi16, _mm_extract_epi16,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_max_epi16, _mm_min_epi16, _mm_or_si128, _mm_set1_epi16,
    _mm_setzero_si128, _mm_slli_epi16, _mm_srai_epi16, _mm_storel_epi64, _mm_storeu_si128,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use super::{Edge, Seg};

/// The deepest samples the 16-bit lanes take without overflowing.
pub(super) const MAX_BIT_DEPTH: u8 = 10;

/// Taps `p3, p2, p1, p0, q0, q1, q2, q3` of a segment, line `i` in lane `i`;
/// lanes 4 to 7 are not used.
type Taps = [__m128i; 8];

/// [`super::luma_lines`] on SSE2: the same decisions and the same samples
/// written, for samples of at most [`MAX_BIT_DEPTH`] bits.
pub(super) fn luma_lines(data: &mut [u16], seg: &Seg, edge: &Edge) {
    // SAFETY: SSE2 is part of every x86-64 processor.
    unsafe { lines(data, seg, edge) }
}

#[target_feature(enable = "sse2")]
fn lines(data: &mut [u16], seg: &Seg, edge: &Edge) {
    let Edge {
        ver,
        beta,
        tc,
        locked: (pl, ql),
        max,
    } = *edge;
    let t = read(data, seg, ver);
    let [p3, p2, p1, p0, q0, q1, q2, q3] = t;
    // dp and dq of every line: |p2 - 2 * p1 + p0| and |q2 - 2 * q1 + q0|.
    let dp = abs(sub(add(p2, p0), twice(p1)));
    let dq = abs(sub(add(q2, q0), twice(q1)));
    let (dp0, dq0) = (_mm_extract_epi16::<0>(dp), _mm_extract_epi16::<0>(dq));
    let (dp3, dq3) = (_mm_extract_epi16::<3>(dp), _mm_extract_epi16::<3>(dq));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= beta {
        return;
    }
    // Every value read out of a lane here is in 0..=0x7fff.
    let flat = add(abs(sub(p3, p0)), abs(sub(q0, q3)));
    let step = abs(sub(p0, q0));
    let sam = |dpq: i32, flat: i32, step: i32| {
        2 * dpq < (beta >> 2) && flat < (beta >> 3) && step < ((5 * tc + 1) >> 1)
    };
    let strong_lines = sam(
        dpq0,
        _mm_extract_epi16::<0>(flat),
        _mm_extract_epi16::<0>(step),
    ) && sam(
        dpq3,
        _mm_extract_epi16::<3>(flat),
        _mm_extract_epi16::<3>(step),
    );
    let thr = (beta + (beta >> 1)) >> 3;
    let new = if strong_lines {
        strong(&t, tc)
    } else {
        weak(&t, (dp0 + dp3 < thr, dq0 + dq3 < thr), tc, max)
    };
    let mut o = t;
    if !pl {
        o[1..4].copy_from_slice(&new[..3]);
    }
    if !ql {
        o[4..7].copy_from_slice(&new[3..]);
    }
    // Only p2..q2 can change, and a locked side keeps its samples.
    let lo = if pl { 4 } else { 1 };
    let hi = if ql { 4 } else { 7 };
    write(data, seg, ver, &o, (lo, hi));
}

/// Eight samples from the start of `s`.
#[target_feature(enable = "sse2")]
fn load8(s: &[u16]) -> __m128i {
    let s = &s[..8];
    // SAFETY: `s` holds eight `u16`, the sixteen bytes `_mm_loadu_si128` reads; it takes any alignment.
    unsafe { _mm_loadu_si128(s.as_ptr().cast()) }
}

/// Four samples from the start of `s`, in lanes 0 to 3.
#[target_feature(enable = "sse2")]
fn load4(s: &[u16]) -> __m128i {
    let s = &s[..4];
    // SAFETY: `s` holds four `u16`, the eight bytes `_mm_loadl_epi64` reads; it takes any alignment.
    unsafe { _mm_loadl_epi64(s.as_ptr().cast()) }
}

/// Writes lanes 0 to 7 over the first eight samples of `s`.
#[target_feature(enable = "sse2")]
fn store8(s: &mut [u16], v: __m128i) {
    let s = &mut s[..8];
    // SAFETY: `s` holds eight `u16`, the sixteen bytes `_mm_storeu_si128` writes; it takes any alignment.
    unsafe { _mm_storeu_si128(s.as_mut_ptr().cast(), v) }
}

/// Writes lanes 0 to 3 over the first four samples of `s`.
#[target_feature(enable = "sse2")]
fn store4(s: &mut [u16], v: __m128i) {
    let s = &mut s[..4];
    // SAFETY: `s` holds four `u16`, the eight bytes `_mm_storel_epi64` writes; it takes any alignment.
    unsafe { _mm_storel_epi64(s.as_mut_ptr().cast(), v) }
}

/// The taps of a segment, read the way its edge runs.
#[target_feature(enable = "sse2")]
fn read(data: &[u16], seg: &Seg, ver: bool) -> Taps {
    let mut t = [_mm_setzero_si128(); 8];
    if !ver {
        // A tap of a horizontal edge is four samples of one row.
        for (j, tap) in t.iter_mut().enumerate() {
            *tap = load4(&data[seg.at(0, j)..]);
        }
        return t;
    }
    // A line of a vertical edge is one row of eight taps: transpose four rows.
    let r = [
        load8(&data[seg.at(0, 0)..]),
        load8(&data[seg.at(1, 0)..]),
        load8(&data[seg.at(2, 0)..]),
        load8(&data[seg.at(3, 0)..]),
    ];
    let a = _mm_unpacklo_epi16(r[0], r[1]);
    let b = _mm_unpacklo_epi16(r[2], r[3]);
    let c = _mm_unpackhi_epi16(r[0], r[1]);
    let d = _mm_unpackhi_epi16(r[2], r[3]);
    // Each holds two taps, the four lines of one in lanes 0 to 3 and of the other in lanes 4 to 7.
    let pairs = [
        _mm_unpacklo_epi32(a, b),
        _mm_unpackhi_epi32(a, b),
        _mm_unpacklo_epi32(c, d),
        _mm_unpackhi_epi32(c, d),
    ];
    for (k, &pair) in pairs.iter().enumerate() {
        t[2 * k] = pair;
        t[2 * k + 1] = _mm_unpackhi_epi64(pair, pair);
    }
    t
}

/// Writes taps `lo..hi` of a segment back, the way its edge runs.
#[target_feature(enable = "sse2")]
fn write(data: &mut [u16], seg: &Seg, ver: bool, t: &Taps, (lo, hi): (usize, usize)) {
    if !ver {
        for (j, &tap) in t.iter().enumerate().take(hi).skip(lo) {
            store4(&mut data[seg.at(0, j)..], tap);
        }
        return;
    }
    // Transpose back to four rows of eight taps. Taps outside `lo..hi` hold the samples as they were read, so the
    // whole row is written.
    let u = [
        _mm_unpacklo_epi16(t[0], t[1]),
        _mm_unpacklo_epi16(t[2], t[3]),
        _mm_unpacklo_epi16(t[4], t[5]),
        _mm_unpacklo_epi16(t[6], t[7]),
    ];
    let v = [
        _mm_unpacklo_epi32(u[0], u[1]),
        _mm_unpackhi_epi32(u[0], u[1]),
        _mm_unpacklo_epi32(u[2], u[3]),
        _mm_unpackhi_epi32(u[2], u[3]),
    ];
    let rows = [
        _mm_unpacklo_epi64(v[0], v[2]),
        _mm_unpackhi_epi64(v[0], v[2]),
        _mm_unpacklo_epi64(v[1], v[3]),
        _mm_unpackhi_epi64(v[1], v[3]),
    ];
    for (i, &row) in rows.iter().enumerate() {
        store8(&mut data[seg.at(i, 0)..], row);
    }
}

#[target_feature(enable = "sse2")]
fn add(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi16(a, b)
}

#[target_feature(enable = "sse2")]
fn sub(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi16(a, b)
}

/// `2 * a`.
#[target_feature(enable = "sse2")]
fn twice(a: __m128i) -> __m128i {
    _mm_slli_epi16::<1>(a)
}

#[target_feature(enable = "sse2")]
fn abs(a: __m128i) -> __m128i {
    _mm_max_epi16(a, _mm_sub_epi16(_mm_setzero_si128(), a))
}

/// `Clip3(lo, hi, v)` lane by lane.
#[target_feature(enable = "sse2")]
fn clip3(lo: __m128i, hi: __m128i, v: __m128i) -> __m128i {
    _mm_min_epi16(_mm_max_epi16(v, lo), hi)
}

/// `Clip3(s - r, s + r, v)` lane by lane.
#[target_feature(enable = "sse2")]
fn near(s: __m128i, r: __m128i, v: __m128i) -> __m128i {
    clip3(sub(s, r), add(s, r), v)
}

/// `if on { a } else { b }` lane by lane, `on` all ones or all zeros.
#[target_feature(enable = "sse2")]
fn select(on: __m128i, a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(on, a), _mm_andnot_si128(on, b))
}

/// The strong filter, `dE` 2: new `p2, p1, p0, q0, q1, q2`.
#[target_feature(enable = "sse2")]
fn strong(t: &Taps, tc: i32) -> [__m128i; 6] {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *t;
    let tc2 = _mm_set1_epi16((2 * tc) as i16);
    let (four, two) = (_mm_set1_epi16(4), _mm_set1_epi16(2));
    let pq = add(p0, q0);
    // (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3
    let np0 = _mm_srai_epi16::<3>(add(add(p2, q1), add(twice(add(p1, pq)), four)));
    // (p2 + p1 + p0 + q0 + 2) >> 2
    let np1 = _mm_srai_epi16::<2>(add(add(p2, p1), add(pq, two)));
    // (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3
    let np2 = _mm_srai_epi16::<3>(add(add(twice(add(p3, p2)), p2), add(add(p1, pq), four)));
    // (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3
    let nq0 = _mm_srai_epi16::<3>(add(add(p1, q2), add(twice(add(pq, q1)), four)));
    // (p0 + q0 + q1 + q2 + 2) >> 2
    let nq1 = _mm_srai_epi16::<2>(add(add(pq, q1), add(q2, two)));
    // (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3
    let nq2 = _mm_srai_epi16::<3>(add(add(twice(add(q3, q2)), q2), add(add(pq, q1), four)));
    [
        near(p2, tc2, np2),
        near(p1, tc2, np1),
        near(p0, tc2, np0),
        near(q0, tc2, nq0),
        near(q1, tc2, nq1),
        near(q2, tc2, nq2),
    ]
}

/// The weak filter, `dE` 1: new `p2, p1, p0, q0, q1, q2` (`p2` and `q2` as they were).
#[target_feature(enable = "sse2")]
fn weak(t: &Taps, (dep, deq): (bool, bool), tc: i32, max: i32) -> [__m128i; 6] {
    let [_, p2, p1, p0, q0, q1, q2, _] = *t;
    let zero = _mm_setzero_si128();
    let top = _mm_set1_epi16(max as i16);
    let tcv = _mm_set1_epi16(tc as i16);
    let half = _mm_set1_epi16((tc >> 1) as i16);
    // (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4
    let a = sub(q0, p0);
    let b = sub(q1, p1);
    let nine_a = add(_mm_slli_epi16::<3>(a), a);
    let three_b = add(twice(b), b);
    let raw = _mm_srai_epi16::<4>(add(sub(nine_a, three_b), _mm_set1_epi16(8)));
    // Each line is filtered only where |raw| < tc * 10.
    let on = _mm_cmplt_epi16(abs(raw), _mm_set1_epi16((tc * 10) as i16));
    let d = near(zero, tcv, raw);
    let np0 = select(on, clip3(zero, top, add(p0, d)), p0);
    let nq0 = select(on, clip3(zero, top, sub(q0, d)), q0);
    let one = _mm_set1_epi16(1);
    let np1 = if dep {
        // (((p2 + p0 + 1) >> 1) - p1 + d) >> 1
        let mid = _mm_srai_epi16::<1>(add(add(p2, p0), one));
        let dv = near(zero, half, _mm_srai_epi16::<1>(add(sub(mid, p1), d)));
        select(on, clip3(zero, top, add(p1, dv)), p1)
    } else {
        p1
    };
    let nq1 = if deq {
        // (((q2 + q0 + 1) >> 1) - q1 - d) >> 1
        let mid = _mm_srai_epi16::<1>(add(add(q2, q0), one));
        let dv = near(zero, half, _mm_srai_epi16::<1>(sub(sub(mid, q1), d)));
        select(on, clip3(zero, top, add(q1, dv)), q1)
    } else {
        q1
    };
    [p2, np1, np0, nq0, nq1, q2]
}
