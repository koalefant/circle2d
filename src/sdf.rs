use crate::*;

pub fn sd_circle(p: Real2, center: Real2, r: Real)->Real {
    (p - center).magnitude() - r
}

pub fn march_circle<F: Fn(Real2)->Real>(start: Real2, end: Real2, radius: Real, distance_func: F)->(Real2, bool) {
    let delta = end - start;
    let max_length = delta.magnitude();
    //assert!(max_length.is_finite());
    let start_d = distance_func(start);
    if start_d < reali(0) {
        return (start, true);
    }
    if max_length == reali(0) {
        if start_d < radius {
            return (start, true);
        } else {
            return (end, false);
        }
    }
    let dir = delta / max_length;

    let mut l = reali(0);
    let mut hit = false;
    let forward_step = reali(1);
    let backward_step = Real::from(0.125);
    loop {
        let point = start + dir * l;
        let d = distance_func(point);
        if d >= radius {
            // to make sure that we can continue marching in d == 0 areas we overstep
            l = (l + (d - radius).max(forward_step)).min(max_length);
        }
        if d < radius {
            hit = true;
            break;
        }
        if l == max_length {
            // it is possible that we are hitting here, due to overstepping
            break;
        }
    }

    let mut b = l;
    while b > reali(0) {
        let point = start + dir * b;
        let d = distance_func(point) - radius;
        if d >= reali(0) {
            if b == max_length {
                return (end, hit)
            } else {
                let result = start + dir * b;
                assert!(hit == true || result == end);
                return (result, hit);
            }
        } else {
            hit = true;
        }
        b -= d.max(backward_step);
    }
    assert!(hit == true || start == end);
    return (start, hit);
}

pub fn march_inflate<F: Fn(Real2)->Real>(start: Real2, end: Real2, end_r: Real, distance_func: F)->(Real2, Real) {
    let delta = end - start;
    let inv_end_r = reali(1) / end_r;
    let mut t = reali(0);
    loop {
        let point = start + delta * t;
        let d = distance_func(point);
        t = (t + d.min(reali(1)) * inv_end_r).min(reali(1));
        if d < t * end_r {
            return (point, d)
        }
        if t >= reali(1) {
            return (end, d)
        }
    }

}

pub fn find_circle_contacts<F: Fn(Real2)->Real, N: Fn(Real2)->Real2>(center: Real2, radius: Real, distance_func: F, normal_func: N)->Vec<(Real2, Real2)> {
    let mut points = Vec::new();
    let d = distance_func(center);
    let pi = Real::from(3.1415926);
    if d <= radius {
        let n = normal_func(center);
        points.push((center + n * -d, n));

        if d > reali(0) && d < radius - Real::from(0.5) {
            let num_divs = 32;
            let subsample_r = reali(8);
            for div in 0..num_divs {
                let angle = pi * reali(2 * div) / reali(num_divs);
                let offset = Real2::from((angle.cos(), angle.sin()));
                let subsample = center + offset * subsample_r;
                let (tp, d) = march_inflate(center, subsample, subsample_r, &distance_func);
                let n = normal_func(tp);
                let candidate = tp + d * -n;
                let rad = candidate - center;
                if rad.dot(rad) <= radius * radius {
                    let epsilon = reali(1);
                    if distance_func(candidate) <= epsilon {
                        points.push((candidate, n));
                    }
                }
            }

            let mut grouped_normals: Vec<_> = points.iter().map(|(_, n)| (Real2::from((reali(0), reali(0))), *n, 0)).collect();
            let compare_dist = Real::from(0.05);
            let compare_dist_sq = compare_dist * compare_dist;
            for i in 0..points.len() {
                for j in 0..points.len() {
                    if (points[i].1 - points[j].1).magnitude_squared() <= compare_dist_sq {
                        let current_dist_sq = (grouped_normals[j].0 - center).magnitude_squared();
                        let new_dist_sq = (points[i].0 - center).magnitude_squared();
                        if new_dist_sq < current_dist_sq {
                            grouped_normals[j].0 = points[i].0;
                        }
                        grouped_normals[j].2 += 1;
                        break;
                    }
                }
            }
            points = grouped_normals.into_iter().filter_map(|(p, n, c)| if c > 0 { Some((p, n)) } else { None }).collect();
        }
    }
    points
}
