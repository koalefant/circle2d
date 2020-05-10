use crate::*;
#[cfg(feature="serde_support")]
use serde_derive::{Serialize, Deserialize};

#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub struct SpatialHash{
    #[cfg_attr(feature="serde_support", serde(with="hashmap_as_pairs"))]
    pub hash: rustc_hash::FxHashMap<(i32, i32), Vec<(BodyKey, ShapeFlags)>>,
    pub map_bodies: Vec<(BodyKey, ShapeFlags)>,
    pub(crate) cell_size: Real,
    pub(crate) inv_cell_size: Real,
}


/// SpatialHash is used to quickly find out what bodies are located at specific position
impl SpatialHash {
    pub(crate) fn register(&mut self, from: Real2, to: Real2, shape: Shape, shape_flags: ShapeFlags, k: BodyKey) {
        match shape {
            Shape::Circle(r) => {
                let (start, end) = self.grid_range(from, to, r);
                for j in start.1..end.1 {
                    for i in start.0..end.0 {
                        let list = self.hash.entry((i, j)).or_insert(Vec::new());
                        list.push((k, shape_flags));
                    }
                }
            }
            Shape::Map => {
                assert!(self.map_bodies.iter().position(|p| p.0 == k).is_none());
                self.map_bodies.push((k, shape_flags));
            }
            Shape::Hollow => {
            }
        }
    }

    pub(crate) fn unregister(&mut self, from: Real2, to: Real2, shape: Shape, k: BodyKey) {
        match shape {
            Shape::Circle(r) => {
                let (start, end) = self.grid_range(from, to, r);
                for j in start.1..end.1 {
                    for i in start.0..end.0 {
                        if let Some(list) =  self.hash.get_mut(&(i, j)) {
                            list.retain(|v| v.0 != k);
                            if list.is_empty() {
                                self.hash.remove(&(i, j));
                            }
                        }
                    }
                }
            }
            Shape::Map => {
                self.map_bodies.retain(|v| v.0 != k);
            }
            Shape::Hollow => {
            }
        }
    }

    pub(crate) fn update(&mut self, from_old: Real2, to_old: Real2, from_new: Real2, to_new: Real2, shape: Shape, shape_flags: ShapeFlags, k: BodyKey) {
        match shape {
            Shape::Circle(r) => {
                let (start_old, end_old) = self.grid_range(from_old, to_old, r);
                let (start_new, end_new) = self.grid_range(from_new, to_new, r);
                if start_new != start_old || end_new != end_old {
                    self.unregister(from_old, to_old, shape, k);
                    self.register(from_new, to_new, shape, shape_flags, k);
                }
            }
            Shape::Map => {}
            Shape::Hollow => {}
        }
    }

    pub(crate) fn find_neighbours(&self, result: &mut Vec<BodyKey>, from: Real2, to: Real2, shape: Shape, mask: ShapeFlags) {
        result.clear();
        let r = shape.radius();
        let (start, end) = self.grid_range(from, to, r);
        for j in start.1..end.1 {
            for i in start.0..end.0 {
                if let Some(list) = self.hash.get(&(i, j)) {
                    result.extend(
                        list.iter()
                        .filter(|v| (v.1 & mask) != 0)
                        .map(|v| v.0)
                        );
                }
            }
        }
        result.extend(
            self.map_bodies.iter()
            .filter(|v| (v.1 & mask) != 0)
            .map(|v| v.0)
        );
        result.sort_unstable();
        result.dedup();
    }

    fn grid_range(&self, from: Real2, to: Real2, r: Real)->((i32, i32), (i32, i32)) {
        let pos_min = from - Real2::from((r, r));
        let pos_max = to + Real2::from((r, r));
        let pos_min = cell_div2(pos_min, self.cell_size, self.inv_cell_size);
        let pos_max = cell_div2(pos_max, self.cell_size, self.inv_cell_size);
        let start = pos_min.map(|v| v.floor());
        let start = (start.x as i32, start.y as i32);
        let end = pos_max.map(|v| v.floor());
        let end = (end.x as i32, end.y as i32);
        let (start, end) = (
            (start.0.min(end.0), start.1.min(end.1)),
            (start.0.max(end.0) + 1, start.1.max(end.1) + 1)
        );
        (start, end)
    }
}

fn cell_div(x: Real, v: Real, inv_v: Real)->Real {
    if x >= reali(0) {
        x * inv_v
    } else {
        (x - v + reali(1)) * inv_v
    }
}

fn cell_div2(x: Real2, v: Real, inv_v: Real)->Real2 {
    Real2::from((cell_div(x.x, v, inv_v), cell_div(x.y, v, inv_v)))
}
