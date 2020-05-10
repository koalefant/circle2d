mod sdf;
mod serialization;
mod spatial_hash;
mod world;
#[allow(unused_imports)]
use sdf::*;
#[cfg(feature = "serde_support")]
use serialization::*;
use spatial_hash::*;
pub use world::*;

/// bit mask that can be either used together with World::set_collision_shape_flags
/// to mark which bodies can collide, or mark bodies with arbitrary flags
pub type ShapeFlags = u32;

/// can be used to identify owner of the body from outside, may be redefined locally
pub type BodyOwner = u64;

/// this can be locally overriden, for example, to use fixed-point math or doubles
pub type Real = f32;
/// used to construct internal constants in a way that survives redifinition of real
fn reali(v: i32) -> Real {
    v as Real
}

/// this can be locally overriden, for example, to use fixed-point math or doubles
use num_traits::identities::Zero;
pub type Real2 = vek::Vec2<f32>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub fn hash_slot_map<
    K: slotmap::Key + std::hash::Hash,
    V: std::hash::Hash + slotmap::Slottable,
    H: std::hash::Hasher,
>(
    map: &slotmap::SlotMap<K, V>,
    h: &mut H,
) {
    map.iter().for_each(|(k, v)| {
        k.hash(h);
        v.hash(h);
    });
}

pub fn hash_secondary_map<
    K: slotmap::Key + std::hash::Hash,
    V: std::hash::Hash,
    H: std::hash::Hasher,
>(
    map: &slotmap::SecondaryMap<K, V>,
    h: &mut H,
) {
    map.iter().for_each(|(k, v)| {
        k.hash(h);
        v.hash(h);
    });
}
