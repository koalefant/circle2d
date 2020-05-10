use crate::*;
#[cfg(feature="serde_support")]
use serde_derive::{Serialize, Deserialize};
#[cfg(feature="serde_support")]
use crate::serialization::{hashmap_as_pairs, btreemap_as_pairs};
use rustc_hash::FxHashMap;
use std::collections::{BTreeSet, BTreeMap};
#[cfg(feature="hash_support")]
use std::hash::{Hash, Hasher};
use crate::sdf::{sd_circle, march_circle, find_circle_contacts};

slotmap::new_key_type! {
    pub struct BodyKey;
}

#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct World {
    bodies: slotmap::SlotMap<BodyKey, Body>,
    /// bodies that need to be simulated
    pub active_bodies: BTreeSet<BodyKey>,

    sleeping_islands: BTreeSet<BodyKey>,
    island_roots: slotmap::SecondaryMap<BodyKey, BodyKey>,
    island_nexts: slotmap::SecondaryMap<BodyKey, BodyKey>,
    island_prevs: slotmap::SecondaryMap<BodyKey, BodyKey>,
    idle_times: slotmap::SecondaryMap<BodyKey, Real>,

    #[cfg_attr(feature="serde_support", serde(with="btreemap_as_pairs"))]
    pub persistent_contacts: BTreeMap<(BodyKey, BodyKey), PersistentContact>,
    #[cfg_attr(feature="serde_support", serde(with="btreemap_as_pairs"))]
    pub sleeping_contacts: BTreeMap<(BodyKey, BodyKey), PersistentContact>,
    pub spatial_hash: SpatialHash,

    pub bounce_velocity: Real,
    pub max_velocity: Real,
    pub max_angular_velocity: Real,
    pub collision_shape_flags: ShapeFlags,
}

#[cfg(feature="hash_support")]
impl Hash for World {
    fn hash<H: Hasher>(&self, h: &mut H) {
        crate::hash_slot_map(&self.bodies, h);
        self.active_bodies.hash(h);
        self.sleeping_islands.hash(h);
        crate::hash_secondary_map(&self.island_roots, h);
        crate::hash_secondary_map(&self.island_nexts, h);
        crate::hash_secondary_map(&self.island_prevs, h);
        crate::hash_secondary_map(&self.idle_times, h);
        self.persistent_contacts.hash(h);
        self.sleeping_contacts.hash(h);
        // self.spatial_hash.hash(h);
        self.bounce_velocity.hash(h);
        self.collision_shape_flags.hash(h);
    }
}


#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Hash, PartialEq)]
pub enum BodyType {
    Static,
    Dynamic
}

impl Default for BodyType {
    fn default()->Self { BodyType::Static }
}

#[cfg_attr(feature="hash_support", derive(Hash))]
#[derive(Copy, Clone)]
pub struct BodyDef {
    pub typ: BodyType,
    pub pos: Real2,
    pub shape: Shape,
    pub shape_flags: ShapeFlags,
    pub mass: Option<Real>,
    pub inertia: Option<Real>,
    pub friction: Real,
    pub restitution: Real,
    pub max_penetration: Real
}

impl Default for BodyDef {
    fn default()->Self {
        Self{
            typ: BodyType::Dynamic,
            pos: Real2::zero(),
            shape: Shape::default(),
            shape_flags: 0xffffffff,
            mass: Some(reali(1)),
            inertia: Some(reali(1)),
            friction: Real::from(0.5),
            restitution: Real::from(0.8),
            max_penetration: reali(8),
        }
    }
}

impl Default for Shape {
    fn default()->Self { Shape::Hollow }
}

#[cfg_attr(feature="hash_support", derive(Hash))]
#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[derive(Copy, Clone)]
pub struct Body {
    pos: Real2,
    rotation: Real,

    velocity: Real2,
    angular_velocity: Real,

    force: Real2,
    torque: Real,

    typ: BodyType,
    inv_mass: Real,
    inv_inertia: Real,
    friction: Real,
    restitution: Real,
    max_penetration: Real,

    shape: Shape,
    shape_flags: ShapeFlags,
    owner: Option<BodyOwner>
}

#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[cfg_attr(feature="hash_support", derive(Hash))]
#[derive(Copy, Clone)]
pub struct ContactPoint {
    pub position: Real2,
    pub normal: Real2,
    r1: Real2,
    r2: Real2,
    separation: Real,
    p_n: Real,
    p_t: Real,
    p_n_b: Real,
    mass_n: Real,
    mass_t: Real,
    bias: Real,
    bounce: Real,
}

#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[cfg_attr(feature="hash_support", derive(Hash))]
#[derive(Clone)]
pub struct PersistentContact {
    pub points: Vec<ContactPoint>,
    friction: Real,
    restitution: Real
}

#[cfg_attr(feature="serde_support", derive(Serialize, Deserialize))]
#[cfg_attr(feature="hash_support", derive(Hash))]
#[derive(Copy, Clone, PartialEq)]
pub enum Shape {
    Hollow,
    Circle(Real),
    Map,
}

impl PersistentContact {
    pub fn total_impulse(&self)->Real {
        let mut result = reali(0);
        for point in self.points.iter() {
            result += point.p_n
        }
        result
    }
}

impl Shape {
    pub fn radius(&self)->Real {
        match self {
            Shape::Circle(r) => *r,
            // FIXME: this should be defined from outside
            Shape::Map => reali(4096),
            Shape::Hollow => reali(0),
        }
    }

    pub fn test_overlap<D: Fn(Real2)->Real>(
            a_shape: Shape,
            a_pos: Real2,
            b_shape: Shape,
            b_pos: Real2,
            distance_func: &D)->bool {
        match (a_shape, b_shape, a_pos, b_pos) {
            (Shape::Circle(a_radius), Shape::Circle(b_radius), a_pos, b_pos) => {
                let dist_square = (a_pos - b_pos).magnitude_squared();
                let radius = a_radius + b_radius;
                let radius_square = radius * radius;
                dist_square < radius_square
            },
            // circle-map coillision
            (Shape::Circle(a_radius), Shape::Map, a_pos, _) | (Shape::Map, Shape::Circle(a_radius), _, a_pos) => {
                let d = distance_func(a_pos);
                d < a_radius
            },
            (Shape::Hollow, _, _, _) | (_, Shape::Hollow, _, _) => false,
            _ => { panic!("missing collision pair") }
        }
    }

    pub fn test_contacts<D: Fn(Real2)->Real, N: Fn(Real2)->Real2>(points: &mut Vec<ContactPoint>,
            a_shape: Shape,
            a_pos: Real2,
            b_shape: Shape,
            b_pos: Real2,
            map_distance: &D,
            map_normal: &N,
            ) {
        match (a_shape, b_shape, a_pos, b_pos) {
            (Shape::Circle(a_radius), Shape::Circle(b_radius), a_pos, b_pos) => {
                let dist_square = (a_pos - b_pos).magnitude_squared();
                let radius = a_radius + b_radius;
                let radius_square = radius * radius;
                if dist_square < radius_square {
                    let dist = dist_square.sqrt();
                    let separation = radius - dist;
                    let pos_delta = b_pos - a_pos;
                    let pos_delta_len = pos_delta.magnitude();
                    let normal = if pos_delta_len > (0.0001) { pos_delta / pos_delta_len } else { Real2::from((reali(0), reali(-1))) };
                    let position = (0.5) * (a_pos + normal * a_radius + b_pos + -normal * b_radius);
                    points.push(ContactPoint{
                        position,
                        normal,
                        separation,
                        r1: Real2::zero(),
                        r2: Real2::zero(),
                        p_n: reali(0),
                        p_n_b: reali(0),
                        p_t: reali(0),
                        mass_n: reali(0),
                        mass_t: reali(0),
                        bias: reali(0),
                        bounce: reali(0),
                    });
                } 
            },
            // circle-map coillision
            (Shape::Circle(a_radius), Shape::Map, a_pos, _) | (Shape::Map, Shape::Circle(a_radius), _, a_pos) => {
                let contacts = find_circle_contacts(a_pos, a_radius, map_distance, map_normal);
                for contact in contacts.into_iter() {
                    let separation = (contact.0 - a_pos).magnitude() - a_radius;
                    let normal = contact.1;
                    let position = contact.0;
                    points.push(ContactPoint{
                        position,
                        normal,
                        separation,
                        r1: Real2::zero(),
                        r2: Real2::zero(),
                        p_n: reali(0),
                        p_n_b: reali(0),
                        p_t: reali(0),
                        mass_n: reali(0),
                        mass_t: reali(0),
                        bias: reali(0),
                        bounce: reali(0),
                    });
                }
            },
            (Shape::Hollow, _, _, _) | (_, Shape::Hollow, _, _) => {},
            _ => { panic!("missing collision pair") }
        }
    }
}

fn cross(a: Real2, b: Real2)->Real { a.x * b.y - a.y * b.x }
fn cross_scalar(s: Real, b: Real2)->Real2 { Real2::from((-s * b.y, s * b.x)) }

impl Body {
    fn kinetic_energy(&self)->Real {
        let mut r = Real::zero();
        if self.inv_mass != reali(0) {
            r += self.velocity.dot(self.velocity) / self.inv_mass;
        }
        if self.inv_inertia != reali(0) {
            r += self.angular_velocity * self.angular_velocity / self.inv_inertia;
        }
        r
    }
}

impl World {
    pub fn new()->Self {
        Self {
            bodies: slotmap::SlotMap::with_capacity_and_key(256),
            active_bodies: BTreeSet::new(),
            persistent_contacts: BTreeMap::new(),
            sleeping_contacts: BTreeMap::new(),
            idle_times: slotmap::SecondaryMap::new(),
            island_roots: slotmap::SecondaryMap::new(),
            island_nexts: slotmap::SecondaryMap::new(),
            island_prevs: slotmap::SecondaryMap::new(),
            sleeping_islands: BTreeSet::new(),
            spatial_hash: SpatialHash{
                hash: FxHashMap::default(),
                map_bodies: Vec::new(),
                cell_size: reali(64),
                inv_cell_size: reali(1) / reali(64),
            },
            bounce_velocity: reali(200),
            max_velocity: reali(4000),
            max_angular_velocity: reali(50),
            collision_shape_flags: 0xffffffff
        }
    }

    /// defines which bits of Body::shape_flags are used for inter-body collision checks. Default
    /// is to use all bits.
    pub fn set_collision_shape_flags(&mut self, collision_shape_flags: ShapeFlags) {
        self.collision_shape_flags = collision_shape_flags;
    }
    pub fn collision_shape_flags(&self)->ShapeFlags { self.collision_shape_flags }

    /// how long does it take for body to be inactive before it is put ot sleep
    pub fn sleep_time_threshold(&self)->Real { Real::from(0.3) }
    /// how slow body has to be put into sleep
    pub fn sleep_velocity_threshold(&self)->Real { reali(25) }

    /// add body to the simulation
    pub fn add_body(&mut self, def: BodyDef)->BodyKey {
        let k = self.bodies.insert(Body{
            typ: def.typ,
            pos: def.pos,
            shape: def.shape,
            friction: def.friction,
            restitution: def.restitution,
            max_penetration: def.max_penetration,
            shape_flags: def.shape_flags,
            rotation: reali(0),
            velocity: Real2::zero(),
            angular_velocity: reali(0),
            force: Real2::zero(),
            torque: reali(0),
            owner: None,
            inv_mass: match def.mass { Some(m) => reali(1) / m, _ => reali(0) },
            inv_inertia: match def.inertia { Some(i) => reali(1) / i, _ => reali(0) },
        });
        if def.typ == BodyType::Dynamic {
            self.active_bodies.insert(k);
        }
        self.spatial_hash.register(def.pos, def.pos, def.shape, def.shape_flags, k);
        k
    }

    pub fn remove_body(&mut self, k: BodyKey) {
        if self.bodies[k].typ == BodyType::Dynamic {
            self.body_activate(k);
        } else {
            // look for possible sleeping contacts of dynamic bodies
            // and activate them
            let mut bodies_to_activate = Vec::new();
            for (c_k, _) in &self.sleeping_contacts {
                if k == c_k.0 {
                    bodies_to_activate.push(c_k.1);
                } else if k == c_k.1 {
                    bodies_to_activate.push(c_k.0);
                }
            }
            for neighbour in bodies_to_activate {
                assert!(self.bodies[neighbour].typ == BodyType::Dynamic);
                self.body_activate(neighbour);
            }
        }
        // verify that there is no sleeping contact left
        #[cfg(debug_assertions)]
        {
            for (c_k, _) in &self.sleeping_contacts {
                assert!(k != c_k.0 && k != c_k.1)
            }
            for (_, n) in &self.island_nexts {
                if *n == k {
                    eprintln!("body {:?} of {:?} is still next-connected with removed body {:?} of {:?}",
                        *n, self.bodies[*n].owner, k, self.bodies[k].owner);
                }
                assert!(*n != k);
            }
            for (_, n) in &self.island_prevs {
                if *n == k {
                    eprintln!("body {:?} of {:?} is still prev-connected with removed body {:?} of {:?}",
                        *n, self.bodies[*n].owner, k, self.bodies[k].owner);
                }
                assert!(*n != k);
            }
            assert!(!self.island_roots.contains_key(k));
            assert!(self.active_bodies.contains(&k) == (self.bodies[k].typ == BodyType::Dynamic));
        }

        let a = self.bodies.get(k).expect("remove_body");
        self.spatial_hash.unregister(a.pos, a.pos, a.shape, k);

        self.active_bodies.remove(&k);
        self.bodies.remove(k);

        let mut contact_remove_list = Vec::new();
        for (c_k, _) in &self.persistent_contacts {
            if k == c_k.0 || k == c_k.1 {
                contact_remove_list.push(*c_k);
            }
        }
        for k in &contact_remove_list {
            self.persistent_contacts.remove(&k);
        }
    }

    pub fn body_is_valid(&self, k: BodyKey)->bool {
        match self.bodies.get(k) { Some(_) => true, _ => false }
    }
    pub fn body_set_owner(&mut self, k: BodyKey, owner: Option<BodyOwner>) { self.bodies.get_mut(k).expect("body_set_owner").owner = owner; }
    pub fn body_set_friction(&mut self, k: BodyKey, friction: Real) { 
        let body = self.bodies.get_mut(k).expect("body_set_friction");
        if body.friction != friction {
            body.friction = friction;
            self.body_activate(k);
        }
    }
    pub fn body_set_velocity(&mut self, k: BodyKey, v: Real2) {
        let body = self.bodies.get_mut(k).expect("body_set_velocity");
        if body.velocity != v {
            body.velocity = v;
            self.body_activate(k);
        }
    }
    pub fn body_set_angular_velocity(&mut self, k: BodyKey, v: Real, activate: bool) { 
        let body = self.bodies.get_mut(k).expect("body_set_angular_velocity");
        if body.angular_velocity != v {
            body.angular_velocity = v;
            if activate {
                self.body_activate(k);
            }
        }
    }
    pub fn body_set_inertia(&mut self, k: BodyKey, inertia: Option<Real>) {
        let body = self.bodies.get_mut(k).expect("body_set_inertia");
        let inv_inertia = match inertia { Some(v) => reali(1) / v, _ => reali(0) };
        if body.inv_inertia != inv_inertia {
            body.inv_inertia = inv_inertia;
            self.body_activate(k);
        }
    }
    pub fn body_set_rotation(&mut self, k: BodyKey, r: Real, activate: bool) {
        let body = self.bodies.get_mut(k).expect("body_set_rotation");
        if body.rotation != r {
            body.rotation = r;
            if activate {
                self.body_activate(k);
            }
        }
    }

    pub fn body_apply_force(&mut self, k: BodyKey, f: Real2) {
        let body = self.bodies.get_mut(k).expect("body_set_force");
        body.force += f;
        self.body_activate(k);
    }

    pub fn body_set_force(&mut self, k: BodyKey, f: Real2) {
        let body = self.bodies.get_mut(k).expect("body_set_force");
        if body.force != f {
            body.force = f;
            self.body_activate(k);
        }
    }

    pub fn body_set_force_by_acceleration(&mut self, k: BodyKey, f: Real2) {
        let body = self.bodies.get_mut(k).expect("body_set_force");
        let force = f / body.inv_mass;
        if body.force != force {
            body.force = force;
            self.body_activate(k);
        }
    }
    pub fn body_set_position(&mut self, k: BodyKey, pos: Real2) {
        let body = self.bodies.get_mut(k).expect("body_set_position");
        if body.pos != pos {
            self.spatial_hash.unregister(body.pos, body.pos, body.shape, k);
            body.pos = pos;
            self.spatial_hash.register(pos, pos, body.shape, body.shape_flags, k);
            self.body_activate(k);
        }
    }
    pub fn body_set_shape(&mut self, k: BodyKey, shape: Shape) {
        let body = self.bodies.get_mut(k).expect("body_set_shape");
        if body.shape != shape {
            self.spatial_hash.unregister(body.pos, body.pos, body.shape, k);
            body.shape = shape;
            self.spatial_hash.register(body.pos, body.pos, body.shape, body.shape_flags, k);
            self.body_activate(k);
        }
    }
    pub fn body_set_shape_flags(&mut self, k: BodyKey, flags: ShapeFlags) {
        let body = self.bodies.get_mut(k).expect("body_set_shape");
        if body.shape_flags != flags {
            self.spatial_hash.unregister(body.pos, body.pos, body.shape, k);
            body.shape_flags = flags;
            self.spatial_hash.register(body.pos, body.pos, body.shape, body.shape_flags, k);
            self.body_activate(k);
        }
    }
    pub fn body_shape_flags(&self, k: BodyKey)->ShapeFlags { self.bodies[k].shape_flags }
    pub fn body_owner(&self, k: BodyKey)->Option<BodyOwner>    { self.bodies[k].owner }
    pub fn body_type(&self, k: BodyKey)->BodyType              { self.bodies[k].typ }
    pub fn body_shape(&self, k: BodyKey)->Shape                { self.bodies[k].shape }
    pub fn body_position(&self, k: BodyKey)->Real2            { self.bodies[k].pos }
    pub fn body_idle_time(&self, k: BodyKey)->Real            { self.idle_times.get(k).copied().unwrap_or(reali(-1)) }
    pub fn body_velocity(&self, k: BodyKey)->Real2            { self.bodies[k].velocity }
    pub fn body_angular_velocity(&self, k: BodyKey)->Real     { self.bodies[k].angular_velocity }
    pub fn body_rotation(&self, k: BodyKey)->Real             { self.bodies[k].rotation }
    pub fn body_inv_mass(&self, k: BodyKey)->Real             { self.bodies[k].inv_mass }
    pub fn body_kinetic_energy(&self, k: BodyKey)->Real {
        let body: &Body = &self.bodies[k];
        body.kinetic_energy()
    }

    pub fn body_is_sleeping(&self, k: BodyKey)->bool {
        assert!(self.bodies[k].typ == BodyType::Dynamic);
        let is_sleeping = self.island_roots.get(k).is_some();
        if is_sleeping {
            assert!(self.active_bodies.contains(&k) == false);
        }
        is_sleeping
    }

    pub fn visit_sleeping_island<F: FnMut(BodyKey, &mut Self)>(&mut self, root: BodyKey, func: &mut F) {
        let mut cur = root;
        loop {
            let next = self.island_nexts.get(cur).copied();
            if self.island_roots.get(cur).copied() != Some(root) {
                eprintln!("member of island {:?} ({:?}) has incorrect root {:?} instead of expected {:?}", cur, self.bodies[cur].owner, self.island_roots.get(cur), root);
            }
            assert!(self.island_roots.get(cur).copied() == Some(root));
            func(cur, self);
            cur = match next {
                Some(k) => k,
                None => break
            };
        }
    }
    fn body_deactivate(&mut self, k: BodyKey)->Vec<(BodyKey, BodyKey, Real2)> {
        let body: &mut Body = self.bodies.get_mut(k).expect("body_sleep");
        assert!(body.typ == BodyType::Dynamic);
        // is this valid?
        body.angular_velocity = reali(0);
        body.velocity = Real2::zero();

        self.active_bodies.remove(&k);

        let contacts_to_sleep: Vec<(BodyKey, BodyKey, Real2)> = self.persistent_contacts.iter().filter_map(|((k1, k2), v)| if *k1 == k || *k2 == k { Some((*k1, *k2, v.points[0].position)) } else { None }).collect();
        for &(k0, k1, _) in contacts_to_sleep.iter() {
            let contact = self.persistent_contacts.remove(&(k0, k1)).unwrap();
            self.sleeping_contacts.insert((k0, k1), contact);
        }
        contacts_to_sleep
    }

    pub fn body_put_to_sleep(&mut self, k: BodyKey) {
        Self::unlink_from_island(
            &mut self.island_roots,
            &mut self.island_nexts,
            &mut self.island_prevs, k);
        self.island_roots.insert(k, k);
        self.body_deactivate(k);
    }

    pub fn body_activate(&mut self, k: BodyKey) {
        self.idle_times.insert(k, reali(0));
        assert!(self.bodies[k].typ == BodyType::Dynamic);
        let root = self.island_roots.get(k).copied();
        if let Some(root) = root {
            if self.body_is_sleeping(root) {
                self.visit_sleeping_island(root, &mut |k, w| {
                    assert!(w.bodies[k].typ == BodyType::Dynamic);
                    w.active_bodies.insert(k);
                    Self::unlink_from_island(
                        &mut w.island_roots,
                        &mut w.island_nexts,
                        &mut w.island_prevs, k);
                    w.idle_times.insert(k, reali(0));
                    let contacts_to_awake: Vec<_> = w.sleeping_contacts.keys().filter(|&(k1, k2)| *k1 == k || *k2 == k).copied().collect();
                    for &c in contacts_to_awake.iter() {
                        assert!(w.bodies.get(c.0).is_some());
                        assert!(w.bodies.get(c.1).is_some());
                        w.persistent_contacts.insert(c, w.sleeping_contacts.remove(&c).unwrap());
                    }
                });
            }
        }
        for (c_k, _) in &self.sleeping_contacts {
            if k == c_k.0 || k == c_k.1 {
                let other = if k == c_k.0 { c_k.1 } else { c_k.0 };
                eprintln!("body {:?} ({:?}) is still in sleeping contact with {:?} ({:?}", k, self.bodies[k].owner, other, self.bodies[other].owner)
            }
            assert!(k != c_k.0 && k != c_k.1)
        }
    }

    pub fn bodies<'a>(&'a self)->impl Iterator<Item = BodyKey> + 'a {
        self.bodies.keys()
    }

    pub fn body_total_contact_impulse(&self, k: BodyKey)->Real {
        // find all contacts
        self.persistent_contacts
            .iter().filter(|(p, _)| {
                p.0 == k || p.1 == k
            })
            .map(|(_, c)| {
                c.total_impulse()
            })
            .fold(reali(0), |a, b| a + b)
    }

    fn unlink_from_island(roots: &mut slotmap::SecondaryMap<BodyKey, BodyKey>, 
                          nexts: &mut slotmap::SecondaryMap<BodyKey, BodyKey>,
                          prevs: &mut slotmap::SecondaryMap<BodyKey, BodyKey>, 
                          body: BodyKey) {
        roots.remove(body);
        let body_next = nexts.remove(body);
        if let Some(prev) = prevs.remove(body) {
            if let Some(body_next) = body_next {
                nexts.insert(prev, body_next);
                prevs.insert(body_next, prev);
            } else {
                nexts.remove(prev);
            }
        }
    }

    pub fn flood_fill_island_r(
        roots: &mut slotmap::SecondaryMap<BodyKey, BodyKey>, 
        nexts: &mut slotmap::SecondaryMap<BodyKey, BodyKey>,
        prevs: &mut slotmap::SecondaryMap<BodyKey, BodyKey>,
        contacts: &BTreeMap<(BodyKey, BodyKey), PersistentContact>,
        root: BodyKey,
        body: BodyKey,
        bodies: &slotmap::SlotMap<BodyKey, Body>,
        ) {
        assert!(bodies[body].typ == BodyType::Dynamic);
        if roots.get(body).is_some() {
            return;
        }
        roots.insert(body, root);
        if body != root {
            match nexts.insert(root, body) {
                Some(root_next) => {
                    nexts.insert(body, root_next);
                    prevs.insert(root_next, body);
                }
                None => {
                    nexts.remove(body);
                }
            };
            prevs.insert(body, root);
        }

        // OPTIMIZE: the list of persistent_contacts can be precomputed to reduce complexity
        for &c in contacts.keys() {
            if c.0 == body && bodies[c.1].typ == BodyType::Dynamic {
                Self::flood_fill_island_r(roots, nexts, prevs, contacts, root, c.1, bodies);
            }
            if c.1 == body && bodies[c.0].typ == BodyType::Dynamic  {
                Self::flood_fill_island_r(roots, nexts, prevs, contacts, root, c.0, bodies);
            }
        }
    }

    fn is_island_active(k: BodyKey,
        nexts: &slotmap::SecondaryMap<BodyKey, BodyKey>,
        idle_times: &slotmap::SecondaryMap<BodyKey, Real>,
        sleep_time_threshold: Real)->bool {
        let mut cur = k;
        loop {
            let idle_time = idle_times.get(cur).copied().unwrap_or(reali(0));
            if idle_time < sleep_time_threshold {
                return true;
            }
            cur = match nexts.get(cur) {
                Some(next) => *next,
                None => break
            }
        }
        false
    }

    #[inline(never)]
    pub fn simulate_physics<D: Fn(Real2)->Real, N: Fn(Real2)->Real2>(&mut self, map_distance: D, map_normal: N, dt: Real, started_contacts: &mut Vec<(BodyKey, BodyKey)>, finished_contacts: &mut Vec<(BodyKey, BodyKey, Real2)>) {
        if dt == reali(0) {
            return;
        }
        let inv_dt = reali(1) / dt;

        let mut neighbours = Vec::new();
        let mut new_contacts = BTreeMap::new();
        for &a_k in self.active_bodies.iter() {
            let a = &self.bodies[a_k];
            self.spatial_hash.find_neighbours(&mut neighbours, a.pos, a.pos, a.shape, a.shape_flags & self.collision_shape_flags);
            for &b_k in &neighbours {
                if b_k == a_k {
                    continue;
                }
                let (a_k, b_k) = (a_k.min(b_k), a_k.max(b_k));
                let key = (a_k, b_k);
                let a = &self.bodies[a_k];
                let b = &self.bodies[b_k];
                let mut points = Vec::new();
                Shape::test_contacts(&mut points, a.shape, a.pos, b.shape, b.pos, &map_distance, &map_normal);
                if !points.is_empty() {
                    let friction = (a.friction * b.friction).sqrt();
                    let restitution = a.restitution.max(b.restitution);
                    let old_contact = self.persistent_contacts.get_mut(&key);
                    if old_contact.is_none() {
                        started_contacts.push(key);
                    }
                    // only reuse old contacts for dynamic bodies
                    let old_contact = if a.typ == BodyType::Dynamic && b.typ == BodyType::Dynamic {
                        old_contact
                    } else {
                        None
                    };
                    match old_contact {
                        Some(old_contact)=> {
                            for point in points.iter_mut() {
                                let mut min_normal_diff = Real::MAX;
                                let mut best_p = None;
                                for (i_p, p) in old_contact.points.iter().enumerate() {
                                    let diff = (point.normal - p.normal).magnitude();
                                    if diff < (0.3) && diff < min_normal_diff {
                                        min_normal_diff = diff;
                                        best_p = Some(i_p);
                                    }
                                }
                                if let Some(best_p) = best_p {
                                    let old_point = old_contact.points[best_p];
                                    point.p_n = old_point.p_n;
                                    point.p_t = old_point.p_t;
                                    point.p_n_b = old_point.p_n_b;
                                    old_contact.points.remove(best_p);
                                }
                            }
                        }
                        None => {}
                    }
                    new_contacts.insert(key, PersistentContact {
                        points,
                        friction,
                        restitution
                    });
                }
            }
        }
        for (key, contact) in self.persistent_contacts.iter() {
            if !new_contacts.contains_key(key) {
                finished_contacts.push((key.0, key.1, contact.points[0].position));
            }
        }
        self.persistent_contacts = new_contacts;

        // detect activity islands and put bodies to sleep
        {
            // reset idle times
            let vel_threshold = self.sleep_velocity_threshold();
            for &k in self.active_bodies.iter() {
                let body = &self.bodies[k];

                if body.inv_mass != reali(0) {
                    let energy_threshold = vel_threshold * vel_threshold / body.inv_mass;
                    if body.kinetic_energy() > energy_threshold {
                        self.idle_times.insert(k, reali(0));
                    } else {
                        let old_value = self.idle_times.get(k).copied().unwrap_or(reali(0));
                        let new_value = old_value + dt;
                        self.idle_times.insert(k, new_value);
                        //println!("idle {} {}->{}", self.bodies.keys().position(|v| v == k).unwrap(), old_value, new_value);
                    }
                }
            }
            
            // awake sleeping bodies that got contacts
            let mut bodies_to_awake = Vec::new();
            for &k in self.persistent_contacts.keys() {
                if self.bodies[k.0].typ == BodyType::Dynamic && self.body_is_sleeping(k.0) {
                    bodies_to_awake.push(k.0);
                }
                if self.bodies[k.1].typ == BodyType::Dynamic && self.body_is_sleeping(k.1) {
                    bodies_to_awake.push(k.1);
                }
            }
            bodies_to_awake.sort_unstable();
            bodies_to_awake.dedup();
            for &k in bodies_to_awake.iter() {
                self.body_activate(k);
            }

            for &k in self.active_bodies.iter() {
                Self::unlink_from_island(&mut self.island_roots,
                                         &mut self.island_nexts,
                                         &mut self.island_prevs, k);
            }

            let mut islands_to_deactivate = Vec::new();
            let sleep_time_threshold = self.sleep_time_threshold();
            let mut active_bodies: Vec<_> = self.active_bodies.iter().copied().collect();
            while let Some(k) = active_bodies.pop() {
                if self.island_roots.get(k).is_none() {
                    Self::flood_fill_island_r(&mut self.island_roots, &mut self.island_nexts, &mut self.island_prevs, &self.persistent_contacts, k, k, &self.bodies);

                    if !Self::is_island_active(k, &self.island_nexts, &self.idle_times, sleep_time_threshold) {
                        // deactivate complete island
                        islands_to_deactivate.push(k);

                        // do not iterate over removed bodies
                        self.visit_sleeping_island(k, &mut |cur, _| {
                            active_bodies.retain(|b| *b != cur);
                        });
                        continue;
                    }
                }

                Self::unlink_from_island(&mut self.island_roots,
                                         &mut self.island_nexts,
                                         &mut self.island_prevs, k);
            }

            for k in islands_to_deactivate.into_iter() {
                self.sleeping_islands.insert(k);
                self.visit_sleeping_island(k, &mut |cur, w| {
                    assert!(w.island_roots.get(cur).is_some());
                    let sleeping_contacts = w.body_deactivate(cur);
                    finished_contacts.extend_from_slice(&sleeping_contacts);
                    assert!(w.body_is_sleeping(cur));
                });
            }
        }

        // update velocities
        for &k in self.active_bodies.iter() {
            let a = self.bodies.get_mut(k).expect("missing k");
            a.velocity = a.velocity + dt * a.force * a.inv_mass;
            a.angular_velocity = a.angular_velocity + dt * a.inv_inertia * a.torque;
            // clamp velocity
            let velocity_mag = a.velocity.magnitude();
            if velocity_mag > self.max_velocity {
                a.velocity = a.velocity * self.max_velocity / velocity_mag;
            }
            // clamp angular velocity
            a.angular_velocity = a.angular_velocity.max(-self.max_angular_velocity).min(self.max_angular_velocity);
        }

        // apply normal and friction impulses
        for (key, contact) in self.persistent_contacts.iter_mut() {
            let (a_k, b_k) = *key;
            let a = self.bodies.get(a_k).expect("a_k");
            let b = self.bodies.get(b_k).expect("b_k");
            let mut a_velocity = a.velocity;
            let mut b_velocity = b.velocity;
            let mut a_angular_velocity = a.angular_velocity;
            let mut b_angular_velocity = b.angular_velocity;
            for point in contact.points.iter_mut() {
                let r1 = point.position - a.pos;
                let r2 = point.position - b.pos;
                let rn1 = r1.dot(point.normal);
                let rn2 = r2.dot(point.normal);
                let k_normal = a.inv_mass + b.inv_mass +
                    (r1.dot(r1) - rn1 * rn1) * a.inv_inertia + 
                    (r2.dot(r2) - rn2 * rn2) * b.inv_inertia;
                assert!(k_normal.abs() > reali(0));
                point.mass_n = reali(1) / k_normal;
                //assert!(point.mass_n.is_finite());

                let tangent = Real2::from((point.normal.y, -point.normal.x));
                let rt1 = r1.dot(tangent);
                let rt2 = r2.dot(tangent);
                let k_tangent = a.inv_mass + b.inv_mass +
                    (r1.dot(r1) - rt1 * rt1) * a.inv_inertia + 
                    (r2.dot(r2) - rt2 * rt2) * b.inv_inertia;
                assert!(k_tangent.abs() > reali(0));
                point.mass_t = reali(1) / k_tangent;
                //assert!(point.mass_t.is_finite());
                let k_bias_factor = Real::from(0.2);
                let k_allowed_penetration = Real::from(0.3);
                point.bias = -k_bias_factor * inv_dt * (point.separation + k_allowed_penetration).min(reali(0));

                let relative_velocity = b_velocity + cross_scalar(b_angular_velocity, point.r2) - 
                                        a_velocity - cross_scalar(a_angular_velocity, point.r1);
                let relative_velocity_n = relative_velocity.dot(point.normal);
                point.bounce = if relative_velocity_n < -self.bounce_velocity {
                    -relative_velocity_n * contact.restitution
                } else {
                    reali(0)
                };

                { // accumulate impulses
                    let p = point.p_n * point.normal + point.p_t * tangent;
                    a_velocity -= p * a.inv_mass;
                    b_velocity += p * b.inv_mass;
                    a_angular_velocity -= cross(r1, p) * a.inv_inertia;
                    b_angular_velocity += cross(r2, p) * b.inv_inertia;
                    //assert!(a_velocity.x.is_finite() && a_velocity.y.is_finite());
                    //assert!(b_velocity.x.is_finite() && b_velocity.y.is_finite());
                }
            }

            // write
            let a = self.bodies.get_mut(a_k).expect("a_k");
            if a.inv_mass != reali(0) {
                a.velocity = a_velocity;
            }
            if a.inv_inertia != reali(0) {
                a.angular_velocity = a_angular_velocity;
            }
            let b = self.bodies.get_mut(b_k).expect("b_k");
            if b.inv_mass != reali(0) {
                b.velocity = b_velocity;
            }
            if b.inv_inertia != reali(0) {
                b.angular_velocity = b_angular_velocity;
            }
        }

        for _iteration in 0..100 {
            for (key, contact) in self.persistent_contacts.iter_mut() {
                let (a_k, b_k) = *key;
                let a = self.bodies.get(a_k).expect("a_k");
                let b = self.bodies.get(b_k).expect("b_k");

                let mut a_velocity = a.velocity;
                let mut a_angular_velocity = a.angular_velocity;
                let mut b_velocity = b.velocity;
                let mut b_angular_velocity = b.angular_velocity;
                // apply impulses
                for point in contact.points.iter_mut() {
                    point.r1 = point.position - a.pos;
                    point.r2 = point.position - b.pos;

                    // relative velocity at point
                    let relative_velocity = b_velocity + cross_scalar(b_angular_velocity, point.r2) - 
                        a_velocity - cross_scalar(a_angular_velocity, point.r1);

                    let normal_impulse = relative_velocity.dot(point.normal);
                    let mut dpn = point.mass_n * (-normal_impulse + point.bias + point.bounce);
                    // accumulate impulses
                    {
                        let pn0 = point.p_n;
                        point.p_n = (pn0 + dpn).max(reali(0));
                        dpn = point.p_n - pn0;
                    } /*
                         else {
                         dpn = dpn.max(0.0);
                         }
                         */
                    if dpn != reali(0) || relative_velocity != Real2::zero() {
                        //println!("dpn: {} relative velocity: {},{}", dpn, relative_velocity.x, relative_velocity.y);
                    }

                    // apply point normal impulse
                    let pn = dpn * point.normal;
                    a_velocity -= pn * a.inv_mass;
                    b_velocity += pn * b.inv_mass;
                    a_angular_velocity -= cross(point.r1, pn) * a.inv_inertia;
                    b_angular_velocity += cross(point.r2, pn) * b.inv_inertia;

                    let relative_velocity = b_velocity + cross_scalar(b_angular_velocity, point.r2) - 
                        a_velocity - cross_scalar(a_angular_velocity, point.r1);
                    let tangent = Real2::from((point.normal.y, -point.normal.x));
                    let tangent_impulse = relative_velocity.dot(tangent);
                    let mut dpt = point.mass_t * -tangent_impulse;

                    // accumulate impulses 
                    {
                        let max_pt = contact.friction * point.p_n;

                        let old_tangent_impulse = point.p_t;
                        point.p_t = (old_tangent_impulse + dpt).max(-max_pt).min(max_pt);
                        dpt = point.p_t - old_tangent_impulse;
                    }
                    /* else {
                       let max_pt = friction * dPn
                       dpt = dpt.max(-max_pt).min(max_pt);
                       } */

                    // apply point tangent impulse
                    let pt = dpt * tangent;
                    a_velocity -= pt * a.inv_mass;
                    b_velocity += pt * b.inv_mass;
                    a_angular_velocity -= cross(point.r1, pt) * a.inv_inertia;
                    b_angular_velocity += cross(point.r2, pt) * b.inv_inertia;
                    //assert!(a_velocity.x.is_finite() && a_velocity.y.is_finite());
                    //assert!(b_velocity.x.is_finite() && b_velocity.y.is_finite());
                }
                let a = self.bodies.get_mut(a_k).expect("a_k");
                a.velocity = a_velocity;
                if a.inv_inertia != reali(0) {
                    a.angular_velocity = a_angular_velocity;
                }

                let b = self.bodies.get_mut(b_k).expect("b_k");
                b.velocity = b_velocity;
                if b.inv_inertia != reali(0) {
                    b.angular_velocity = b_angular_velocity;
                }
            }
        }
    }

    pub fn update_positions<D: Fn(Real2)->Real>(&mut self, dt: Real, map_distance: &D) {
        for (a_k, a) in self.bodies.iter_mut() {
            if a.inv_mass != reali(0) {
                //assert!(a.velocity.x.is_finite() && a.velocity.y.is_finite());
                let new_pos = a.pos + a.velocity * dt;
                //assert!(new_pos.x.is_finite() && new_pos.y.is_finite());
                let max_penetration = a.max_penetration;
                let (new_pos, _hit) = march_circle(a.pos, new_pos, (a.shape.radius() - max_penetration).max(reali(0)), &map_distance);
                if a.pos != new_pos {
                    self.spatial_hash.update(a.pos, a.pos, new_pos, new_pos, a.shape, a.shape_flags, a_k);
                    a.pos = new_pos;
                }
                a.rotation += a.angular_velocity * dt;
            }
            //a.torque = 0.0;
            //a.force = Real2::zero();
        }
    }

    pub fn sample_distance(&self, pos: Real2, ignore_k: BodyKey, mask: ShapeFlags, sdf_upper_bound: Real)->Real {
        let mut bodies = Vec::new();
        self.spatial_hash.find_neighbours(&mut bodies, pos, pos, Shape::Circle(sdf_upper_bound), mask);
        let mut result = sdf_upper_bound;
        for &k in bodies.iter() {
            if k == ignore_k {
                continue;
            }
            let body = self.bodies.get(k).expect("k");
            match body.shape {
                Shape::Circle(r) => {
                    result = result.min(sd_circle(pos, body.pos, r));
                }
                _ => {
                }
            }
        }
        result
    }

    pub fn sample_normal(&self, pos: Real2, ignore_k: BodyKey, mask: ShapeFlags, sdf_upper_bound: Real)->(Real, Real2) {
        let mut bodies = Vec::new();
        self.spatial_hash.find_neighbours(&mut bodies, pos, pos, Shape::Circle(sdf_upper_bound), mask);
        let distance_func = |p| {
            let mut result = sdf_upper_bound;
            for &k in bodies.iter() {
                if k == ignore_k {
                    continue;
                }
                let body = self.bodies.get(k).expect("k");
                match body.shape {
                    Shape::Circle(r) => {
                        result = result.min(sd_circle(p, body.pos, r));
                    }
                    _ => {
                    }
                }
            }
            result
        };
        let c = distance_func(pos);
        let grad = Real2::from((distance_func(pos + Real2::from((reali(1), reali(0)))) - c,
                                distance_func(pos + Real2::from((reali(0), reali(1)))) - c));
        let n = if grad.magnitude_squared() > (0.00001) {
            grad.normalized()
        } else {
            Real2::from((reali(0), reali(-1)))
        };
        (c, n)
    }


    pub fn find_exact<D: Fn(Real2)->Real>(&self, result: &mut Vec<BodyKey>, position: Real2, shape: Shape, mask: ShapeFlags, distance_func: &D) {
        self.spatial_hash.find_neighbours(result, position, position, shape, mask);
        result.retain(|k| {
            let b = &self.bodies[*k];
            Shape::test_overlap(b.shape, b.pos, shape, position, distance_func)
        });
    }

}
