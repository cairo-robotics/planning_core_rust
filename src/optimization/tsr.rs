use nalgebra::geometry::Isometry3;
use nalgebra::Vector3;
use std::f32::consts::PI;

pub struct TSR {
    pub T0_w:  Isometry3<f64>,
    pub Tw_e:  Isometry3<f64>,
    pub Bw:  Vec<Vec<f64>>
}

impl TSR {
    pub fn new_from_poses(T0_w: &Vec<f64>, Tw_e: &Vec<f64>, Bw: &Vec<Vec<f64>>) -> Self {
        let T0_w_translation = Vector3::new(T0_w[0], T0_w[1], T0_w[2]);
        let T0_w_axisangle = Vector3::new(T0_w[3], T0_w[4], T0_w[5]);
        let Tw_e_translation = Vector3::new(Tw_e[0], Tw_e[1], Tw_e[2]);
        let Tw_e_axisangle = Vector3::new(Tw_e[3], Tw_e[4], Tw_e[5]);
        let T0_w_iso = Isometry3::new(T0_w_translation, T0_w_axisangle);
        let Tw_e_iso = Isometry3::new(Tw_e_translation, Tw_e_axisangle);

        TSR {
            T0_w: T0_w_iso,
            Tw_e: Tw_e_iso,
            Bw: Bw.clone()
        }
    }
}

pub fn distance_to_TSR(T0_s: &Isometry3<f64>, tsr: &TSR) -> (f64, Vec<f64>) {
    // pose of the grasp location or the pose of the object held by the hand in world coordinates
    let T0_sp = T0_s * tsr.Tw_e.inverse();
    // T0_sp in terms of the coordinates of the target frame w given by the Task Space Region tsr.
    let Tw_sp = tsr.T0_w.inverse() * T0_sp;
    // Generate the displacement vector of Tw_sp. Displacement represents the error given T0_s relative to Tw_e transform.
    let mut disp = displacement(&Tw_sp);
    // We want the global translation, not the translation dependent on rotation between the transformations.
    disp[0] = T0_sp.translation.x - tsr.T0_w.translation.x;
    disp[1] = T0_sp.translation.y - tsr.T0_w.translation.y;
    disp[2] = T0_sp.translation.z - tsr.T0_w.translation.z;
    // Since there are equivalent angle displacements for rpy, generate those equivalents by added +/- PI.
    // Use the smallest delta_x_dist of the equivalency set.
    let rpys = generate_equivalent_euler_angles(&disp[2..6]);
    let mut deltas = Vec::new();
    deltas.push(delta_x(disp.as_slice(), &tsr.Bw));
    for rpy in rpys.iter() {
        deltas.push(
            delta_x(vec![disp[0], disp[1], disp[2], rpy[0], rpy[1], rpy[2]].as_slice(), &tsr.Bw))
    }
    let mut distances = Vec::new();
    for delta in deltas.clone() {
        distances.push(l2_norm(delta.clone().as_slice()))
    }
    let min_dist = distances.clone().into_iter().reduce(f64::min).unwrap();
    let index = distances.into_iter().position(|r| r == min_dist).unwrap();
    let delta_x_values = deltas.remove(index);
    return (min_dist, delta_x_values)
} 

pub fn displacement(T0_s: &Isometry3<f64>) -> Vec<f64>{
   
    // Tv = [-Tm[0:3, 3][2], -Tm[0:3, 3][0], -Tm[0:3, 3][1]]
    let mut displacements = Vec::new();
    displacements.push(T0_s.translation.vector[0]);
    displacements.push(T0_s.translation.vector[1]);    
    displacements.push(T0_s.translation.vector[2]);
    let quat = T0_s.rotation;
    let euler = quat.euler_angles();
    displacements.push(euler.0);
    displacements.push(euler.1);
    displacements.push(euler.2);
    return displacements
}

pub fn generate_equivalent_euler_angles(rpy: &[f64]) -> Vec<Vec<f64>>{
    // Given rpy angles, produces a set of rpy's that are equivalent +/- pi 
    // but might ultimately produce smaller distances from a TSR.
    let f64PI = f64::from(PI);
    let rolls = vec![rpy[0] + f64PI, rpy[0] - f64PI];
    let pitches = vec![rpy[1] + f64PI, rpy[1] - f64PI];
    let yaws = vec![rpy[2] + f64PI, rpy[2] - f64PI];

    cartesian_product(&[rolls.as_slice(), pitches.as_slice(), yaws.as_slice()])
}

pub fn delta_x(displacement: &[f64], bounds: &[Vec<f64>]) -> Vec<f64>{

    // Given a vector of displacements and a bounds/constraint matrix it produces a differential vector
    // that represents the distance the displacement is from the bounds dictated by the constraint bounds. 

    // For each displacement value, if the value is within the limits of the respective bound, it will be 0.
    let mut delta = Vec::new();
    for i in 0..displacement.len(){
        let cmin = bounds[i][0];
        let cmax = bounds[i][1];
        let di = displacement[i];
        if di > cmax {
            delta.push(di - cmax);
        } else if di < cmin {
            delta.push(di - cmin);
        } else {
            delta.push(0.0);
        }
    }
    delta
}

pub fn delta_x_replacement(displacement: &[f64], bounds: &[Vec<f64>], replacements: &[f64]) -> Vec<f64>{

    // Given a vector of displacements and a bounds/constraint matrix it produces a differential vector
    // that represents the distance the displacement is from the bounds dictated by the constraint bounds. 

    // For each displacement value, if the value is within the limits of the respective bound, it will be 0.
    let mut delta = Vec::new();
    for i in 0..displacement.len(){
        let cmin = bounds[i][0];
        let cmax = bounds[i][1];
        let di = displacement[i];
        if di > cmax {
            delta.push(replacements[i] + di - cmax);
        } else if di < cmin {
            delta.push(replacements[i] + di - cmin);
        } else {
            delta.push(replacements[i]);
        }
    }
    delta
}


pub fn partial_cartesian(a: Vec<Vec<f64>>, b: &[f64]) -> Vec<Vec<f64>> {
    a.into_iter().flat_map(|xs| {
        b.iter().cloned().map(|y| {
            let mut vec = xs.clone();
            vec.push(y);
            vec
        }).collect::<Vec<_>>()
    }).collect()
}

pub fn cartesian_product(lists: &[&[f64]]) -> Vec<Vec<f64>> {
        match lists.split_first() {
            Some((first, rest)) => {
                let init: Vec<Vec<f64>> = first.iter().cloned().map(|n| vec![n]).collect();
    
                rest.iter().cloned().fold(init, |vec, list| {
                    partial_cartesian(vec, list)
                })
            },
            None => {
                vec![]
            }
        }
    }

pub fn l2_norm(vec: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..vec.len(){
        sum += f64::powi(vec[i], 2);
    }
    sum.sqrt()
}