use nalgebra::geometry::Isometry3;
use std::f32::consts::PI;

pub struct TSR {
    pub T0_w:  Isometry3<f64>,
    pub Tw_e:  Isometry3<f64>,
    pub Bw:  Vec<Vec<f64>>
}


pub fn distanceToTSR(T0_s: &Isometry3<f64>, tsr: &TSR) -> (f64, Vec<f64>) {
    // pose of the grasp location or the pose of the object held by the hand in world coordinates
    let T0_sp = T0_s * tsr.Tw_e.inverse();
    // T0_sp in terms of the coordinates of the target frame w given by the Task Space Region tsr.
    let Tw_sp = tsr.T0_w.inverse() * T0_sp;
    // Generate the displacement vector of Tw_sp. Displacement represents the error given T0_s relative to Tw_e transform.
    let disp = displacement(&Tw_sp);
    // Since there are equivalent angle displacements for rpy, generate those equivalents by added +/- PI.
    // Use the smallest delta_x_dist of the equivalency set.
    let rpys = generate_equivalent_euler_angles(&disp[2..6]);
    let deltas = Vec::new();
    deltas.push(delta_x(disp.as_slice(), &tsr.Bw));
    for rpy in rpys.iter() {
        deltas.push(
            delta_x(vec![disp[0], disp[1], disp[2], rpy[0], rpy[1], rpy[2]].as_slice(), &tsr.Bw))
    }
    let distances = Vec::new();
    for delta in deltas {
        distances.push(l2_norm(delta.as_slice()))
    }
    let min_dist = distances.into_iter().reduce(f64::min).unwrap();
    let index = distances.iter().position(|&r| r == min_dist).unwrap();
    return (min_dist, deltas[index])
}

pub fn displacement(T0_s: &Isometry3<f64>) -> Vec<f64>{
   
    // Tv = [-Tm[0:3, 3][2], -Tm[0:3, 3][0], -Tm[0:3, 3][1]]
    let displacements = Vec::new();
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
    let delta = Vec::new();
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
    let sum = 0.0;
    for i in 0..vec.len(){
        sum += f64::powi(vec[i], 2);
    }
    sum.sqrt()
}