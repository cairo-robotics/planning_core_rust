use crate::core::{vars};
use crate::collision::env_collision::{*};
use crate::utils_rust::transformations::{*};
use crate::optimization::loss::{groove_loss, groove_loss_derivative};
use crate::optimization::tsr::{*};
use nalgebra::geometry::{UnitQuaternion, Quaternion, IsometryMatrix3, Translation3, Rotation3};
use nalgebra::{Point3, Vector3, Normed};
use ncollide3d::{shape, query};
use std::ops::Deref;


pub trait ObjectiveTrait {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64;
    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64;
    fn gradient(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = Vec::new();
        let f_0 = self.call(x, v, frames);
        
        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.000000001;
            let frames_h = v.robot.get_frames_immutable(x_h.as_slice());
            let f_h = self.call(x_h.as_slice(), v, &frames_h);
            grad.push( (-f_0 + f_h) / 0.000000001);
        }

        (f_0, grad)
    }
    fn gradient_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> (f64, Vec<f64>) {
        let mut grad: Vec<f64> = Vec::new();
        let f_0 = self.call_lite(x, v, ee_poses);

        for i in 0..x.len() {
            let mut x_h = x.to_vec();
            x_h[i] += 0.0000001;
            let ee_poses_h = v.robot.get_ee_pos_and_quat_immutable(x_h.as_slice());
            let f_h = self.call_lite(x_h.as_slice(), v, &ee_poses_h);
            grad.push( (-f_0 + f_h) / 0.0000001);
        }

        (f_0, grad)
    }
    fn gradient_type(&self) -> usize {return 1}  // manual diff = 0, finite diff = 1
}

pub struct MatchEEPosGoals {
    pub arm_idx: usize
}
impl MatchEEPosGoals {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for MatchEEPosGoals {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_elem = frames[self.arm_idx].0.len() - 1;
        let x_val = ( frames[self.arm_idx].0[last_elem] - v.goal_positions[self.arm_idx] ).norm();

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let x_val = ( ee_poses[self.arm_idx].0 - v.goal_positions[self.arm_idx] ).norm();
        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }
}

pub struct MatchEEQuatGoals {
    pub arm_idx: usize
}
impl MatchEEQuatGoals {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for MatchEEQuatGoals {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_elem = frames[self.arm_idx].1.len() - 1;
        let tmp = Quaternion::new(-frames[self.arm_idx].1[last_elem].w, -frames[self.arm_idx].1[last_elem].i, -frames[self.arm_idx].1[last_elem].j, -frames[self.arm_idx].1[last_elem].k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

        let disp = angle_between(v.goal_quats[self.arm_idx], frames[self.arm_idx].1[last_elem]);
        let disp2 = angle_between(v.goal_quats[self.arm_idx], ee_quat2);
        let x_val = disp.min(disp2);

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let tmp = Quaternion::new(-ee_poses[self.arm_idx].1.w, -ee_poses[self.arm_idx].1.i, -ee_poses[self.arm_idx].1.j, -ee_poses[self.arm_idx].1.k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

        let disp = angle_between(v.goal_quats[self.arm_idx], ee_poses[self.arm_idx].1);
        let disp2 = angle_between(v.goal_quats[self.arm_idx], ee_quat2);
        let x_val = disp.min(disp2);
        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }
}


pub struct NNSelfCollision;
impl ObjectiveTrait for NNSelfCollision {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val = v.collision_nn.predict(&x.to_vec());

        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val = v.collision_nn.predict(&x.to_vec());
        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }

    fn gradient(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> (f64, Vec<f64>) {
        let (x_val, mut grad) = v.collision_nn.gradient(&x.to_vec());
        let g_prime = groove_loss_derivative(x_val, 0., 2, 2.1, 0.0002, 4);
        for i in 0..grad.len() {
            grad[i] *= g_prime;
        }
        (x_val, grad)
    }

    fn gradient_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> (f64, Vec<f64>) {
        let (x_val, mut grad) = v.collision_nn.gradient(&x.to_vec());
        let g_prime = groove_loss_derivative(x_val, 0., 2, 2.1, 0.0002, 4);
        for i in 0..grad.len() {
            grad[i] *= g_prime;
        }
        (x_val, grad)
    }

    fn gradient_type(&self) -> usize {return 0}
}

pub struct EnvCollision {
    pub arm_idx: usize
}
impl EnvCollision {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for EnvCollision {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        // let start = PreciseTime::now();\
        let mut x_val: f64 = 0.0;
        let link_radius = v.env_collision.link_radius;
        let penalty_cutoff: f64 = link_radius * 2.0;
        let a = penalty_cutoff.powi(2);
        for (option, score) in &v.env_collision.active_obstacles[self.arm_idx] {
            if let Some(handle) = option {
                let mut sum: f64 = 0.0;
                let obstacle = v.env_collision.world.objects.get(*handle).unwrap();
                let last_elem = frames[self.arm_idx].0.len() - 1;
                for i in 0..last_elem {
                    let start_pt = Point3::from(frames[self.arm_idx].0[i]);
                    let end_pt = Point3::from(frames[self.arm_idx].0[i + 1]);
                    let segment = shape::Segment::new(start_pt, end_pt);
                    let segment_pos = nalgebra::one();
                    let dis = query::distance(obstacle.position(), obstacle.shape().deref(), &segment_pos, &segment) - link_radius;
                    sum += a / (dis + link_radius).powi(2);
                }
                x_val += sum;
            }
        }
        
        // let end = PreciseTime::now();

        groove_loss(x_val, 0., 2, 3.5, 0.00005, 4)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let x_val = 1.0; // placeholder
        groove_loss(x_val, 0., 2, 2.1, 0.0002, 4)
    }
}

pub struct JointLimits;
impl ObjectiveTrait for JointLimits {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut sum = 0.0;
        let penalty_cutoff: f64 = 0.9;
        let a = 0.05 / (penalty_cutoff.powi(50));
        for i in 0..v.robot.num_dof {
            let l = v.robot.bounds[i][0];
            let u = v.robot.bounds[i][1];
            let r = (x[i] - l) / (u - l);
            let n = 2.0 * (r - 0.5);
            sum += a*n.powf(50.);
        }
        groove_loss(sum, 0.0, 2, 0.32950, 0.1, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut sum = 0.0;
        let penalty_cutoff: f64 = 0.85;
        let a = 0.05 / (penalty_cutoff.powi(50));
        for i in 0..v.robot.num_dof {
            let l = v.robot.bounds[i][0];
            let u = v.robot.bounds[i][1];
            let r = (x[i] - l) / (u - l);
            let n = 2.0 * (r - 0.5);
            sum += a*n.powi(50);
        }

        groove_loss(sum, 0.0, 2, 0.32950, 0.1, 2)
    }
}

pub struct MinimizeVelocity;
impl ObjectiveTrait for MinimizeVelocity {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
           x_val += (x[i] - v.xopt[i]).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
           x_val += (x[i] - v.xopt[i]).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

}

pub struct MinimizeAcceleration;
impl ObjectiveTrait for MinimizeAcceleration {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            x_val += (v1 - v2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            x_val += (v1 - v2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }
}

pub struct MinimizeJerk;
impl ObjectiveTrait for MinimizeJerk {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            let v3 = v.prev_state[i] - v.prev_state2[i];
            let a1 = v1 - v2;
            let a2 = v2 - v3;
            x_val += (a1 - a2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let mut x_val: f64 = 0.0;
        for i in 0..x.len() {
            let v1 = x[i] - v.xopt[i];
            let v2 = v.xopt[i] - v.prev_state[i];
            let v3 = v.prev_state[i] - v.prev_state2[i];
            let a1 = v1 - v2;
            let a2 = v2 - v3;
            x_val += (a1 - a2).powi(2);
        }
        x_val = x_val.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }
}
pub struct MinimizeDistanceKeyframeMeanPosition{
    pub arm_idx: usize
}
impl MinimizeDistanceKeyframeMeanPosition {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for MinimizeDistanceKeyframeMeanPosition {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_pos_elem = frames[self.arm_idx].0.len() - 1;
        let pos = vec![frames[self.arm_idx].0[last_pos_elem].x, frames[self.arm_idx].0[last_pos_elem].y, frames[self.arm_idx].0[last_pos_elem].z];
        let keyframe_mean_position = v.keyframe_mean_pose[0].clone();
        let mut sum: f64 = 0.0;
        for i in 0..pos.len() {
            let diff = pos[i] - keyframe_mean_position[i];
            sum += diff.powi(2);
        }
        let x_val = sum.sqrt();

        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let pos = vec![ee_poses[self.arm_idx].0.x, ee_poses[self.arm_idx].0.y, ee_poses[self.arm_idx].0.z];
        let keyframe_mean_position = v.keyframe_mean_pose[0].clone();
        let mut sum: f64 = 0.0;
        for i in 0..pos.len() {
            let diff = pos[i] - keyframe_mean_position[i];
            sum += diff.powi(2);
        }
        let x_val = sum.sqrt();
        groove_loss(x_val, 0.0, 2, 0.1, 10.0, 2)
    }
}

pub struct MinimizeDistanceKeyframeMeanOrientation{
    pub arm_idx: usize
}
impl MinimizeDistanceKeyframeMeanOrientation {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}impl ObjectiveTrait for MinimizeDistanceKeyframeMeanOrientation {
    fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_elem = frames[self.arm_idx].1.len() - 1;
        let tmp = Quaternion::new(-frames[self.arm_idx].1[last_elem].w, -frames[self.arm_idx].1[last_elem].i, -frames[self.arm_idx].1[last_elem].j, -frames[self.arm_idx].1[last_elem].k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

        let keyframe_mean_orientation = v.keyframe_mean_pose[1].clone();
        let target_quat =  Quaternion::new(keyframe_mean_orientation[3], keyframe_mean_orientation[0], keyframe_mean_orientation[1], keyframe_mean_orientation[2]);
        let target_unit_quat = UnitQuaternion::from_quaternion(target_quat);

        let disp = angle_between(target_unit_quat, frames[self.arm_idx].1[last_elem]);
        let disp2 = angle_between(target_unit_quat, ee_quat2);
        let x_val = disp.min(disp2);

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }


    fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let tmp = Quaternion::new(ee_poses[self.arm_idx].1.w, ee_poses[self.arm_idx].1.i, ee_poses[self.arm_idx].1.j, ee_poses[self.arm_idx].1.k);
        let ee_quat2 = UnitQuaternion::from_quaternion(tmp);
             
        let keyframe_mean_orientation = v.keyframe_mean_pose[1].clone();
        let target_quat =  Quaternion::new(keyframe_mean_orientation[3], keyframe_mean_orientation[0], keyframe_mean_orientation[1], keyframe_mean_orientation[2]);
        let target_unit_quat = UnitQuaternion::from_quaternion(target_quat);
        
        let disp = angle_between(target_unit_quat, ee_poses[self.arm_idx].1);
        let disp2 = angle_between(target_unit_quat, ee_quat2);
        let x_val = disp.min(disp2);

        groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    }

    
}

pub struct PlanningTSRError{
    pub arm_idx: usize 
}
impl PlanningTSRError {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for PlanningTSRError {
    fn call(&self, _x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_pos_elem = frames[self.arm_idx].0.len() - 1;
        let pos = Translation3::new(frames[self.arm_idx].0[last_pos_elem].x, frames[self.arm_idx].0[last_pos_elem].y, frames[self.arm_idx].0[last_pos_elem].z);
        let last_quat_elem = frames[0].1.len() - 1;
        let tmp = Quaternion::new(frames[0].1[last_quat_elem].w, frames[0].1[last_quat_elem].i, frames[0].1[last_quat_elem].j, frames[0].1[last_quat_elem].k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        
        // println!("{:?}", &v.planning_tsr.T0_w);
        // println!("{:?}", &v.planning_tsr.Tw_e);
        // println!("{:?}", &v.planning_tsr.Bw);
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        // println!("{:?}", _x);
        // println!("{:?}, {:?}", pos, rot);
        // println!("{}", distance_and_delta.0);
        // println!("{:?}", distance_and_delta.1);
        groove_loss(distance_and_delta.0, 0.0, 2, 0.5, 10.0, 2)
    }

    fn call_lite(&self, _x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let pos = Translation3::new(ee_poses[self.arm_idx].0.x, ee_poses[self.arm_idx].0.y, ee_poses[self.arm_idx].0.z);
        let tmp = Quaternion::new(ee_poses[self.arm_idx].1.w, ee_poses[self.arm_idx].1.i, ee_poses[self.arm_idx].1.j, ee_poses[self.arm_idx].1.k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        groove_loss(distance_and_delta.0, 0.0, 2, 0.5, 10.0, 2)
    }
}

pub struct SecondaryTSRError{
    pub arm_idx: usize
}
impl SecondaryTSRError {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for SecondaryTSRError {
    fn call(&self, _x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_pos_elem = frames[self.arm_idx].0.len() - 1;
        let pos = Translation3::new(frames[self.arm_idx].0[last_pos_elem].x, frames[self.arm_idx].0[last_pos_elem].y, frames[self.arm_idx].0[last_pos_elem].z);
        let last_quat_elem = frames[0].1.len() - 1;
        let tmp = Quaternion::new(frames[0].1[last_quat_elem].w, frames[0].1[last_quat_elem].i, frames[0].1[last_quat_elem].j, frames[0].1[last_quat_elem].k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        
        // println!("{:?}", &v.planning_tsr.T0_w);
        // println!("{:?}", &v.planning_tsr.Tw_e);
        // println!("{:?}", &v.planning_tsr.Bw);
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        // println!("{:?}", _x);
        // println!("{:?}, {:?}", pos, rot);
        // println!("{}", distance_and_delta.0);
        // println!("{:?}", distance_and_delta.1);
        groove_loss(distance_and_delta.0, 0.0, 2, 0.1, 10.0, 2)
    }

    fn call_lite(&self, _x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let pos = Translation3::new(ee_poses[self.arm_idx].0.x, ee_poses[self.arm_idx].0.y, ee_poses[self.arm_idx].0.z);
        let tmp = Quaternion::new(ee_poses[self.arm_idx].1.w, ee_poses[self.arm_idx].1.i, ee_poses[self.arm_idx].1.j, ee_poses[self.arm_idx].1.k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        groove_loss(distance_and_delta.0, 0.0, 2, 0.1, 10.0, 2)
    }
}

pub struct TSRPosGoal {
    pub arm_idx: usize
}
impl TSRPosGoal {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for TSRPosGoal {
    fn call(&self, _x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
        let last_pos_elem = frames[self.arm_idx].0.len() - 1;
        let pos = Translation3::new(frames[self.arm_idx].0[last_pos_elem].x, frames[self.arm_idx].0[last_pos_elem].y, frames[self.arm_idx].0[last_pos_elem].z);
        let last_quat_elem = frames[0].1.len() - 1;
        let tmp = Quaternion::new(frames[0].1[last_quat_elem].w, frames[0].1[last_quat_elem].i, frames[0].1[last_quat_elem].j, frames[0].1[last_quat_elem].k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        let translation_deltas = &distance_and_delta.1[0..2];
        let x_val = l2_norm(translation_deltas);
        
        groove_loss(x_val, 0., 2, 0.3, 10.0, 2)
    }

    fn call_lite(&self, _x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let pos = Translation3::new(ee_poses[self.arm_idx].0.x, ee_poses[self.arm_idx].0.y, ee_poses[self.arm_idx].0.z);
        let tmp = Quaternion::new(ee_poses[self.arm_idx].1.w, ee_poses[self.arm_idx].1.i, ee_poses[self.arm_idx].1.j, ee_poses[self.arm_idx].1.k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        let translation_deltas = &distance_and_delta.1[0..2];
        let x_val = l2_norm(translation_deltas);

        groove_loss(x_val, 0., 2, 0.3, 10.0, 2)
    }
}

// impl ObjectiveTrait for TSRPosGoal {
//     fn call(&self, _x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
//         let last_pos_elem = frames[self.arm_idx].0.len() - 1;
//         let current_pos = frames[self.arm_idx].0[last_pos_elem];
//         let target_pos = v.planning_tsr.T0_w.translation.vector;       
//         let x_val = (target_pos - current_pos).norm();
//         groove_loss(x_val, 0., 2, 0.3, 10.0, 2)
//     }

//     fn call_lite(&self, _x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
//         let current_pos = ee_poses[self.arm_idx].0;
//         let target_pos = v.planning_tsr.T0_w.translation.vector;       
//         let x_val = (target_pos - current_pos).norm();
//         groove_loss(x_val, 0., 2, 0.3, 10.0, 2)
//     }
// }


pub struct TSRQuatGoal {
    pub arm_idx: usize
}
impl TSRQuatGoal {
    pub fn new(arm_idx: usize) -> Self {Self{arm_idx}}
}
impl ObjectiveTrait for TSRQuatGoal {
    fn call(&self, _x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
         let last_pos_elem = frames[self.arm_idx].0.len() - 1;
        let pos = Translation3::new(frames[self.arm_idx].0[last_pos_elem].x, frames[self.arm_idx].0[last_pos_elem].y, frames[self.arm_idx].0[last_pos_elem].z);
        let last_quat_elem = frames[0].1.len() - 1;
        let tmp = Quaternion::new(frames[0].1[last_quat_elem].w, frames[0].1[last_quat_elem].i, frames[0].1[last_quat_elem].j, frames[0].1[last_quat_elem].k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
        
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        let rotation_deltas = &distance_and_delta.1[2..5];
        let x_val = l2_norm(rotation_deltas);
        groove_loss(x_val, 0., 2, 0.5, 10.0, 2)
    }

    fn call_lite(&self, _x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
        let pos = Translation3::new(ee_poses[self.arm_idx].0.x, ee_poses[self.arm_idx].0.y, ee_poses[self.arm_idx].0.z);
        let tmp = Quaternion::new(ee_poses[self.arm_idx].1.w, ee_poses[self.arm_idx].1.i, ee_poses[self.arm_idx].1.j, ee_poses[self.arm_idx].1.k);
        let rot = Rotation3::from(UnitQuaternion::from_quaternion(tmp));
        let ts0_s_iso = IsometryMatrix3::from_parts(pos, rot);
     
        let distance_and_delta = distance_to_TSR(&ts0_s_iso, &v.planning_tsr);
        let rotation_deltas = &distance_and_delta.1[2..5];
        let x_val = l2_norm(rotation_deltas);

        groove_loss(x_val, 0., 2, 0.5, 10.0, 2)
    }

    // fn call(&self, x: &[f64], v: &vars::AgentVars, frames: &Vec<(Vec<nalgebra::Vector3<f64>>, Vec<nalgebra::UnitQuaternion<f64>>)>) -> f64 {
    //     let last_elem = frames[self.arm_idx].1.len() - 1;
    //     let tmp = Quaternion::new(-frames[self.arm_idx].1[last_elem].w, -frames[self.arm_idx].1[last_elem].i, -frames[self.arm_idx].1[last_elem].j, -frames[self.arm_idx].1[last_elem].k);
    //     let ee_quat2 = UnitQuaternion::from_quaternion(tmp);
    //     let target_quat = nalgebra::try_convert(v.planning_tsr.T0_w.rotation).unwrap();
    //     let disp = angle_between(target_quat, frames[self.arm_idx].1[last_elem]);
    //     let disp2 = angle_between(target_quat, ee_quat2);
    //     let x_val = disp.min(disp2);
    //     groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    // }

    // fn call_lite(&self, x: &[f64], v: &vars::AgentVars, ee_poses: &Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)>) -> f64 {
    //     let tmp = Quaternion::new(-ee_poses[self.arm_idx].1.w, -ee_poses[self.arm_idx].1.i, -ee_poses[self.arm_idx].1.j, -ee_poses[self.arm_idx].1.k);
    //     let ee_quat2 = UnitQuaternion::from_quaternion(tmp);

    //     let target_quat = nalgebra::try_convert(v.planning_tsr.T0_w.rotation).unwrap();
    //     let disp = angle_between(target_quat, ee_poses[self.arm_idx].1);
    //     let disp2 = angle_between(target_quat, ee_quat2);
    //     let x_val = disp.min(disp2);
    //     groove_loss(x_val, 0., 2, 0.1, 10.0, 2)
    // }
}