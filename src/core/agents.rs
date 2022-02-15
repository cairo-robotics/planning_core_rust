use nalgebra::{Quaternion, UnitQuaternion, Vector3, Translation3, Isometry3};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::collision::collision_nn::CollisionNN;
use crate::collision::env_collision::*;
use crate::core::vars::AgentVars;
use crate::core::relaxed_ik::{RelaxedIK, Opt};
use crate::core::projection::OmegaProjection;
use crate::spacetime::robot::Robot;
use crate::utils_rust::file_utils::*;
use crate::utils_rust::sampler::ThreadRobotSampler;
use crate::utils_rust::yaml_utils::*;
use crate::utils_rust::subscriber_utils::EEPoseGoalsSubscriber;


#[pyclass]
struct Agent {
    pub agent_vars: AgentVars,
    pub relaxed_ik: Arc<Mutex<RelaxedIK>>,
    pub omega_projection: Arc<Mutex<OmegaProjection>>
}

#[pymethods]
impl Agent {
    #[new]
    fn new(settings_fp: String, position_mode_relative: bool, rotation_mode_relative: bool) -> Self {
        let agent_vars = init_agent_vars(settings_fp.clone(), position_mode_relative, rotation_mode_relative);
        let relaxed_ik: Arc<Mutex<RelaxedIK>> = Arc::new(Mutex::new(RelaxedIK::from_loaded(1)));
        let omega_projection: Arc<Mutex<OmegaProjection>> = Arc::new(Mutex::new(OmegaProjection::from_loaded(1)));
        return Agent { agent_vars, relaxed_ik, omega_projection};
    }

    fn forward_kinematics(&mut self, j_config: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let ee_poses: Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)> = self
            .agent_vars
            .robot
            .get_ee_pos_and_quat_immutable(&mut j_config.as_slice());

        let position_end_chain = ee_poses[0].0;
        let quat_chain = ee_poses[0].1.clone().into_inner();
        let mut ee_position = Vec::new();
        ee_position.push(position_end_chain.x);
        ee_position.push(position_end_chain.y);
        ee_position.push(position_end_chain.z);

        let mut ee_quat = Vec::new();
        ee_quat.push(quat_chain.w);
        ee_quat.push(quat_chain.i);
        ee_quat.push(quat_chain.j);
        ee_quat.push(quat_chain.k);

        let mut pose = Vec::new();
        pose.push(ee_position);
        pose.push(ee_quat);

        Ok(pose)
    }
    
    fn relaxed_inverse_kinematics(&mut self, pos_vec: Vec<f64>,  
        quat_vec: Vec<f64>) -> PyResult<Opt> {
    
        let arc = Arc::new(Mutex::new(EEPoseGoalsSubscriber::new()));
        let mut g = arc.lock().unwrap();
        
        g.pos_goals.push(Vector3::new(pos_vec[0], pos_vec[1], pos_vec[2]));
        let tmp_q = Quaternion::new(quat_vec[3], quat_vec[0], quat_vec[1], quat_vec[2]);
        g.quat_goals.push(UnitQuaternion::from_quaternion(tmp_q));
        
        let ja = self.relaxed_ik.lock().unwrap().solve(&g).clone();    
        let len = ja.len();
        
        Ok(Opt {data: ja, length: len})
    }

    fn dynamic_obstacle_cb(&mut self, name: String, pos_vec: Vec<f64>, quat_vec:Vec<f64>) ->  PyResult<()> {
    
        let ts = Translation3::new(pos_vec[0], pos_vec[1], pos_vec[2]);
        let tmp_q = Quaternion::new(quat_vec[3], quat_vec[0], quat_vec[1], quat_vec[2]);
        let rot = UnitQuaternion::from_quaternion(tmp_q);
        let pos = Isometry3::from_parts(ts, rot);
    
        self.relaxed_ik.lock().unwrap().vars.env_collision.update_dynamic_obstacle(&name, pos);
        Ok(())
    }

    fn update_keyframe_mean(&mut self, config_vec: Vec<f64>) -> PyResult<()> {
        self.agent_vars.keyframe_mean = config_vec.clone();
        Ok(())
    }

    fn omega_project(&mut self, pos_arr: Vec<f64>, pos_length: i32, 
        quat_arr: Vec<f64>, quat_length: i32, keyframe_mean_config: Vec<f64>) -> PyResult<Opt> {

        let _ = self.update_keyframe_mean(keyframe_mean_config);
        let pos_slice: &[f64] = &pos_arr.as_slice()[..pos_length as usize];
        let quat_slice: &[f64] = &quat_arr.as_slice()[..quat_length as usize];
    
        let pos_vec = pos_slice.to_vec();
        let quat_vec = quat_slice.to_vec();
    
        let arc = Arc::new(Mutex::new(EEPoseGoalsSubscriber::new()));
        let mut g = arc.lock().unwrap();
        
        for i in 0..self.omega_projection.lock().unwrap().vars.robot.num_chains {
            g.pos_goals.push( Vector3::new(pos_vec[3*i], pos_vec[3*i+1], pos_vec[3*i+2]) );
            let tmp_q = Quaternion::new(quat_vec[4*i+3], quat_vec[4*i], quat_vec[4*i+1], quat_vec[4*i+2]);
            g.quat_goals.push( UnitQuaternion::from_quaternion(tmp_q) );
        }
        
        let ja = self.omega_projection.lock().unwrap().solve(&g);    
        let len = ja.len();
        Ok(Opt {data: ja, length: len})
        }
}

fn init_agent_vars(settings_fp: String, position_mode_relative: bool, rotation_mode_relative: bool) -> AgentVars {
        
    let info_file_name = get_info_file_name(settings_fp);
    let path_to_config = get_path_to_config();
    let fp = path_to_config + "/info_files/" + info_file_name.as_str();
    let ifp = InfoFileParser::from_yaml_path(fp.clone());
    println!("{:?}", ifp.joint_names);
    println!("{}", get_path_to_config());
    let mut robot = Robot::from_yaml_path(fp.clone());
    let num_chains = ifp.joint_names.len();
    let sampler = ThreadRobotSampler::new(robot.clone());

    let mut goal_positions: Vec<Vector3<f64>> = Vec::new();
    let mut goal_quats: Vec<UnitQuaternion<f64>> = Vec::new();

    let init_ee_positions = robot.get_ee_positions(ifp.starting_config.as_slice());
    let init_ee_quats = robot.get_ee_quats(ifp.starting_config.as_slice());

    for i in 0..num_chains {
        goal_positions.push(init_ee_positions[i]);
        goal_quats.push(init_ee_quats[i]);
    }

    let collision_nn_path = get_path_to_config()
        + "/collision_nn_rust/"
        + ifp.collision_nn_file.as_str()
        + ".yaml";
    let collision_nn = CollisionNN::from_yaml_path(collision_nn_path);

    let fp = get_path_to_config() + "/settings.yaml";
    let fp2 = fp.clone();
    println!("AgentVars from_yaml_path {}", fp);
    let env_collision_file = EnvCollisionFileParser::from_yaml_path(fp);
    let frames = robot.get_frames_immutable(&ifp.starting_config.clone());
    let env_collision =
        RelaxedIKEnvCollision::init_collision_world(env_collision_file, &frames);
    let objective_mode = get_objective_mode(fp2);

    let agent_vars = AgentVars {
        robot,
        sampler,
        init_state: ifp.starting_config.clone(),
        xopt: ifp.starting_config.clone(),
        prev_state: ifp.starting_config.clone(),
        prev_state2: ifp.starting_config.clone(),
        prev_state3: ifp.starting_config.clone(),
        goal_positions,
        goal_quats,
        init_ee_positions,
        init_ee_quats,
        position_mode_relative,
        rotation_mode_relative,
        collision_nn,
        env_collision,
        objective_mode,
        keyframe_mean: ifp.starting_config.clone()
    };
    agent_vars
}

fn init_relaked_ik(settings_fp: String, position_mode_relative: bool, rotation_mode_relative: bool) -> AgentVars {
        
    let info_file_name = get_info_file_name(settings_fp);
    let path_to_config = get_path_to_config();
    let fp = path_to_config + "/info_files/" + info_file_name.as_str();
    let ifp = InfoFileParser::from_yaml_path(fp.clone());
    println!("{:?}", ifp.joint_names);
    println!("{}", get_path_to_config());
    let mut robot = Robot::from_yaml_path(fp.clone());
    let num_chains = ifp.joint_names.len();
    let sampler = ThreadRobotSampler::new(robot.clone());

    let mut goal_positions: Vec<Vector3<f64>> = Vec::new();
    let mut goal_quats: Vec<UnitQuaternion<f64>> = Vec::new();

    let init_ee_positions = robot.get_ee_positions(ifp.starting_config.as_slice());
    let init_ee_quats = robot.get_ee_quats(ifp.starting_config.as_slice());

    for i in 0..num_chains {
        goal_positions.push(init_ee_positions[i]);
        goal_quats.push(init_ee_quats[i]);
    }

    let collision_nn_path = get_path_to_config()
        + "/collision_nn_rust/"
        + ifp.collision_nn_file.as_str()
        + ".yaml";
    let collision_nn = CollisionNN::from_yaml_path(collision_nn_path);

    let fp = get_path_to_config() + "/settings.yaml";
    let fp2 = fp.clone();
    println!("AgentVars from_yaml_path {}", fp);
    let env_collision_file = EnvCollisionFileParser::from_yaml_path(fp);
    let frames = robot.get_frames_immutable(&ifp.starting_config.clone());
    let env_collision =
        RelaxedIKEnvCollision::init_collision_world(env_collision_file, &frames);
    let objective_mode = get_objective_mode(fp2);

    let agent_vars = AgentVars {
        robot,
        sampler,
        init_state: ifp.starting_config.clone(),
        xopt: ifp.starting_config.clone(),
        prev_state: ifp.starting_config.clone(),
        prev_state2: ifp.starting_config.clone(),
        prev_state3: ifp.starting_config.clone(),
        goal_positions,
        goal_quats,
        init_ee_positions,
        init_ee_quats,
        position_mode_relative,
        rotation_mode_relative,
        collision_nn,
        env_collision,
        objective_mode,
        keyframe_mean: ifp.starting_config.clone()
    };
    agent_vars
}


pub(crate) fn register(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Agent>()?;
    Ok(())
}