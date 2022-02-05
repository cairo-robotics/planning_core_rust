use pyo3::prelude::*;
use pyo3::types::{PyList};
use nalgebra::{UnitQuaternion, Vector3, Quaternion, Point3};
use crate::utils_rust::yaml_utils::{*};
use crate::spacetime::robot::Robot;
use crate::collision::collision_nn::CollisionNN;
use crate::utils_rust::sampler::ThreadRobotSampler;
use crate::utils_rust::file_utils::{*};
use crate::collision::env_collision::{*};

use crate::core::vars::AgentVars;


#[pyclass]
struct Agent {
    pub agent_vars: AgentVars 
}

#[pymethods]
impl Agent {

    #[new]
    fn new(fp: String, position_mode_relative: bool, rotation_mode_relative: bool) -> Agent {
        let ifp = InfoFileParser::from_yaml_path(fp.clone());
        println!("{:?}", ifp.joint_names);
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

        let collision_nn_path = get_path_to_src()+ "relaxed_ik_core/config/collision_nn_rust/" + ifp.collision_nn_file.as_str() + ".yaml";
        let collision_nn = CollisionNN::from_yaml_path(collision_nn_path);

        let fp = get_path_to_src() + "relaxed_ik_core/config/settings.yaml";
        let fp2 = fp.clone();
        println!("AgentVars from_yaml_path {}", fp);
        let env_collision_file = EnvCollisionFileParser::from_yaml_path(fp);
        let frames = robot.get_frames_immutable(&ifp.starting_config.clone());
        let env_collision = RelaxedIKEnvCollision::init_collision_world(env_collision_file, &frames);
        let objective_mode = get_objective_mode(fp2);

        let agent_vars = AgentVars{robot, sampler, init_state: ifp.starting_config.clone(), xopt: ifp.starting_config.clone(),
            prev_state: ifp.starting_config.clone(), prev_state2: ifp.starting_config.clone(), prev_state3: ifp.starting_config.clone(),
            goal_positions, goal_quats, init_ee_positions, init_ee_quats, position_mode_relative, rotation_mode_relative, collision_nn, 
            env_collision, objective_mode};
        return Agent { agent_vars };
    }

    fn get_pose(&mut self, j_config: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let args: &[f64] = j_config.as_slice();
        let ee_poses: Vec<(nalgebra::Vector3<f64>, nalgebra::UnitQuaternion<f64>)> = self.agent_vars.robot.get_ee_pos_and_quat_immutable(args);
       
 
        let position_end_chain= ee_poses[self.agent_vars.robot.num_chains].0;
        let quat_chain = ee_poses[self.agent_vars.robot.num_chains].1.clone().into_inner();
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

}

#[pymodule]
fn planning_core_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Agent>()?;
    Ok(())
}