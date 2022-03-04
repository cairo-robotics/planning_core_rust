use pyo3::prelude::*;

use crate::core::vars::AgentVars;
use crate::optimization::opt::{OptimizationEngineOpen, OptimizationEngineNLopt};
use crate::optimization::objective_master::ObjectiveMaster;
use crate::utils_rust::file_utils::{*};
use crate::utils_rust::subscriber_utils::EEPoseGoalsSubscriber;
use crate::utils_rust::transformations::{*};
use crate::utils_rust::yaml_utils::{*};
use nalgebra::{Vector3, UnitQuaternion, Quaternion};
use crate::utils_rust::sampler::ThreadSampler;


pub struct TSROptimization {
    pub vars: AgentVars,
    pub om: ObjectiveMaster,
    pub groove: OptimizationEngineOpen,
    pub groove_nlopt: OptimizationEngineNLopt
}

impl TSROptimization {
    pub fn from_info_file_name(info_file_name: String, mode: usize) -> Self {
        let path_to_src = get_path_to_config();
        let fp = path_to_src + "/info_files/" + info_file_name.as_str();
        TSROptimization::from_yaml_path(fp.clone(), mode.clone())
    }

    pub fn from_yaml_path(fp: String, mode: usize) -> Self {
        let vars = AgentVars::from_yaml_path(fp.clone(), false, false);
        let mut om = ObjectiveMaster::tsr_optimize(vars.robot.num_chains, vars.objective_mode.clone());
        if mode == 0 {
            om = ObjectiveMaster::standard_ik(vars.robot.num_chains);
        }

        let groove = OptimizationEngineOpen::new(vars.robot.num_dof.clone());
        let groove_nlopt = OptimizationEngineNLopt::new();

        Self{vars, om, groove, groove_nlopt}
    }

    pub fn from_loaded(mode: usize) -> Self {
        let path_to_src = get_path_to_config();
        let fp1 = path_to_src +  "/settings.yaml";
        let info_file_name = get_info_file_name(fp1);
        TSROptimization::from_info_file_name(info_file_name.clone(), mode.clone())
    }

    pub fn solve(&mut self) -> Vec<f64> {
        let mut out_x = self.vars.xopt.clone();

        let in_collision = self.vars.update_collision_world();
        if !in_collision {
            if self.vars.objective_mode == "ECAA" {
                self.om.tune_weight_priors(&self.vars);
            }
            self.groove.optimize(&mut out_x, &self.vars, &self.om, 100000);
            self.vars.update(out_x.clone());  
        }  
        out_x
    }

}